import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from nets.third_party.wan.distributed.fsdp import shard_model, shard_wan_dit_for_fsdp
from nets.third_party.wan.modules.model import WanModel
from nets.third_party.wan.modules.t5 import T5EncoderModel
from nets.third_party.wan.modules.vae import WanVAE
from nets.third_party.wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from nets.third_party.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from nets.third_party.wan.utils.utils import safe_distributed_barrier
from nets.omni.modules.video_model import VideoMixedConditionModel


def rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))


def gaussian_blur(tensor, k=3, sigma=1.0):
    """2D Gaussian blur on [C, T, H, W]."""
    if k % 2 == 0:
        k += 1

    coords = torch.arange(k, dtype=tensor.dtype, device=tensor.device)
    coords = coords - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.view(1, 1, k, 1) * g.view(1, 1, 1, k)

    C, T, H, W = tensor.shape
    tensor_reshaped = tensor.view(C * T, 1, H, W)
    padding = k // 2
    blurred = torch.nn.functional.conv2d(tensor_reshaped, kernel, padding=padding, groups=1)
    return blurred.view(C, T, H, W)

class VideoX2XUnified:

    def __init__(
        self,
        config,
        checkpoint_dir,
        checkpoint_dir_14b,
        adapter_checkpoint_dir,
        vision_head_ckpt_dir,
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
        use_visual_context_adapter=False,
        visual_context_adapter_patch_size=(1,4,4),
        max_context_len=None,
        use_mixed_context=False,
        wan_inference_activation_checkpointing=False,
        defer_dit_fsdp_wrap=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model with adapter components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            checkpoint_dir_14b (`str`):
                Path to directory containing 14B model checkpoints
            adapter_checkpoint_dir (`str`):
                Path to directory containing adapter checkpoints
            adapter_in_channels (`int`, *optional*, defaults to 1152):
                Input channels for the adapter
            adapter_out_channels (`int`, *optional*, defaults to 4096):
                Output channels for the adapter
            adapter_query_length (`int`, *optional*, defaults to 64):
                Query length for the adapter
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP FULL_SHARD on both DiT subtrees. If ``config.param_dtype`` is
                ``torch.float32``, FSDP uses no MixedPrecision (fp32 + sharding only); otherwise
                1.3B stays fp32 in MixedPrecision and 14B uses ``param_dtype``.
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to True):
                Same as Wan2.1-T2V ``generate.py --t5_cpu``: keep T5 weights on CPU, encode on CPU,
                then move text embeddings to GPU. Set False to run T5 on GPU (faster, more VRAM).
                Incompatible with ``t5_fsdp`` (use one or the other).
            wan_inference_activation_checkpointing (`bool`, *optional*, defaults to False):
                If True, use activation checkpointing on Wan DiT blocks during inference (more compute, lower peak activations).
            defer_dit_fsdp_wrap (`bool`, *optional*, defaults to False):
                If True with dit_fsdp, skip FSDP in __init__; caller must call finalize_dit_fsdp() after load_state_dict.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self._defer_dit_fsdp_wrap = bool(defer_dit_fsdp_wrap and dit_fsdp)
        self._dit_fsdp_applied = False
        self._cuda_device_id = device_id

        _stagger = float(os.environ.get("WAN_LOAD_STAGGER_SEC", "0") or "0")
        if _stagger > 0.0 and dist.is_initialized():
            delay = self.rank * _stagger
            if delay > 0:
                logging.info(
                    "GPU %s: WAN_LOAD_STAGGER_SEC=%s, sleeping %.1fs before checkpoint I/O",
                    self.rank,
                    _stagger,
                    delay,
                )
                time.sleep(delay)

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        # Wan2.1-T2V: weights loaded on CPU; with t5_cpu=True, generate() encodes on CPU then moves embeddings to GPU.
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
        logging.info(
            "T5 UMT5: %s (same idea as Wan2.1 ``--t5_cpu``)",
            "CPU encode each call, embeddings to GPU" if t5_cpu else "moved to GPU for encode (t5_cpu=False)",
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VideoMixedConditionModel from {checkpoint_dir} and {adapter_checkpoint_dir}")
        # Initialize the VideoMixedConditionModel
        self.model = VideoMixedConditionModel.from_pretrained(
                        wan_ckpt_dir=checkpoint_dir,
                        wan_ckpt_dir_14b=checkpoint_dir_14b,
                        adapter_ckpt_dir=adapter_checkpoint_dir,
                        vision_head_ckpt_dir=vision_head_ckpt_dir, 
                        adapter_in_channels=adapter_in_channels,
                        adapter_out_channels=adapter_out_channels,
                        adapter_query_length=adapter_query_length,
                        precision_dtype=self.param_dtype,
                        device_id=device_id,
                        rank=rank,
                        dit_fsdp=dit_fsdp,
                        use_usp=use_usp,
                        use_visual_context_adapter=use_visual_context_adapter,
                        visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                        max_context_len=max_context_len,
                        use_mixed_context=use_mixed_context,
                    )
        
        self.model.enable_eval()

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from nets.third_party.wan.distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.wan_model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.wan_2_model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.wan_model.forward = types.MethodType(usp_dit_forward, self.model.wan_model)
            self.model.wan_2_model.forward = types.MethodType(usp_dit_forward, self.model.wan_2_model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            try:
                safe_distributed_barrier(timeout_minutes=30)
            except Exception as e:
                logging.error(f"GPU {self.rank}: Initial barrier synchronization failed: {e}")

        # Align Omni-Video1 inference path: place model on GPU but do not force-cast
        # the whole module via .to(dtype=...). Dtype is controlled by weights/autocast.
        self.model.to(self.device)

        if wan_inference_activation_checkpointing:
            self.model.set_inference_activation_checkpointing(True)
            logging.info(
                "Wan inference activation checkpointing enabled on 1.3B and 14B DiT blocks"
            )

        if dit_fsdp and not self._defer_dit_fsdp_wrap:
            self._apply_dit_fsdp()
        elif dit_fsdp and self._defer_dit_fsdp_wrap:
            logging.info(
                "dit_fsdp: FSDP wrap deferred — call finalize_dit_fsdp() after load_state_dict"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Model precision: {self.param_dtype}")
        try:
            logging.info(
                f"First model parameter dtype: {next(self.model.parameters()).dtype}"
            )
        except Exception:
            logging.info("First model parameter dtype: (unavailable under FSDP)")

        self.sample_neg_prompt = config.sample_neg_prompt

    def _apply_dit_fsdp(self):
        if self._dit_fsdp_applied:
            return

        m = self.model
        m.wan_model = shard_wan_dit_for_fsdp(
            m.wan_model,
            self._cuda_device_id,
        )
        m.wan_2_model = shard_wan_dit_for_fsdp(
            m.wan_2_model,
            self._cuda_device_id,
        )
        setattr(m, "_dit_fsdp_wrapped", True)
        m.wan_model.eval()
        m.wan_2_model.eval()
        self._dit_fsdp_applied = True
        logging.info(
            "GPU %s: FSDP FULL_SHARD applied to wan_model and wan_2_model",
            self.rank,
        )

    def finalize_dit_fsdp(self) -> None:
        """Apply DiT FSDP after full state is loaded (e.g. transformer ckpt). No-op if not dit_fsdp or already applied."""
        if not self.dit_fsdp or not self._defer_dit_fsdp_wrap:
            return
        if self._dit_fsdp_applied:
            return
        self._apply_dit_fsdp()

    def generate(self,
                 input_prompt,
                 aligned_emb=None,
                 ar_vision_input=None,
                 visual_emb=None,
                 ref_images=None,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=False,
                 special_tokens=None,
                 classifier_free_ratio=0.0,
                 unconditioned_context=None,
                 condition_mode="auto",
                 use_visual_as_input=False,
                 gamma_w=0.5,
                 gamma_hf=0.7,
                 sigma_min=0.35,
                 sigma_max=0.7):
        r"""
        Generates video frames from text prompt using diffusion process with adapter.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            aligned_emb (`torch.Tensor`, *optional*, defaults to None):
                Aligned embedding for the adapter
            visual_emb (`torch.Tensor`, *optional*, defaults to None):
                Visual embedding for the visual context adapter
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to False):
                If True, moves DiT to CPU after each generate() to save VRAM (slow for batch:
                next call pays full GPU weight transfer again). Keep False when VRAM allows.
            special_tokens (`dict`, *optional*, defaults to None):
                Dictionary of special token embeddings
            classifier_free_ratio (`float`, *optional*, defaults to 0.0):
                Ratio for classifier-free guidance during training
            unconditioned_context (`torch.Tensor`, *optional*, defaults to None):
                Unconditioned context for classifier-free guidance
            condition_mode (`str`, *optional*, defaults to "auto"):
                Mode for conditioning, options:
                - "auto": Automatically determine based on inputs (default)
                - "full": Use context + visual_emb + aligned_emb
                - "aligned_emb_with_text": Use aligned_emb + context
                - "aligned_emb_only": Use aligned_emb only
                - "visual_with_aligned_emb": Use visual_emb + aligned_emb
                - "text_only": Use context only
            use_visual_as_input (`bool`, *optional*, defaults to False):
                Whether to use visual embedding as part of the input
            gamma_w (`float`, *optional*, defaults to 0.5):
                Exponent for temporal gating ``w_t`` in cascade denoise (see ``math.pow(1.0 - tau, gamma_w)``).
            gamma_hf (`float`, *optional*, defaults to 0.7):
                Scale applied to the 1.3B high-frequency branch in the LF/HF fusion (``HF = ... + w_t * (gamma_hf * HF13)``).
            sigma_min (`float`, *optional*, defaults to 0.35):
                Minimum Gaussian blur ``sigma`` for LF/HF split (when ``w_t`` is low).
            sigma_max (`float`, *optional*, defaults to 0.7):
                Maximum Gaussian blur ``sigma`` for LF/HF split (when ``w_t`` is high).

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # T5 encode path — align Omni-Video1 omni_video_unified_gen.generate
        # Runs after outer "Starting generation..." log; UMT5-xxl is slow on CPU (no tqdm here).
        if not self.t5_cpu:
            logging.info("T5: encoding on GPU (prompt + negative prompt)...")
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
            logging.info("T5: encode done.")
        else:
            logging.info(
                "T5: encoding on CPU — UMT5-xxl forward x2 (pos + neg); often tens of seconds; then .to(GPU)"
            )
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            logging.info("T5: encode done, embeddings on GPU.")

        if ar_vision_input is not None: 
            aligned_emb = None # make sure aligned_emb is not used when ar_vision_input is provided
        # Process aligned embedding if provided
        if aligned_emb is not None and not isinstance(aligned_emb, torch.Tensor):
            aligned_emb = torch.tensor(aligned_emb, dtype=torch.float32, device=self.device)
        
        if aligned_emb is not None and aligned_emb.dim() == 1:
            # Add batch dimension if needed
            aligned_emb = aligned_emb.unsqueeze(0)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode (skip autocast when fp32 — avoids unintended dtype promotion)
        _infer_ctx = (
            nullcontext()
            if self.param_dtype == torch.float32
            else amp.autocast(dtype=self.param_dtype)
        )
        with _infer_ctx, torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            ar_vision_input_null = None
            aligned_emb_null = None
            visual_emb_null = None
            ref_images_null = None
            if not condition_mode == 'text_only':
                assert type(unconditioned_context) is dict, "unconditioned_context must be provided as a dict if condition_mode is not 'text_only'"

                if ar_vision_input is not None and 'uncond_ar_vision' in unconditioned_context:
                    if isinstance(ar_vision_input, list):
                        ar_vision_input_null = [unconditioned_context['uncond_ar_vision'] for _ in range(len(ar_vision_input))]
                    else:
                        ar_vision_input_null = unconditioned_context['uncond_ar_vision']

                if context is not None and isinstance(context, list):
                    context_null = [unconditioned_context['uncond_context'] for c in context]
                elif context is not None and isinstance(context, torch.Tensor):
                    context_null = unconditioned_context['uncond_context']
                    if context_null is not None and context_null.dim() < context.dim():
                        context_null = context_null.unsqueeze(0)
                else:
                    context_null = context_null # use t5 null context

                if visual_emb is not None:
                    if isinstance(visual_emb, list):
                        visual_emb_null = [torch.zeros_like(ve) for ve in visual_emb]
                    else:
                        visual_emb_null = torch.zeros_like(visual_emb) if isinstance(visual_emb, torch.Tensor) else None
            
            arg_c = {
                'context': context, 
                'seq_len': seq_len,
                'aligned_emb': aligned_emb,
                'ar_vision_input': ar_vision_input,
                'visual_emb': visual_emb,
                'ref_images': ref_images,
                'special_token_dict': special_tokens,
                'classifier_free_ratio': 0.0,  # During inference, we don't use random dropout
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }
               
            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
                'aligned_emb': aligned_emb_null,
                'ar_vision_input': ar_vision_input_null,
                'visual_emb': visual_emb,
                'ref_images': ref_images,
                'special_token_dict': special_tokens,
                'classifier_free_ratio': 0.0,
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }

            # Ensure DiT on GPU once per generate(); FSDP roots must not be blindly .to()'d.
            if not self.dit_fsdp:
                logging.info("Moving OmniVideo DiT / adapter stack to GPU (first call or after offload can be slow)...")
                self.model.to(self.device)

            logging.info("=== Cascade denoise: 1.3B + 14B per timestep ===")
            latents_ = noise.copy()
            for step_idx, t in enumerate(tqdm(timesteps, desc="1.3B Model")):
                    if use_visual_as_input and visual_emb is not None:
                        assert len(latents_) == 1, "currently only support one latent at a time"
                        latent_model_input = [latents_[0] + visual_emb]
                    else:
                        latent_model_input = latents_
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    noise_pred_cond_1_3 = self.model(
                        latent_model_input, t=timestep, use_wan_14b=False, **arg_c)[0]
                    noise_pred_uncond_1_3 = self.model(latent_model_input, t=timestep, use_wan_14b=False, **arg_null)[0]
                    noise_pred_1_3 = noise_pred_cond_1_3 + guide_scale * (noise_pred_cond_1_3 - noise_pred_uncond_1_3)

                    noise_pred_cond_14 = self.model(latent_model_input, t=timestep, use_wan_14b=True, **arg_c)[0]
                    noise_pred_uncond_14 = self.model(latent_model_input, t=timestep, use_wan_14b=True, **arg_null)[0]
                    noise_pred_14 = noise_pred_cond_14 + guide_scale * (noise_pred_cond_14 - noise_pred_uncond_14)

                    tau = t / (self.num_train_timesteps - 1)
                    w_t = 0.5 * (1.0 + math.cos(math.pi * math.pow(1.0 - tau, gamma_w)))

                    H, W = latents_[0].shape[-2], latents_[0].shape[-1]
                    scale = (H * W) ** 0.5 / 1024.0
                    k = max(3, int(2 * int(scale) + 1))
                    sigma = sigma_min + (sigma_max - sigma_min) * w_t
                    LF13 = gaussian_blur(noise_pred_1_3, k=k, sigma=sigma)
                    LF14 = gaussian_blur(noise_pred_14, k=k, sigma=sigma)
                    HF13 = noise_pred_1_3 - LF13
                    HF14 = noise_pred_14 - LF14

                    LF = w_t * LF13 + (1.0 - w_t) * LF14
                    HF = (1.0 - w_t) * HF14 + w_t * (gamma_hf * HF13)

                    noise_pred = LF + HF

                    eps = 1e-6
                    r = rms(noise_pred_1_3) / (rms(noise_pred_14) + eps)
                    noise_pred_14 = noise_pred_14 * r

                    rms_tar = 0.5 * (rms(noise_pred_1_3) + rms(noise_pred_14))
                    noise_pred = noise_pred * (rms_tar / (rms(noise_pred) + eps))

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents_[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents_= [temp_x0.squeeze(0)]

            logging.info(f"Fused model completed. Final latent shape: {latents_[0].shape}")
            x0 = latents_
            if offload_model and not self.dit_fsdp:
                self.model.cpu()
                torch.cuda.empty_cache()
            elif offload_model and self.dit_fsdp:
                logging.warning("offload_model is ignored when dit_fsdp is enabled (FSDP cannot .cpu() root)")
            videos = self.vae.decode(x0)

        del noise, latents_
        del sample_scheduler

        if offload_model and not self.dit_fsdp:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            try:
                safe_distributed_barrier(timeout_minutes=30)
            except Exception as e:
                logging.error(f"GPU {self.rank}: Final barrier synchronization failed: {e}")

        return videos[0]
