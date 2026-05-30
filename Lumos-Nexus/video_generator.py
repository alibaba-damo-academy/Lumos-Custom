#!/usr/bin/env python3
"""
OmniVideo Generator Class

This module contains the main VideoGenerator class that handles
model initialization and provides a unified interface for various
video/image generation tasks.
"""

import argparse
import gc
import logging
import os
import pickle as pkl
from pathlib import Path

import torch
import torch.distributed as dist

from nets.third_party.wan.configs import WAN_CONFIGS
from nets.third_party.wan.utils.utils import torch_checkpoint_load
from nets.omni.modules.vila_ar_model import VilaARVideoModel
from video_unified_gen_cascade import VideoX2XUnified


def _apply_cuda_inference_speed_defaults(args: argparse.Namespace) -> None:
    """
    Best-effort throughput knobs when weights live on GPU.
    Disable with --fast_cuda false or env OMNIVIDEO_FAST_CUDA=0.
    """
    if os.environ.get("OMNIVIDEO_FAST_CUDA", "1").lower() in ("0", "false", "no"):
        return
    if not getattr(args, "fast_cuda", True):
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass
    logging.info(
        "fast_cuda: cudnn.benchmark, TF32 (where supported), matmul_precision=high, "
        "SDPA flash/mem_efficient enabled"
    )


def _repo_root() -> str:
    """Repo root whether this file lives at repo root or under nets/omni/."""
    p = Path(__file__).resolve().parent
    if p.name == "omni" and p.parent.name == "nets":
        return str(p.parent.parent.parent)
    return str(p)

class VideoGenerator:
    """
    Main class for OmniVideo generation pipeline.
    
    Handles initialization of models and provides unified interface for
    various video/image generation tasks.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the Video generator.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.device = None
        self.rank = 0
        self.world_size = 1
        self.ar_model = None
        self.video_x2x = None
        self.special_tokens = None
        self.unconditioned_context = None
        
        # Fixed model root: absolute path (optional override via args.models_dir)
        project_root = _repo_root()
        default_models_dir = os.path.join(project_root, "model_ckpts")
        override = getattr(args, "models_dir", None)
        if override:
            self.models_dir = os.path.abspath(os.path.expanduser(override))
        else:
            self.models_dir = os.path.abspath(default_models_dir)
        # Fixed model paths
        self.model_paths = {
            'wan_ckpt': os.path.join(self.models_dir, "wan", "wanxiang1_3b"),
            'wan_ckpt_14b': os.path.join(self.models_dir, "Wan2.1-T2V-14B"),
            'adapter_ckpt': os.path.join(self.models_dir, "adapter", "model.pt"),
            'vision_head_ckpt': os.path.join(self.models_dir, "vision_head", "vision_head"),
            'transformer_ckpt': os.path.join(self.models_dir, "transformer", "model.pt"),
            'ar_model': os.path.join(self.models_dir, "ar_model", "checkpoint"),
            'unconditioned_context': os.path.join(self.models_dir, "unconditioned_context", "context.pkl"),
            'special_tokens': os.path.join(self.models_dir, "special_tokens", "tokens.pkl")
        }
        
        # Validate model directory exists
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}. Please run reorganize_models.sh first.")
            
        logging.info(f"Using models from: {self.models_dir}")
        
    def setup_distributed(self) -> None:
        """Setup distributed training environment if available."""
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = local_rank
        
        logging.info(f'Distributed setup: world_size={self.world_size}, rank={self.rank}, local_rank={local_rank}')

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size
        )

        _apply_cuda_inference_speed_defaults(self.args)

    def load_special_tokens(self) -> None:
        """Load special token embeddings from fixed path."""
        special_tokens_path = self.model_paths['special_tokens']
        
        if not os.path.exists(special_tokens_path):
            logging.warning(f"Special tokens file not found: {special_tokens_path}")
            return
            
        try:
            with open(special_tokens_path, 'rb') as f:
                self.special_tokens = pkl.load(f)
            
            if not isinstance(self.special_tokens, dict):
                raise ValueError("Special tokens should be a dictionary")
                
            # float32 on device — align Omni-Video1 (nets/omni/video_generator.py)
            precision_dtype = torch.float32
            for key, value in self.special_tokens.items():
                self.special_tokens[key] = value.to(precision_dtype).to(self.device)
                
            logging.info(f"Loaded special token embeddings: {list(self.special_tokens.keys())}")
            
        except Exception as e:
            logging.error(f"Failed to load special tokens: {e}")
            self.special_tokens = None
            
    def load_unconditioned_context(self) -> None:
        """Load unconditioned context from fixed path."""
        unconditioned_context_path = self.model_paths['unconditioned_context']
        
        if not os.path.exists(unconditioned_context_path):
            logging.warning(f"Unconditioned context file not found: {unconditioned_context_path}")
            return
            
        try:
            # float32 — align Omni-Video1
            precision_dtype = torch.float32
            pstate = pkl.load(open(unconditioned_context_path, 'rb'))
            
            # Validate required keys
            required_keys = ['text_emb_siglip2', 'text_emb', 'vlm_last_hidden_states']
            for key in required_keys:
                if key not in pstate:
                    raise KeyError(f"Key '{key}' not found in unconditioned context file")
            
            # Load embeddings with fallback adapter channels
            adapter_in_channels = getattr(self.args, 'adapter_in_channels', 1152)
            
            uncond_aligned_emb = pstate['text_emb_siglip2'][0].to(precision_dtype).to(self.device)
            if uncond_aligned_emb.dim() < 2:
                uncond_aligned_emb = uncond_aligned_emb.unsqueeze(0)

            # Validate dimensions
            if uncond_aligned_emb.shape[-1] != adapter_in_channels:
                logging.warning(
                    f"Unconditioned context dimension {uncond_aligned_emb.shape[-1]} "
                    f"doesn't match adapter input channels {adapter_in_channels}"
                )
            
            unconditioned_t5 = pstate['text_emb'][0].to(precision_dtype).to(self.device)
            if unconditioned_t5.dim() < 2:
                unconditioned_t5 = unconditioned_t5.unsqueeze(0)
            
            uncond_ar_vision = pstate['vlm_last_hidden_states'].to(precision_dtype).to(self.device)
            
            self.unconditioned_context = {
                'uncond_aligned_emb': uncond_aligned_emb,
                'uncond_context': unconditioned_t5,
                'uncond_ar_vision': uncond_ar_vision
            }
            
            logging.info(
                f"Loaded unconditioned context with shapes: "
                f"{uncond_aligned_emb.shape}, {unconditioned_t5.shape}, {uncond_ar_vision.shape}"
            )
            
        except Exception as e:
            logging.error(f"Failed to load unconditioned context: {e}")
            self.unconditioned_context = None
            
    def initialize_models(self) -> None:
        """Initialize OmniVideo pipeline first (peak VRAM), then AR/VLM (optional CPU init)."""
        ar_model_path = self.model_paths['ar_model']
        if not os.path.exists(ar_model_path):
            raise FileNotFoundError(f"AR model path not found: {ar_model_path}")

        ar_conv_mode = getattr(self.args, 'ar_conv_mode', 'llama_3')
        ar_num_frames = getattr(self.args, 'ar_model_num_video_frames', 8)
        ar_device = self.device

        cfg = WAN_CONFIGS.get('t2v-1.3B', {})  # Default config
        adapter_in_channels = getattr(self.args, 'adapter_in_channels', 1152)
        adapter_out_channels = getattr(self.args, 'adapter_out_channels', 4096)
        adapter_query_length = getattr(self.args, 'adapter_query_length', 256)
        use_visual_context_adapter = getattr(self.args, 'use_visual_context_adapter', True)
        visual_context_adapter_patch_size = getattr(self.args, 'visual_context_adapter_patch_size', (1, 4, 4))
        max_context_len = getattr(self.args, 'max_context_len', 2560)
        use_mixed_context = False
        wan_inference_activation_checkpointing = getattr(
            self.args, "wan_inference_activation_checkpoint", False
        )
        t5_cpu = getattr(self.args, "t5_cpu", True)

        want_fsdp = getattr(self.args, "dit_fsdp", False)
        dit_fsdp = bool(
            want_fsdp
            and self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if want_fsdp and not dit_fsdp:
            logging.warning(
                "dit_fsdp requested but requires torchrun with world_size>1 and initialized process group; "
                "running without DiT FSDP."
            )
        defer_dit_fsdp_wrap = dit_fsdp
        if dit_fsdp:
            logging.info(
                f"GPU {self.rank}: DiT FSDP enabled (defer wrap until after transformer ckpt); world_size={self.world_size}"
            )

        logging.info("Initializing OmniVideo pipeline (before AR to reduce load-time VRAM overlap)...")
        self.video_x2x = VideoX2XUnified(
            config=cfg,
            checkpoint_dir=self.model_paths['wan_ckpt'],
            checkpoint_dir_14b=self.model_paths['wan_ckpt_14b'],
            adapter_checkpoint_dir=self.model_paths['adapter_ckpt'],
            vision_head_ckpt_dir=self.model_paths['vision_head_ckpt'],
            adapter_in_channels=adapter_in_channels,
            adapter_out_channels=adapter_out_channels,
            adapter_query_length=adapter_query_length,
            device_id=self.device,
            rank=self.rank,
            dit_fsdp=dit_fsdp,
            defer_dit_fsdp_wrap=defer_dit_fsdp_wrap,
            use_visual_context_adapter=use_visual_context_adapter,
            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
            max_context_len=max_context_len,
            use_mixed_context=use_mixed_context,
            wan_inference_activation_checkpointing=wan_inference_activation_checkpointing,
            t5_cpu=t5_cpu,
        )

        transformer_path = self.model_paths['transformer_ckpt']
        if os.path.exists(transformer_path):
            self.load_transformer_checkpoint(transformer_path)

        if dit_fsdp:
            self.video_x2x.finalize_dit_fsdp()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Initializing AR model (device={self.device})...")
        self.ar_model = VilaARVideoModel(
            model_path=ar_model_path,
            conv_mode=ar_conv_mode,
            num_video_frames=ar_num_frames,
            device=ar_device,
        )

    def load_transformer_checkpoint(self, checkpoint_path: str) -> None:
        """Load transformer model checkpoint from fixed path."""
        try:
            logging.info(f"Loading transformer checkpoint from {checkpoint_path}")
            map_loc = torch.device(f"cuda:{int(self.device)}")
            state_dict = torch_checkpoint_load(checkpoint_path, map_location=map_loc)
            
            if 'module' in state_dict:
                state_dict = state_dict['module']
                
            self.video_x2x.model.load_state_dict(state_dict, strict=False)
            logging.info("Transformer checkpoint loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load transformer checkpoint: {e}")
            raise
