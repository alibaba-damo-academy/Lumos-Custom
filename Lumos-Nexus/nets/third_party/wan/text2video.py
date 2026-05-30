# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import torch_checkpoint_load
        
class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        new_checkpoint=None
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        if new_checkpoint:
            state_dict = torch_checkpoint_load(new_checkpoint, map_location='cpu')
            if 'module' in state_dict:
                state_dict = state_dict['module']
            self.model.load_state_dict(state_dict,strict=False)
            logging.info(f"load from {new_checkpoint} sucessfully!")
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        self.dit_fsdp = dit_fsdp
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
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
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

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

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

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

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

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
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.dit_fsdp:
                if self.rank == 0:
                    videos = self.vae.decode(x0)
            else:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0]

    def train_with_flow_matching_with_offline_feat(self,
                                                 train_dataloader,
                                                 optimizer,
                                                 num_epochs,
                                                 scheduler=None,
                                                 log_interval=100,
                                                 shift=3.0,
                                                 save_dir=None,
                                                 save_interval=100,
                                                 max_grad_norm=0.1):
        """
        Trains the text-to-video model using FlowMatchScheduler with offline pre-encoded features.
        
        Args:
            train_dataloader (DataLoader): DataLoader for training data with pre-encoded features
            optimizer (torch.optim.Optimizer): Model optimizer  
            num_epochs (int): Number of training epochs
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            log_interval (int): Steps between logging
            shift (float, optional): Noise schedule shift parameter
            save_dir (str, optional): Directory to save model checkpoints
            save_interval (int, optional): Epochs between saving model
            max_grad_norm (float, optional): Maximum norm for gradient clipping. Defaults to 0.1.
        """
        from .modules.schedulers.flow_match import FlowMatchScheduler
        
        self.model.train()
        
        # Initialize Flow Match scheduler for training
        flow_scheduler = FlowMatchScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=shift)
        
        # Set up timesteps for training
        flow_scheduler.set_timesteps(num_inference_steps=self.num_train_timesteps, training=True)
        
        # Initialize gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Track best model if needed
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for step, batch in enumerate(pbar):
                    # Get pre-encoded text embeddings, prompts, and video latents from batch
                    context = batch['text_emb']
                    prompts = batch['prompt']  # For logging purposes
                    videos = batch['latent_feature']
                    
                    # Move data to device
                    videos = videos.to(self.device)
                    context = [t.to(self.device) for t in context]
                    batch_size = videos.shape[0]
                    
                    # Calculate sequence length based on video dimensions
                    # Extract video dimensions
                    _, frames, height, width = videos.shape
                    
                    # Calculate sequence length based on patch size and dimensions
                    seq_len = math.ceil((height * width) / 
                                       (self.patch_size[1] * self.patch_size[2]) * 
                                       frames / self.sp_size) * self.sp_size
                    
                    # Sample timesteps uniformly
                    t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device)
                    
                    # Generate noise
                    noise = torch.randn_like(videos)
                    
                    # Add noise to target video using flow matching scheduler
                    noisy_video = flow_scheduler.add_noise(videos, noise, t)
                    
                    # Get training target (velocity field)
                    target = flow_scheduler.training_target(videos, noise, t)
                    
                    # Get training weights for current timesteps
                    weights = flow_scheduler.training_weight(t)
                    
                    # Predict velocity field
                    optimizer.zero_grad()
                    
                    # Use autocast for mixed precision training
                    with torch.cuda.amp.autocast(dtype=self.param_dtype):
                        # Model predicts the velocity field - properly pass required parameters
                        velocity_pred = self.model(
                            noisy_video, 
                            t=t, 
                            context=context, 
                            seq_len=seq_len
                        )
                        
                        # Flow matching loss: weighted MSE between predicted and target velocity
                        if weights.ndim > 0:  # If weights are per-sample
                            weights = weights.view(-1, 1, 1, 1, 1).to(self.device)
                            loss = torch.mean(weights * (velocity_pred - target) ** 2)
                        else:  # If weights are scalar
                            loss = torch.nn.functional.mse_loss(velocity_pred, target)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Update weights with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update learning rate scheduler if provided
                    if scheduler is not None:
                        scheduler.step()
                    
                    epoch_loss += loss.item()
                    
                    if step % log_interval == 0:
                        # Use the first prompt for logging example
                        example_prompt = prompts[0] if isinstance(prompts, list) else "N/A"
                        logging.info(f"Epoch {epoch+1}, Step {step}, Flow Matching Loss: {loss.item():.4f}, Example prompt: {example_prompt[:50]}...")
                    pbar.set_postfix({"loss": loss.item()})
                
                avg_loss = epoch_loss / len(train_dataloader)
                logging.info(f"Epoch {epoch+1} average flow matching loss: {avg_loss:.4f}")
                
                # Save model checkpoint
                if save_dir and (epoch + 1) % save_interval == 0 and self.rank == 0:
                    checkpoint_path = os.path.join(save_dir, f"flow_model_epoch_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                        'loss': avg_loss,
                    }, checkpoint_path)
                    logging.info(f"Flow matching model saved to {checkpoint_path}")
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_model_path = os.path.join(save_dir, "flow_model_best.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                            'loss': avg_loss,
                        }, best_model_path)
                        logging.info(f"Best model saved with loss: {avg_loss:.4f}")
            
        return self.model

    def test_flow_matching_with_offline_feat(self, 
                                           test_dir=None, 
                                           batch_size=2, 
                                           num_steps=5,
                                           shift=3.0):
        """
        Test the flow matching training with a dummy dataset.
        
        Args:
            test_dir (str, optional): Directory to save test data. If None, uses a temporary directory.
            batch_size (int, optional): Batch size for testing. Defaults to 2.
            num_steps (int, optional): Number of training steps to run. Defaults to 5.
            shift (float, optional): Noise schedule shift parameter. Defaults to 3.0.
            
        Returns:
            float: Average loss over test steps
        """
        import tempfile
        import shutil
        from .datasets.wan_t2v_dataset import create_dummy_dataset, create_wan_dataloader
        from .modules.schedulers.flow_match import FlowMatchScheduler
        
        # Create temporary directory if test_dir is not provided
        if test_dir is None:
            test_dir = tempfile.mkdtemp()
            cleanup = True
        else:
            os.makedirs(test_dir, exist_ok=True)
            cleanup = False
        
        try:
            # Set model to training mode
            self.model.train()
            
            # Create dummy dataset with appropriate dimensions
            # Use smaller dimensions for testing
            frames = 5
            height = 16
            width = 16
            
            logging.info(f"Creating dummy dataset in {test_dir}")
            file_paths = create_dummy_dataset(
                test_dir, 
                num_samples=batch_size*num_steps,
                frames=frames,
                height=height,
                width=width
            )
            
            # Create dataloader
            dataloader = create_wan_dataloader(
                file_paths, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
            
            # Create optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
            
            # Initialize Flow Match scheduler
            flow_scheduler = FlowMatchScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=shift)
            
            # Set up timesteps for training
            flow_scheduler.set_timesteps(num_inference_steps=self.num_train_timesteps, training=True)
            
            # Run a few training steps
            total_loss = 0
            with tqdm(enumerate(dataloader), total=num_steps, desc="Testing flow matching") as pbar:
                for step, batch in pbar:
                    if step >= num_steps:
                        break
                        
                    # Get data from batch
                    context = batch['text_emb']
                    videos = batch['latent_feature']
                    
                    # Move data to device
                    videos = videos.to(self.device)
                    context = [t.to(self.device) for t in context]
                    batch_size = videos.shape[0]
                    
                    # Calculate sequence length
                    _, frames, height, width = videos.shape
                    seq_len = math.ceil((height * width) / 
                                       (self.patch_size[1] * self.patch_size[2]) * 
                                       frames / self.sp_size) * self.sp_size
                    
                    # Sample timesteps uniformly
                    t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device)
                    
                    # Generate noise
                    noise = torch.randn_like(videos)
                    
                    # Add noise to target video using flow matching scheduler
                    noisy_video = flow_scheduler.add_noise(videos, noise, t)
                    
                    # Get training target (velocity field)
                    target = flow_scheduler.training_target(videos, noise, t)
                    
                    # Get training weights for current timesteps
                    weights = flow_scheduler.training_weight(t)
                    
                    # Predict velocity field
                    optimizer.zero_grad()
                    
                    with amp.autocast(dtype=self.param_dtype):
                        # Model predicts the velocity field
                        velocity_pred = self.model(
                            noisy_video, 
                            t=t, 
                            context=context, 
                            seq_len=seq_len
                        )
                        
                        # Flow matching loss
                        if weights.ndim > 0:
                            weights = weights.view(-1, 1, 1, 1, 1).to(self.device)
                            loss = torch.mean(weights * (velocity_pred - target) ** 2)
                        else:
                            loss = torch.nn.functional.mse_loss(velocity_pred, target)
                    
                    # Backward pass (optional for testing)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / num_steps
            logging.info(f"Test completed with average loss: {avg_loss:.4f}")
            return avg_loss
            
        except Exception as e:
            logging.error(f"Test failed with error: {e}")
            raise
            
        finally:
            # Clean up temporary directory
            if cleanup:
                shutil.rmtree(test_dir)
                logging.info(f"Cleaned up test directory: {test_dir}")