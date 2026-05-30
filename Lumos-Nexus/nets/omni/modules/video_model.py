import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import einops
from typing import List, Dict, Optional, Union, Tuple
import random

from nets.third_party.wan.modules.model import WanModel
from nets.omni.modules.adapter import DM_Adapter
from nets.omni.modules.visual_context_adapter import VisualContextAdapter
from nets.third_party.llava.model.vila_with_vision_head import VisionHead


def _wan_from_pretrained(path: Union[str, WanModel]) -> WanModel:
    if isinstance(path, WanModel):
        return path
    try:
        return WanModel.from_pretrained(path, low_cpu_mem_usage=True)
    except TypeError:
        return WanModel.from_pretrained(path)


class VideoMixedConditionModel(nn.Module):
    """    
    This model integrates the adapter's output with the text embeddings
    for the WanModel, allowing for a unified forward pass. It also integrates the visual embeddings and reference images with the text embeddings.
    """
    
    def __init__(
        self,
        wan_model_or_ckpt_dir: Union[WanModel, str],
        wan_model_or_ckpt_dir_14b: Union[WanModel, str],
        adapter_or_ckpt_dir: Union[DM_Adapter, str],
        vision_head_or_ckpt_dir: Union[VisionHead, str],
        llm_hidden_size: int = 4096,
        learnable_query_length: int = 4,
        adapter_in_channels: int = 1152,
        adapter_out_channels: int = 4096,
        adapter_query_length: int = 64,
        precision_dtype: torch.dtype = torch.float32,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len = None, 
        eps: float = 1e-6,
        use_mixed_context: bool = False,
    ):
        """
        Initialize the combined model with WanModel and DM_Adapter.
        
        Args:
            wan_model_or_ckpt_dir: Either a WanModel instance or a path to a checkpoint directory
            adapter_or_ckpt_dir: Either a DM_Adapter instance or a path to a checkpoint directory
            adapter_in_channels: Input channels for adapter (used only if adapter_or_ckpt_dir is a path)
            adapter_out_channels: Output channels for adapter (used only if adapter_or_ckpt_dir is a path)
            adapter_query_length: Query length for adapter (used only if adapter_or_ckpt_dir is a path)
            precision_dtype: Precision type for computation
            device_id: GPU device ID
            rank: Process rank
            dit_fsdp: Whether to use FSDP for DiT model
            use_usp: Whether to use USP
        """
        super().__init__()
        
        # Load WanModel from checkpoint directory - now using original 16 channels
        self.wan_model = _wan_from_pretrained(wan_model_or_ckpt_dir)
        self.precision_dtype = precision_dtype
        self.wan_2_model = _wan_from_pretrained(wan_model_or_ckpt_dir_14b)
        # Keep 14B branch at checkpoint/original dtype (do not force-cast to precision_dtype).


        self.use_mixed_context = use_mixed_context
        
        if max_context_len is not None:
            self.wan_model.text_len = max_context_len
            self.wan_2_model.text_len = 512
        self.max_context_len = self.wan_model.text_len

        # Handle DM_Adapter initialization
        self.adapter = DM_Adapter(
            in_channels=adapter_in_channels,
            out_channels=adapter_out_channels,
            learnable_query_length=adapter_query_length,
            TRAINABLE_PRECISION=precision_dtype,
            device_id=device_id,
            rank=rank,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            load_ckpt_dir=adapter_or_ckpt_dir
        )
        
        if adapter_or_ckpt_dir is not None and os.path.exists(adapter_or_ckpt_dir):
            logging.info(f"Loading adapter from {adapter_or_ckpt_dir}")
            self.adapter.load_checkpoint(adapter_or_ckpt_dir)
        
        # AR Vision Head
        self.ar_vision_head = VisionHead(llm_hidden_size = llm_hidden_size, learnable_query_length=learnable_query_length)
        if vision_head_or_ckpt_dir is not None and os.path.exists(vision_head_or_ckpt_dir):
            logging.info(f"Loading AR Vision Head from {vision_head_or_ckpt_dir}")
            _vh_map = (
                torch.device(f"cuda:{device_id}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.ar_vision_head.load_checkpoint(vision_head_or_ckpt_dir, map_location=_vh_map)
       

        # Visual Context Adapter
        if use_visual_context_adapter:
            self.visual_context_adapter = VisualContextAdapter(
                patch_size=self.wan_model.patch_size if visual_context_adapter_patch_size is None else visual_context_adapter_patch_size ,
                in_channels=self.wan_model.in_dim,
                hidden_dim=self.wan_model.dim,
                out_dim=self.wan_model.text_dim,
                eps=eps,
            )
        else:
            self.visual_context_adapter = None
    @classmethod
    def from_pretrained(
        cls,
        wan_ckpt_dir: str,
        wan_ckpt_dir_14b: str,
        adapter_ckpt_dir: str,
        vision_head_ckpt_dir: str, 
        learnable_query_length: int = 4,
        adapter_in_channels: int = 1152,
        adapter_out_channels: int = 4096,
        adapter_query_length: int = 64,
        precision_dtype: torch.dtype = torch.float32,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len = None, 
        eps: float = 1e-6,
        use_mixed_context: bool = False,
    ):
        """
        Create a WanWithAdapterModel from pretrained checkpoints.
        
        This is a convenience method that calls the constructor with checkpoint directories.
        
        Args:
            wan_ckpt_dir: Directory containing WanModel checkpoint
            adapter_ckpt_dir: Directory containing DM_Adapter checkpoint
            adapter_in_channels: Input channels for adapter
            adapter_out_channels: Output channels for adapter
            adapter_query_length: Query length for adapter
            precision_dtype: Precision type for computation
            device_id: GPU device ID
            rank: Process rank
            dit_fsdp: Whether to use FSDP for DiT model
            use_usp: Whether to use USP
            
        Returns:
            WanWithAdapterModel: The combined model
        """
        return cls(
            wan_model_or_ckpt_dir=wan_ckpt_dir,
            wan_model_or_ckpt_dir_14b=wan_ckpt_dir_14b,
            adapter_or_ckpt_dir=adapter_ckpt_dir,
            vision_head_or_ckpt_dir=vision_head_ckpt_dir,
            learnable_query_length=learnable_query_length,
            adapter_in_channels=adapter_in_channels,
            adapter_out_channels=adapter_out_channels,
            adapter_query_length=adapter_query_length,
            precision_dtype=precision_dtype,
            device_id=device_id,
            rank=rank,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            use_visual_context_adapter=use_visual_context_adapter,
            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
            max_context_len=max_context_len,
            eps=eps,
            use_mixed_context=use_mixed_context,
        )
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for both models"""
        self.wan_model.enable_gradient_checkpointing()
        self.wan_2_model.enable_gradient_checkpointing()
        # Add gradient checkpointing for adapter if it supports it
        if hasattr(self.adapter, 'enable_gradient_checkpointing'):
            self.adapter.enable_gradient_checkpointing()
        if hasattr(self.ar_vision_head, 'enable_gradient_checkpointing'): #TODO: check if this is correct
            self.ar_vision_head.enable_gradient_checkpointing()

    def set_inference_activation_checkpointing(self, enabled: bool = True):
        """Enable block-wise activation checkpointing on both Wan DiTs during eval (slower, less peak VRAM)."""
        self.wan_model.enable_inference_activation_checkpointing(enabled)
        self.wan_2_model.enable_inference_activation_checkpointing(enabled)

    def reset_wan_text_len(self, new_len: int):
        """Reset the text_len attribute of the WanModel"""
        self.wan_model.text_len = new_len
        # self.wan_2_model.text_len = new_len
    
    def forward(
        self, 
        x: list[torch.Tensor], 
        t: torch.Tensor, 
        context: List[torch.Tensor] = None, 
        aligned_emb: Optional[torch.Tensor] = None,
        ar_vision_input: Optional[List[torch.Tensor]] = None,
        visual_emb: Optional[torch.Tensor] = None,
        ref_images: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        special_token_dict: Optional[Dict[str, torch.Tensor]] = None,
        classifier_free_ratio: Optional[float] = None,
        unconditioned_context: Optional[dict[torch.Tensor]] = None,
        condition_mode: str = "full",
        use_wan_14b: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass that integrates the adapter into the WanModel pipeline.
        
        Args:
            x: Input tensor for WanModel  (List[Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
            t: Timestep tensor
            context: List of context tensors (text embeddings) or single tensor
            aligned_emb: Aligned embeddings for the adapter (can be list or tensor)
            ar_vision_input: AR vision input (can be list or tensor)
            visual_emb: Visual embeddings (can be list or tensor)
            ref_images: Reference images (can be list or tensor)
            seq_len: Sequence length for WanModel
            special_token_dict: Dictionary of special tokens for embedding concatenation
            classifier_free_ratio: Ratio for classifier-free guidance
            unconditioned_context: Unconditioned context for classifier-free guidance
            condition_mode: Mode for conditioning, options:
                - "auto": Automatically determine based on inputs (default)
                - "full": Universal mode that handles all task types with available inputs
            use_wan_14b: If True, run the 14B DiT branch; if False, the 1.3B branch (cascade uses both).
                Context: 1.3B uses ``mixed_context`` (T5 + adapter/AR fusion); 14B uses ``context_t5_only``.
            
        Returns:
            Output adapter-processed embeddings
        """
        # Process based on condition mode
        if condition_mode == "full":
            # Universal full mode - handles all task types with available inputs
            # Validate required inputs
            assert x is not None, "Full mode requires at least x"
            batch_size = x.size(0) if isinstance(x, torch.Tensor) else len(x)

            for idx in range(batch_size):
                # Apply classifier-free guidance if needed
                if classifier_free_ratio is not None and classifier_free_ratio > 0 and random.random() < classifier_free_ratio:
                    if unconditioned_context is None:
                        raise ValueError("unconditioned_context must be provided when classifier_free_ratio > 0")

                    # Apply unconditioning based on format of unconditioned_context
                    # the unconditioned context is feature named "uncond_context"
                    # the unconditioned ar_vision_head_output is another feature named "uncond_ar_vision", the tensor shape is [1, 164, 4096]
                    if isinstance(unconditioned_context, dict):
                        # For dictionary format (multiple unconditioned elements)
                        if ar_vision_input is not None and ar_vision_input[idx] is not None and 'uncond_ar_vision' in unconditioned_context:
                            ar_vision_input[idx] = unconditioned_context['uncond_ar_vision']

                        if context is not None and context[idx] is not None and 'uncond_context' in unconditioned_context:
                            context[idx] = unconditioned_context['uncond_context']
                
            # Process AR Vision Head and adapter
            ar_vision_head_output = None if ar_vision_input is None else [None] * batch_size
            adapter_output = None if ar_vision_input is None else [None] * batch_size
            if self.ar_vision_head is not None and ar_vision_input is not None:
                if isinstance(ar_vision_input, list): # do padding for aligned_emb to the longest length
                    # ar_vision_head_output = [self.ar_vision_head(ar_vision_input[idx]) if ar_vision_input[idx] is not None else None for idx in range(batch_size)]
                    for idx in range(batch_size):
                        if ar_vision_input[idx] is not None:
                            ar_vision_head_output[idx] = self.ar_vision_head(ar_vision_input[idx])
                            if x[idx].size(1) == 1:
                                ar_vision_head_output[idx] = ar_vision_head_output[idx][:, 0:1, :] # [bs, 1, 1152]
                                adapter_output[idx] = self.adapter(ar_vision_head_output[idx])
                            else:
                                # for v2v tasks, the ar_vision_head_output is [1, 4, 1152]
                                adapter_output[idx] = self.adapter(ar_vision_head_output[idx].reshape(-1, 1, ar_vision_head_output[idx].size(2))) # [1 * 4, 1, 1152] -> [1 * 4, 256, 4096]
                                adapter_output[idx] = einops.rearrange(adapter_output[idx], '(b n) c d -> b (n c) d', b=ar_vision_head_output[idx].size(0)) #  [1*4, 256, 4096] -> [1, 4*256, 4096]
                        else:
                            adapter_output[idx] = None
                else:
                    ar_vision_head_output = self.ar_vision_head(ar_vision_input) # [bs, 4, 1152] or [bs, 1, 1152]
                    if x[0].size(1) == 1:
                        ar_vision_head_output = ar_vision_head_output[:, 0:1, :] # [bs, 1, 1152]
                        adapter_output = self.adapter(ar_vision_head_output)
                    else:
                        adapter_output = self.adapter(ar_vision_head_output.reshape(-1, 1, ar_vision_head_output.size(2))) # [bs * 4, 1, 1152] -> [bs * 4, 256, 4096]
                        adapter_output = einops.rearrange(adapter_output, '(b n) c d -> b (n c) d', b=ar_vision_head_output.size(0)) #  [bs*4, 256, 4096] -> [bs, 4*256, 4096]
            
            # import pdb; pdb.set_trace();
            # Process visual embeddings and reference images if available
            processed_visual_emb = None if visual_emb is None else [None] * batch_size
            processed_ref_images = None if ref_images is None else [None] * batch_size
            
            if visual_emb is not None and self.visual_context_adapter is not None:
                if isinstance(visual_emb, list):
                    for idx in range(batch_size):
                        if visual_emb[idx] is not None:
                            processed_visual_emb[idx] = self.visual_context_adapter(visual_emb[idx]).squeeze(0)
                        else:
                            processed_visual_emb[idx] = None
                else:
                    processed_visual_emb = self.visual_context_adapter(visual_emb)
            
            if ref_images is not None and self.visual_context_adapter is not None:
                if isinstance(ref_images, list):
                    for idx in range(batch_size):
                        if ref_images[idx] is not None:
                            processed_ref_images[idx] = self.visual_context_adapter(ref_images[idx]).squeeze(0)
                        else:
                            processed_ref_images[idx] = None
                else:
                    processed_ref_images = self.visual_context_adapter(ref_images)
                    
            
            # Create a new context list with all available components
            mixed_context = []
            context_t5_only = []
            # import pdb; pdb.set_trace();
            for idx in range(batch_size):
                # Extract all available components for this batch item
                components = []
                context_t5_components = []
                
                # Get adapter output for this batch item if available
                adapter_item = None
                if adapter_output is not None:
                    if isinstance(adapter_output, list):
                        if adapter_output[idx].dim() == 3:
                            adapter_item = adapter_output[idx][0]
                        else:
                            adapter_item = adapter_output[idx]
                    elif adapter_output.dim() == 3:
                        adapter_item = adapter_output[idx]
                    else:
                        adapter_item = adapter_output
                
                # Get context item if available
                context_item = None
                if context is not None:
                    if isinstance(context, list):
                        if context[idx].dim() == 3:
                            context_item = context[idx][0]
                        else:
                            context_item = context[idx]
                    elif context.dim() > 2:
                        context_item = context[idx]
                    else:
                        context_item = context
                
                # Get visual embeddings if available
                visual_item = None
                if processed_visual_emb is not None:
                    visual_item = processed_visual_emb[idx]
                
                # Get reference images if available
                ref_item = None
                if processed_ref_images is not None:
                    ref_item = processed_ref_images[idx]
                
                # Prepare context based on available inputs and special tokens
                if special_token_dict is not None:
                    # Use special tokens if available
                    
                    # Start with visual embeddings (source image/video)
                    if visual_item is not None and '<img_st>' in special_token_dict and '<img_ed>' in special_token_dict:
                        components.extend([
                            special_token_dict['<img_st>'],
                            visual_item,
                            special_token_dict['<img_ed>']
                        ])
                    
                    # Add reference images if available
                    if ref_item is not None and '<img_st>' in special_token_dict and '<img_ed>' in special_token_dict:
                        # Fall back to image tokens if reference tokens aren't available
                        components.extend([
                            special_token_dict['<img_st>'],
                            ref_item,
                            special_token_dict['<img_ed>']
                        ])
                    
                    # Add adapter output (aligned embeddings)
                    if adapter_item is not None:
                        if '<ipl_st>' in special_token_dict and '<ipl_ed>' in special_token_dict:
                            components.extend([
                                special_token_dict['<ipl_st>'],
                                adapter_item,
                                special_token_dict['<ipl_ed>']
                            ])
                        else:
                            # Add adapter output without special tokens if not available
                            components.append(adapter_item)
                    
                    # Add text context if available
                    if context_item is not None:
                        if '<prp_st>' in special_token_dict and '<prp_ed>' in special_token_dict:
                            components.extend([
                                special_token_dict['<prp_st>'],
                                context_item,
                                special_token_dict['<prp_ed>']
                            ])
                            context_t5_components.extend([
                                context_item,
                            ])
                        else:
                            # Add context without special tokens if not available
                            components.append(context_item)
                            context_t5_components.append(context_item)
                    
                    # Concatenate all components (match Omni-Video1: no dtype cast before cat; truncate only)
                    if components:
                        new_context = torch.cat(components, dim=0)
                        if context_t5_components:
                            new_context_t5 = torch.cat(context_t5_components, dim=0)
                        else:
                            new_context_t5 = new_context

                        if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:
                            new_context = new_context[0: self.max_context_len]
                            new_context_t5 = new_context_t5[0: 512]
                    else:
                        # If no components were added (should never happen)
                        raise ValueError("No components available to create context")
                else:
                    # Simple concatenation without special tokens
                    items_to_concat = []
                    
                    # Add all available components in a logical order
                    if visual_item is not None:
                        items_to_concat.append(visual_item)
                    
                    if ref_item is not None:
                        items_to_concat.append(ref_item)
                    
                    if adapter_item is not None:
                        items_to_concat.append(adapter_item)
                    
                    if context_item is not None:
                        items_to_concat.append(context_item)
                    
                    if items_to_concat:
                        new_context = torch.cat(items_to_concat, dim=0)

                        if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:
                            new_context = new_context[0: self.max_context_len]

                        new_context_t5 = new_context
                    else:
                        # If no components were added (should never happen)
                        raise ValueError("No components available to create context")
                
                mixed_context.append(new_context)
                context_t5_only.append(new_context_t5)
        
        else:
            raise ValueError(f"Condition mode {condition_mode} is not supported")
        
        if use_wan_14b:
            out = self.wan_2_model(x, t=t, context=context_t5_only, seq_len=seq_len)  # [B, 16, F, H/8, W/8]
        else:
            out = self.wan_model(x, t=t, context=mixed_context, seq_len=seq_len)
        return out

    def save_pretrained(self, save_dir: str):
        """
        Save the VideoMixedConditionModel to the specified directory.
        
        Args:
            save_dir: Directory to save the models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save WanModel
        wan_save_dir = os.path.join(save_dir, "wan_model")
        wan2_save_dir = os.path.join(save_dir, "wan_14b")

        os.makedirs(wan_save_dir, exist_ok=True)
        os.makedirs(wan2_save_dir, exist_ok=True)
        self.wan_model.save_pretrained(wan_save_dir)
        self.wan_2_model.save_pretrained(wan2_save_dir)
        
        # Save adapter
        if self.adapter is not None:
            adapter_save_dir = os.path.join(save_dir, "adapter")
            os.makedirs(adapter_save_dir, exist_ok=True)
            adapter_state_dict = self.adapter.state_dict()
            torch.save(adapter_state_dict, os.path.join(adapter_save_dir, "adapter_pytorch_model.bin"))
        
        # Save visual context adapter if it exists
        if self.visual_context_adapter is not None:
            visual_adapter_save_dir = os.path.join(save_dir, "visual_context_adapter")
            os.makedirs(visual_adapter_save_dir, exist_ok=True)
            visual_adapter_state_dict = self.visual_context_adapter.state_dict()
            torch.save(visual_adapter_state_dict, os.path.join(visual_adapter_save_dir, "visual_context_adapter_pytorch_model.bin"))
            
        # Save AR Vision Head
        if self.ar_vision_head is not None:
            ar_vision_head_save_dir = os.path.join(save_dir, "ar_vision_head")
            os.makedirs(ar_vision_head_save_dir, exist_ok=True)
            ar_vision_head_state_dict = self.ar_vision_head.state_dict()
            torch.save(ar_vision_head_state_dict, os.path.join(ar_vision_head_save_dir, "ar_vision_head_pytorch_model.bin"))
        
        logging.info(f"Model saved to {save_dir}")
    
    def to(self, *args, **kwargs):
        """Move the model to the specified device and dtype"""
        dtype = kwargs.get('dtype', None)
        if dtype is None and len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]

        if getattr(self, "_dit_fsdp_wrapped", False):
            is_device_move = dtype is None and (
                (len(args) > 0 and isinstance(args[0], torch.device))
                or (len(args) > 0 and isinstance(args[0], str) and ('cuda' in args[0] or args[0] == 'cpu'))
                or kwargs.get('device') is not None
            )
            if is_device_move or dtype is not None:
                if self.adapter is not None:
                    self.adapter = self.adapter.to(*args, **kwargs)
                if self.ar_vision_head is not None:
                    self.ar_vision_head = self.ar_vision_head.to(*args, **kwargs)
                if self.visual_context_adapter is not None:
                    self.visual_context_adapter = self.visual_context_adapter.to(*args, **kwargs)
            return self

        if dtype is not None:
            self.wan_model = self.wan_model.to(*args, **kwargs)
            self.wan_2_model = self.wan_2_model.to(*args, **kwargs)
            if self.adapter is not None:
                self.adapter = self.adapter.to(*args, **kwargs)
            if self.ar_vision_head is not None:
                self.ar_vision_head = self.ar_vision_head.to(*args, **kwargs)
            if self.visual_context_adapter is not None:
                self.visual_context_adapter = self.visual_context_adapter.to(*args, **kwargs)
        else:
            self.wan_model = self.wan_model.to(*args, **kwargs)
            self.wan_2_model = self.wan_2_model.to(*args, **kwargs)
            if self.adapter is not None:
                self.adapter = self.adapter.to(*args, **kwargs)
            if self.ar_vision_head is not None:
                self.ar_vision_head = self.ar_vision_head.to(*args, **kwargs)
            if self.visual_context_adapter is not None:
                self.visual_context_adapter = self.visual_context_adapter.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def enable_train(self):
        """Enable training mode for all models"""
        self.wan_model.train()
        self.wan_2_model.train()
        if self.adapter is not None:
            self.adapter.train()
        if self.ar_vision_head is not None:
            self.ar_vision_head.train()
        if self.visual_context_adapter is not None:
            self.visual_context_adapter.train()

    def enable_eval(self):
        """Enable evaluation mode for all models"""
        self.wan_model.eval()
        self.wan_2_model.eval()
        if self.adapter is not None:
            self.adapter.eval()
        if self.ar_vision_head is not None:
            self.ar_vision_head.eval()
        if self.visual_context_adapter is not None:
            self.visual_context_adapter.eval()
    
    def is_train(self):
        """Check if the model is in training mode"""
        is_training = self.wan_model.training and self.adapter.training
        if self.visual_context_adapter is not None:
            is_training = is_training and self.visual_context_adapter.training
        return is_training
