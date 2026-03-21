import functools
import json
import operator
import os
from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, 
    set_state_dict, 
    get_model_state_dict, 
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url

from .misc import get_logger
from typing import Optional, Tuple, List

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {}

def load_model_by_layer(model, checkpoint_path):
    # Unused function, consider this approach if memory optimization is needed during model loading. Just keeping the implementation here.
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])
            del state_dict[name]  # Delete used parameters promptly
        
    gc.collect()  # Trigger garbage collection

def load_checkpoint(model, ckpt_path, save_as_pt=False, model_name="ema", strict=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict, assign=True)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif os.path.isdir(ckpt_path):
        state_dict = dcp_to_torch_save(ckpt_path, model_name, save_as_pt)
        filterd_state_dict = {k: v for k, v in state_dict.items() if "freqs" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filterd_state_dict, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

def load_qkv_bg_checkpoint(model, ckpt_path, save_as_pt=False, model_name="ema", strict=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict, assign=True)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)   
        state_dict_new = {}

        original_weight = state_dict["patch_embedding.weight"]
        original_bias = state_dict.get("patch_embedding.bias", None)

        zero_weight = torch.zeros_like(original_weight)
        new_weight = torch.cat([original_weight, zero_weight], dim= 1)
            
        state_dict_new["patch_embedding_new.weight"] = new_weight
        if original_bias is not None:
            state_dict_new["patch_embedding_new.bias"] = original_bias

        for key, value in state_dict.items():
            if 'self_attn' or 'cross_attn' in key and 'norm' not in key:
                parts1 = key.split('.')
                if 'q' in parts1 or 'k' in parts1 or 'v' in parts1:
                    idx = next(i for i, p in enumerate(parts1) if p in ['q', 'k', 'v'])
                    parts1[idx] = parts1[idx] + '_fore'
                    new_key1 = '.'.join(parts1[:idx+1] + parts1[idx+1:])
                    state_dict_new[new_key1] = value
                state_dict_new[key] = value
            else:
                state_dict_new[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif os.path.isdir(ckpt_path):
        state_dict = dcp_to_torch_save(ckpt_path, model_name, save_as_pt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def load_qkv_checkpoint(model, ckpt_path, save_as_pt=False, model_name="ema", strict=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict, assign=True)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)   
        state_dict_new = {}
        for key, value in state_dict.items():
            if 'self_attn' or 'cross_attn' in key and 'norm' not in key:
                parts1 = key.split('.')
                parts2 = key.split('.')
                if 'q' in parts1 or 'k' in parts1 or 'v' in parts1:
                    idx = next(i for i, p in enumerate(parts1) if p in ['q', 'k', 'v'])
                    parts1[idx] = parts1[idx] + '_fore'
                    parts2[idx] = parts2[idx] + '_back'
                    new_key1 = '.'.join(parts1[:idx+1] + parts1[idx+1:])
                    new_key2 = '.'.join(parts2[:idx+1] + parts2[idx+1:])
                    state_dict_new[new_key1] = value
                    state_dict_new[new_key2] = value
                state_dict_new[key] = value
            else:
                state_dict_new[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif os.path.isdir(ckpt_path):
        state_dict = dcp_to_torch_save(ckpt_path, model_name, save_as_pt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# save and load for training
def get_state_dicts(model: nn.Module, ema: Optional[nn.Module], optimizer: Optional[Optimizer], options=None):
    """Get state_dicts for model, EMA and optimizer."""
    state_dicts = {}
    if ema is not None:
        state_dicts["ema"] = get_model_state_dict(ema, options=options)
    if model is not None:
        state_dicts["model"] = get_model_state_dict(model, options=options)
    if optimizer is not None:
        state_dicts["optimizer"] = get_optimizer_state_dict(model, optimizer, options=options)
    return state_dicts

def async_save_state_dict(state_dict: Optional[dict], path: str, device_mesh):
    """Handle asynchronous saving of state dict."""
    if state_dict is not None and device_mesh['rep'].get_group().rank() == 0:
        return dcp.async_save(state_dict, checkpoint_id=path, process_group=device_mesh['shard'].get_group())
    return None

def async_save(
    save_dir: str,
    model: Optional[nn.Module] = None,
    ema: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    sampler=None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    global_step: Optional[int] = None,
    batch_size: Optional[int] = None,
    device_mesh=None,
    last_dcp_handles: Optional[dict] = None,
    mode='all'
) -> Tuple:
    rank = dist.get_rank()
    epoch_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    
    # Synchronize directory creation across ranks
    if rank == 0:
        os.makedirs(epoch_dir, exist_ok=True)
    dist.barrier()

    # Wait for prior async save operations to complete
    if last_dcp_handles is not None:
        for k, handle in last_dcp_handles.items():
            if handle is not None:
                if rank == 0:
                    print('waiting for last dcp save finished.')
                handle.result()

    # Set options for saving state dicts
    options = StateDictOptions(cpu_offload=True, full_state_dict=False)
    
    # Get state dicts
    if rank == 0:
        print(f'before getting all state_dict, memory allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB') 
    state_dicts = get_state_dicts(model, ema, optimizer, options)
    if rank == 0:
        print(f'after getting all state_dict, memory allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB') 
        print('Since I optimize state_dicts to CPU, You can use these numbers to ensure no more GPU memory usage!')

    # Save state dicts asynchronously
    handles = {}
    # default to save them all together. ortherwise in a seperate way
    if mode == 'all':
        path = os.path.join(epoch_dir, mode)
        handles[mode] = async_save_state_dict(state_dicts, path, device_mesh)
    else:
        save_order = ["ema", "model", "optimizer"]  # Order by GPU memory usage from smallest to largest
        for key in save_order:
            if state_dicts[key] is None:
                continue
            path = os.path.join(epoch_dir, key)
            handles[key] = async_save_state_dict(state_dicts[key], path, device_mesh)
            if handles[key] is not None and key in ['ema', 'model']:
                if rank == 0:
                    print(f'Bolcking at {key} saving...')
                    # TODO, use pinned memory to speed up this stage. 
                    # see https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html#even-more-performance-with-pinned-memory
                handles[key].result()
                del state_dicts[key]
    torch.cuda.empty_cache()  # Optionally clear cache to manage memory

    if rank == 0:
        # Save running states and other components
        running_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "batch_size": batch_size,
        }
        save_json(running_states, os.path.join(epoch_dir, "running_states.json"))

        if sampler is not None:
            torch.save(sampler.state_dict(step), os.path.join(epoch_dir, "sampler"))

        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(epoch_dir, "lr_scheduler"))

    dist.barrier()
    
    return handles

def sharded_load(
    load_dir: str,
    model: Optional[nn.Module] = None,
    ema: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    sampler=None,
    mode='all'
) -> Tuple[int, int]:
    # Handle absence of running_states.json (for inference)
    running_states = dict(epoch=0, step=0)  # Default values for inference

    running_states_path = os.path.join(load_dir, "running_states.json")
    if os.path.exists(running_states_path):
        running_states = load_json(running_states_path)

    # Load state dicts
    # TODO: this may cause peak memory usage when loading these state_dict since here we use GPU memory.
    state_dicts = get_state_dicts(model, ema, optimizer)
    if mode == 'all':
        dcp.load(
            state_dict=state_dicts,
            checkpoint_id=os.path.join(load_dir, mode),
            process_group=dist.group.WORLD
        )
        if model is not None:
            set_model_state_dict(model, state_dicts['model'])
        if optimizer is not None:
            set_optimizer_state_dict(model, optimizer, state_dicts['optimizer'])
        if ema is not None:
            set_model_state_dict(ema, state_dicts['ema'])
    else:
        if model is not None:
            dcp.load(
                state_dict=state_dicts['model'],
                checkpoint_id=os.path.join(load_dir, "model"),
                process_group=dist.group.WORLD
            )
            set_model_state_dict(model, state_dicts['model'])

        if optimizer is not None:
            dcp.load(
                state_dict=state_dicts['optimizer'],
                checkpoint_id=os.path.join(load_dir, "optimizer"),
                process_group=dist.group.WORLD
            )
            set_optimizer_state_dict(model, optimizer, state_dicts['optimizer'])

        if ema is not None:
            dcp.load(
                state_dict=state_dicts['ema'],
                checkpoint_id=os.path.join(load_dir, "ema"),
                process_group=dist.group.WORLD
            )
            set_model_state_dict(ema, state_dicts['ema'])

    del state_dicts
    # Load other components like sampler and lr_scheduler
    if sampler is not None:
        sampler_path = os.path.join(load_dir, "sampler")
        if os.path.exists(sampler_path):
            sampler.load_state_dict(torch.load(sampler_path))
    
    if lr_scheduler is not None:
        lr_scheduler_path = os.path.join(load_dir, "lr_scheduler")
        if os.path.exists(lr_scheduler_path):
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))

    dist.barrier()
    torch.cuda.empty_cache()
    
    return running_states["epoch"], running_states["step"]

def dcp_to_torch_save(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    model_name=None,
    save_as_pt=True
):
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if model_name in dcp_checkpoint_dir and model_name not in sd:
        # Path is for a single model's distributed checkpoint, not mixed
        used_sd = sd
    else:
        # Path stores mixed shards of model/ema/optim
        used_sd = sd[model_name]
    if save_as_pt:
        save_path = dcp_checkpoint_dir+'.pt'
        torch.save(used_sd, save_path)
        get_logger().info("Model checkpoint saved to %s", save_path)
    return used_sd