import random
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.io import write_video

from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# save sample
def save_sample(x, save_path=None, fps=12, normalize=True, value_range=(-1, 1), force_video=False, verbose=True):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264", options={"quality": "100"})
    if verbose:
        print(f"Saved to {save_path}")
    return save_path

# Auto grad checkpoint
def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        else:
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)