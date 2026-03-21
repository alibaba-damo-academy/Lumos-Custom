import math
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .misc import get_logger

@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999, initialize=False
) -> None:
    """
    Step the EMA model towards the current model.
    """
    if initialize:
        # Directly copy parameters
        with torch.no_grad():
            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                ema_param.copy_(model_param)
    else:
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            if not param.requires_grad:
                continue
            # Check if original parameter is NaN
            if torch.isnan(param.data).any():
                print(f"Warning: NaN detected in model parameter {name}")
                continue
            
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
            # Check if updated EMA parameter is NaN
            if torch.isnan(ema_params[name]).any():
                print(f"Warning: NaN detected in EMA parameter {name}")

class TemporalMaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        get_logger().info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


# ===============================================
# MAE Functions
# ===============================================
# https://github.com/facebookresearch/mae https://github.com/Anima-Lab/MaskDiT

class SpatialMaskGenerator:
    
    def __init__(self, mask_ratio, patch_size):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
    def get_mask(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, T, H, W = x.shape  # batch, length, dim
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        L = H*W
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, T, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, T, L], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return {'mask': mask, 
                'ids_keep': ids_keep, 
                'ids_restore': ids_restore,
                'shape': (T, H, W)}
        
        
class CubeSpatialMaskGenerator:
    
    def __init__(self, mask_ratio, patch_size, cube_size):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.cube_size = cube_size
        
    def get_mask(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, T, H, W = x.shape  # batch, length, dim
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        
        if T == 1:
            cube_size = [1, self.cube_size[1], self.cube_size[2]]
        else:
            cube_size = self.cube_size
        T_cube = T // cube_size[0]
        H_cube = H // cube_size[1]
        W_cube = W // cube_size[2]
        
        L_cube = H_cube*W_cube
        len_keep = int(L_cube * (1 - self.mask_ratio))

        noise = torch.rand(N, T_cube, L_cube, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore1 = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, T_cube, L_cube], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore1)
        
        # expand mask to shape before cube
        mask = mask.view(N, T_cube, H_cube, W_cube)
        mask = mask.unsqueeze(2).repeat(1, 1, cube_size[0], 1, 1)  # Shape: (N, T, cube_size[0], H, W)
        mask = mask.view(N, T_cube * cube_size[0], H_cube, W_cube)
        mask = mask.unsqueeze(3).unsqueeze(5)  # Shape: (N, cube_size[0]*T, H, 1, W, 1)
        mask = mask.repeat(1, 1, 1, cube_size[1], 1, cube_size[2])  # Shape: (N, cube_size[0]*T, H, cube_size[1], W, cube_size[2])
        mask = mask.view(N, cube_size[0] * T_cube, cube_size[1] * H_cube, cube_size[2] * W_cube)
        mask_full = torch.zeros([N, T, H, W], device=x.device)
        mask_full[:, :cube_size[0] * T_cube, :cube_size[1] * H_cube, :cube_size[2] * W_cube] = mask
        delta_t = T - cube_size[0] * T_cube
        if delta_t > 0:
            mask_full[:, cube_size[0] * T_cube:, :cube_size[1] * H_cube, :cube_size[2] * W_cube] = mask[:, -delta_t:]
        
        # from torchvision.utils import save_image
        # mask_normalized = mask_full.float()  
        # mask_normalized = (mask_normalized - mask_normalized.min()) / (mask_normalized.max() - mask_normalized.min())
        # for i in range(N):
        #     for j in range(T):
        #         save_path = '{}-{}.jpg'.format(i,j)
        #         save_image(mask_normalized[i][j].unsqueeze(0).unsqueeze(0), save_path)

        
        mask_full = mask_full.view(N, T, H*W)
        ids_keep = torch.where(mask_full == 0)[-1].view(N, T, -1) # mask==0 
        ids_drop = torch.where(mask_full == 1)[-1].view(N, T, -1)  # mask==1 
        
        all_indices = torch.cat([ids_keep, ids_drop], dim=-1)
        ids_restore = torch.argsort(all_indices, dim=2)
        # diff = ids_restore - ids_restore1
        # print(torch.sum(diff))
        
        # mask_image = torch.gather(mask_full.unsqueeze(-1), dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, 1))
        # mask_token = torch.ones(1, 1, 1, 1, device=mask_image.device)
        # unmask_image = unmask_tokens(mask_image, ids_restore, mask_token)
        # from torchvision.utils import save_image
        # unmask_image = unmask_image.view(N, T, H, W)
        # # unmask_image = unmask_image.float()  
        # # unmask_image = (unmask_image - unmask_image.min()) / (unmask_image.max() - unmask_image.min())
        # diff = unmask_image.view(N, T, H*W) - mask_full
        # diff = torch.sum(diff)
        # print(diff)
        # for i in range(N):
        #     for j in range(T):
        #         save_path = '{}-{}.jpg'.format(i,j)
        #         save_image(unmask_image[i][j], save_path)

        return {'mask': mask_full, 
                'ids_keep': ids_keep, 
                'ids_restore': ids_restore,
                'shape': (T, H, W)}


def unmask_tokens(x, ids_restore, mask_token):
    B, T, S = ids_restore.shape
    x = x.view(B, T, -1, x.shape[-1])
    mask_tokens = mask_token.repeat(B, T, S-x.shape[2], 1)
    x_ = torch.cat([x, mask_tokens], dim=2)  
    x = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])) # unshuffle
    return x.view(B, T*S, x.shape[-1])

def mask_rope_embedding(rope_freq, mask, B, T, N):
    _, M, C = rope_freq.shape
    rope_freq = rope_freq[None,:, :, :].repeat(B, 1, 1, 1)
    rope_freq = rope_freq.reshape(B, T, N//T, M*C)
    rope_freq = torch.gather(rope_freq, dim=2, index=mask.unsqueeze(-1).repeat(1, 1, 1, rope_freq.shape[-1]))
    rope_freq = rope_freq.reshape(B, -1, M, C)
    return rope_freq

def mask_rope_embedding_hunyuan(rope_freq, mask, B, T):
    N, C = rope_freq.shape
    rope_freq = rope_freq[None,:, :].repeat(B, 1, 1)
    rope_freq = rope_freq.reshape(B, T, N//T, C)
    rope_freq = rope_freq.to(mask.device)
    rope_freq = torch.gather(rope_freq, dim=2, index=mask.unsqueeze(-1).repeat(1, 1, 1, rope_freq.shape[-1]))
    rope_freq = rope_freq.reshape(B, -1, C)
    return rope_freq

def compute_packed_indices(N, text_mask):
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        N: Number of visual tokens.
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    assert N > 0 and len(text_mask) == 1
    text_mask = text_mask[0]

    mask = F.pad(text_mask, (N, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(
        mask.flatten(), as_tuple=False
    ).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * N  # At least (B * N,)
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens,
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
    }
    
def get_packed_indices(y_mask, N):
    assert len(y_mask) == 1
    packed_indices = compute_packed_indices(N, y_mask)
    device = y_mask[0].device
    for key in packed_indices.keys():
        if isinstance(packed_indices, torch.Tensor):
            packed_indices[key] = packed_indices[key].to(device)
    return packed_indices

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)