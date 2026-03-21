import torch
import torch.nn as nn
import torch.amp as amp
import math

from .attention import flash_attention
import torch.distributed as dist
from source.registry import MODELS
from source.utils.ckpt_utils import load_checkpoint
from source.acceleration.checkpoint import auto_grad_checkpoint
import torch.nn.functional as F
from training_acc.dist.parallel_state import is_enable_sequence_parallel
from training_acc.patches.wanx2_1_t2v import split, gather, collect_tokens, collect_heads
import time

from magi_attention.common import AttnRanges


__all__ = ['Transformer']

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_apply_with_shift_1condition(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, ((f, h, w), (cf, ch, cw)) in enumerate(grid_sizes):
        seq_len = (f * h * w) + (cf * ch * cw) 

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape((f * h * w), 1, -1)
        freqs_i_c = torch.cat([
            freqs[0][:cf].view(cf, 1, 1, -1).expand(cf, ch, cw, -1),
            freqs[1][h:h + ch].view(1, ch, 1, -1).expand(cf, ch, cw, -1),
            freqs[2][w:w + cw].view(1, 1, cw, -1).expand(cf, ch, cw, -1)
        ], dim=-1).reshape((cf * ch * cw), 1, -1)
        
        freqs_i = torch.cat((freqs_i, freqs_i_c), dim=0)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

@amp.autocast("cuda", enabled=False)
def rope_apply_with_shift_2conditions(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, ((f, h, w), (cf, ch, cw), (cf2, ch2, cw2)) in enumerate(grid_sizes):
        seq_len = (f * h * w) + (cf * ch * cw) + (cf2 * ch2 * cw2)

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape((f * h * w), 1, -1)
        freqs_i_c = torch.cat([
            freqs[0][:cf].view(cf, 1, 1, -1).expand(cf, ch, cw, -1),
            freqs[1][h:h + ch].view(1, ch, 1, -1).expand(cf, ch, cw, -1),
            freqs[2][w:w + cw].view(1, 1, cw, -1).expand(cf, ch, cw, -1)
        ], dim=-1).reshape((cf * ch * cw), 1, -1)
        freqs_i_c2 = torch.cat([
            freqs[0][:cf2].view(cf2, 1, 1, -1).expand(cf2, ch2, cw2, -1),
            freqs[1][h + ch:h + ch + ch2].view(1, ch2, 1, -1).expand(cf2, ch2, cw2, -1),
            freqs[2][w + cw:w + cw + cw2].view(1, 1, cw2, -1).expand(cf2, ch2, cw2, -1)
        ], dim=-1).reshape((cf2 * ch2 * cw2), 1, -1)
        freqs_i = torch.cat((freqs_i, freqs_i_c, freqs_i_c2), dim=0)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@amp.autocast("cuda", enabled=False)
def rope_apply_with_shift_T_multi(x, grid_sizes, freqs, cond_length, cond_non_human_length):
    n, c = x.size(2), x.size(3) // 2
    
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (grid_blocks, cond_len, cond_non_human_len) in enumerate(zip(grid_sizes, cond_length, cond_non_human_length)):
        seq_len = 0
        freqs_i_components = []
        x_parts = []
        f_former = 0
        for block_idx, (f, h, w) in enumerate(grid_blocks):
            if block_idx  <  cond_len + 1:
                if block_idx <= cond_non_human_len:
                    freq_block =  torch.cat([
                    freqs[0][f_former : f + f_former].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][ : h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][ : w].view(1, 1, w, -1).expand(f, h, w, -1)
                    ], dim=-1).reshape(f * h * w, 1, -1)
                    
                    f_former += f

                    freqs_i_components.append(freq_block)
                    
                    seq_len += f * h * w
                else:
                    h_former = 0
                    w_former = 0 
                    freq_block_f_i_components = list()
                    for f_i in range(f):
                        freq_block_f_i =  torch.cat([
                        freqs[0][f_former : f_former + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                        freqs[1][h_former : h + h_former].view(1, h, 1, -1).expand(1, h, w, -1),
                        freqs[2][ w_former: w + w_former].view(1, 1, w, -1).expand(1, h, w, -1)
                        ], dim=-1).reshape(1 * h * w, 1, -1)                        
                        h_former += h
                        w_former += w
                        freq_block_f_i_components.append(freq_block_f_i)                        
                        seq_len += 1 * h * w
                    freq_block = torch.cat(freq_block_f_i_components,dim=0)
                    freqs_i_components.append(freq_block)
                    f_former += 1
                    
            else:
                break

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(freqs_i_components, dim=0)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).to(x.dtype)


@amp.autocast("cuda", enabled=False)
def rope_apply_with_shift_T(x, grid_sizes, freqs, cond_length, cond_non_human_length):
    n, c = x.size(2), x.size(3) // 2
    
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (grid_blocks, cond_len) in enumerate(zip(grid_sizes, cond_length)):

        seq_len = 0
        freqs_i_components = []
        x_parts = []
        f_former = 0
        h_former = 0
        w_former = 0 
        for block_idx, (f, h, w) in enumerate(grid_blocks):
            if block_idx <  cond_len + 1:
                freq_block =  torch.cat([
                freqs[0][f_former : f + f_former].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][ : h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][ : w].view(1, 1, w, -1).expand(f, h, w, -1)
                ], dim=-1).reshape(f * h * w, 1, -1)
                
                f_former += f
                h_former += h
                w_former += w

                freqs_i_components.append(freq_block)
                
                seq_len += f * h * w
            else:
                break

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(freqs_i_components, dim=0)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).to(x.dtype)

@amp.autocast("cuda", enabled=False)
def rope_apply_with_shift(x, grid_sizes, freqs, cond_length):
    n, c = x.size(2), x.size(3) // 2
    
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (grid_blocks, cond_len) in enumerate(zip(grid_sizes,cond_length)):

        seq_len = 0
        freqs_i_components = []
        x_parts = []
        f_former = 0
        h_former = 0
        w_former = 0 
        for block_idx, (f, h, w) in enumerate(grid_blocks):
            if block_idx <  cond_len + 1:
                freq_block =  torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][h_former : h + h_former].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][w_former : w + w_former].view(1, 1, w, -1).expand(f, h, w, -1)
                ], dim=-1).reshape(f * h * w, 1, -1)
                
                f_former += f
                h_former += h
                w_former += w

                freqs_i_components.append(freq_block)
                
                seq_len += f * h * w
            else:
                break

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(freqs_i_components, dim=0)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).to(x.dtype)


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
    
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        self_mask_bool = True, 
        cross_mask_bool = True, 
        pooling_type = 'mean', 
        ratio = 0.1,
        RoPE_shift = True
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.self_mask_bool = self_mask_bool
        self.cross_mask_bool = cross_mask_bool
        self.pooling_type = pooling_type
        self.ratio = ratio
        self.RoPE_shift = RoPE_shift

        # layers
        self.q = nn.Linear(dim, dim)

        self.k = nn.Linear(dim, dim)

        self.v = nn.Linear(dim, dim)

        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, seq_lens, grid_sizes, freqs, cond_length, cond_non_human_length, cond_length_human):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        
        T, H, W = grid_sizes[0][0]       
       
        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)
        # end_time = time.time()
        # print(f"part1_time:{end_time-start_time}")
        
        # start_time = time.time()
        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = collect_tokens(k)
            v = collect_tokens(v)

        T, H, W = grid_sizes[0][0]
        N = int(q.shape[1] / (H * W))
        # end_time = time.time()
        # print(f"part2_time:{end_time-start_time}")
        
        # mask = torch.zeros(b,N*H*W,N*H*W,device=q.device).bool()

        # for ii, (cond_len, cond_non_human, cond_len_human)  in enumerate(zip(cond_length, cond_non_human_length, cond_length_human)):
        #     N_ii = T + cond_len
        #     cond_len_human_value = [value for key, value in cond_len_human.items()]
        #     positions = torch.arange(N_ii, device=q.device).unsqueeze(1).repeat(1, 1).view(-1)  # shape [41340]
        #     positions = torch.arange(N_ii, device=q.device).unsqueeze(1).repeat(1, H*W).view(-1)  # shape [41340]
        #     same_group = positions.unsqueeze(1) == positions.unsqueeze(0) 
        #     is_high_group = positions >= T  # [seq_len]
        #     mask_group = (~ (is_high_group.unsqueeze(1) & is_high_group.unsqueeze(0))) 
        #     mask_group[is_high_group,:] = False
        #     mask_ii = same_group | mask_group
        #     mask[ii,:N_ii*H*W,:N_ii*H*W] = mask_ii
        #     j_human = 0
        #     for cond_len_human_v in cond_len_human_value:
        #         N_human_start = T + j_human + cond_non_human 
        #         N_human_end = T + j_human + cond_non_human  + cond_len_human_v
        #         j_human += cond_len_human_v
        #         mask[ii, N_human_start*H*W:N_human_end*H*W, N_human_start*H*W:N_human_end*H*W] = True
            
        #     # mask[ii,:N_ii*H*W,:N_ii*H*W] = True
        #     q_lens.append(N_ii*H*W)
        #     k_lens.append(N_ii*H*W)
        # start_time = time.time()
        q_lens = list()
        k_lens = list()
        q_range = list()
        k_range = list()
        start_idx = 0
        for ii, (cond_len, cond_non_human, cond_len_human, grid_size)  in enumerate(zip(cond_length, cond_non_human_length, cond_length_human, grid_sizes)):
            N_ii = T + cond_len
            grid_start = 0

            for grid_ in grid_size:
                if int(grid_start - T) < cond_len:
                    grid_shape = int(grid_[0])
                    if grid_shape != T:
                        grid_start_idx = int(start_idx + grid_start*H*W)
                        grid_end_idx = int(start_idx + (grid_start + grid_shape)*H*W)
                        k_range.append([grid_start_idx, grid_end_idx])
                        q_range.append([grid_start_idx, grid_end_idx])
                        grid_start += grid_shape
                    else:
                        grid_start_idx = int(start_idx + grid_start*H*W)
                        grid_end_idx = int(start_idx + (grid_start + N_ii)*H*W)
                        k_range.append([grid_start_idx, grid_end_idx])
                        q_range.append([grid_start_idx, int(start_idx + ( grid_start + grid_shape)*H*W)])
                        grid_start += grid_shape
            q_lens.append(N_ii*H*W)
            k_lens.append(N_ii*H*W)
            start_idx += N_ii*H*W

        q_ranges = AttnRanges.from_ranges(q_range).to_tensor(device=q.device)
        k_ranges = AttnRanges.from_ranges(k_range).to_tensor(device=q.device)
        attn_type_map = torch.tensor([0]*len(q_range), device=q.device, dtype=torch.int32)

        if self.RoPE_shift:
            rope_q = rope_apply_with_shift_T_multi(q, grid_sizes, freqs, cond_length, cond_non_human_length)
            rope_k = rope_apply_with_shift_T_multi(k, grid_sizes, freqs, cond_length, cond_non_human_length)
        else:
            # rope_apply_with_shift_T(x, grid_sizes, freqs, cond_length, cond_non_human_length)
            rope_q = rope_apply_with_shift_T(q, grid_sizes, freqs, cond_length, cond_non_human_length)
            rope_k = rope_apply_with_shift_T(k, grid_sizes, freqs, cond_length, cond_non_human_length)
        # end_time = time.time()
        # print(f"part3_time:{end_time-start_time}")
        q_lens = torch.tensor(q_lens, dtype=torch.int32).to(device=q.device, non_blocking=True)
        k_lens = torch.tensor(k_lens, dtype=torch.int32).to(device=q.device, non_blocking=True)
        # res = torch.softmax(rope_q[0, :, 0] @ rope_k[0, :, 0].transpose(0, 1) / math.sqrt(128), dim=-1)
        # print(res[0])
        # start_time = time.time()
        if self.self_mask_bool:
            x = flash_attention(
                q=rope_q,
                k=rope_k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                q_ranges = q_ranges,
                k_ranges = k_ranges,
                attn_type_map = attn_type_map,
                window_size=self.window_size,
                self_mask_bool =  self.self_mask_bool
            )
        else:
            x = flash_attention(
                q=rope_q,
                k=rope_k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                window_size=self.window_size,
                self_mask_bool =  self.self_mask_bool
            )
        # end_time = time.time()
        # print(f"part4_time:{end_time-start_time}")


        # start_time = time.time()
        if is_enable_sequence_parallel():
            x = collect_heads(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        # end_time = time.time()
        # print(f"part5_time:{end_time-start_time}")
        return x


class CrossAttention(SelfAttention):

    def forward(self, x, context, context_lens, grid_sizes, mask_text_bool, cond_length, cond_length_human):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        # b, n, d = x.size(0), self.num_heads, self.head_dim
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        T, H, W = grid_sizes[0][0]
       
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = split(k, 2)
            v = split(v, 2)
        
        text_len = v.shape[1]
        

        #######################################Obtain Cross Attention Mask#######################################
        b, s = x.shape[:2]
        T, H, W = grid_sizes[0][0]
        text_len = context.shape[1]
        T_masks = torch.zeros(b,T*H*W, text_len, device=mask_text_bool.device)
        mask_ = list()
        

        pool_size = 8
        condi_attn_weights = list()
        for (mask_text_bool_bi, cond_len, cond_len_human, text_emb, img_emb) in zip(mask_text_bool, cond_length, cond_length_human, k, q):
            text_emb_mask = text_emb.unsqueeze(0) # * mask_text_bool_bi.float().unsqueeze(2).unsqueeze(2)
            cond_img_emb_ = list()
            for idx, condi_mask_text_bool in enumerate(mask_text_bool_bi):
                cond_img_emb = img_emb[(T+idx)*H*W:(T+idx+1)*H*W,:]
                if self.pooling_type == 'mean':
                    # cond_img_emb_mean = cond_img_emb.mean(dim=0).unsqueeze(0)
                    H_new = ((H + pool_size -1) // pool_size) * pool_size   
                    W_new = ((W + pool_size -1) // pool_size) * pool_size   
                    cond_img_emb_re = cond_img_emb.reshape(H, W, n, d)
                    cond_img_emb_padded = F.pad(cond_img_emb_re, (0, 0, 0, 0, 0, W_new - W, 0, H_new - H)).permute(2, 3, 0, 1)                    
                    cond_img_emb_mean =  torch.nn.functional.avg_pool2d(cond_img_emb_padded, kernel_size=pool_size, stride=pool_size) 
                    H_down, W_down = cond_img_emb_mean.shape[-2:]
                    cond_img_emb_mean = cond_img_emb_mean.permute(2, 3, 0, 1).reshape(-1, n, d) 
                cond_img_emb_.append(cond_img_emb_mean)
            cond_img_emb_ = torch.stack(cond_img_emb_, dim=0)
            scale_factor = 1 / math.sqrt(cond_img_emb_.size(-1))  # face -> man > mean_score (-20)+ 1 * abs(scale (2)) / 1,2.5, 7.5,5, 10/10
            attn_weight = cond_img_emb_.transpose(1,2) @ text_emb_mask.transpose(1,2).transpose(-2, -1) * scale_factor # [10, 512， 12， 128]  -> subject img -> text score + score * 2/5/7/9/10
            attn_weight_re = attn_weight.reshape(-1, n, H_down, W_down, text_len)
            attn_weight_upsample = attn_weight_re.repeat_interleave(pool_size, dim=2).repeat_interleave(pool_size, dim=3)  
            attn_weight = attn_weight_upsample.reshape(-1, n, H_new* W_new, text_len)[:,:,:H*W,:]
        condi_attn_weights.append(attn_weight)

        for (T_mask, mask_text_bool_bi, cond_len, cond_len_human, condi_attn_weight) in zip(T_masks, mask_text_bool, cond_length, cond_length_human, condi_attn_weights):
            mask_text_bool_bi_ = list()
            # Obtain all condi pos
            # mask_condi_pos = mask_text_bool_bi.any(dim=0)
            # with condi 0, without condi -0.005
            # mask_template = torch.where(mask_condi_pos, torch.tensor(0.0), torch.tensor(-0.005))
            # objects & bg
            cond_len_human_ = [value for key, value in cond_len_human.items()] 
            cond_len_non_human = cond_len - sum(cond_len_human_)
            for i_non in range(cond_len_non_human):
                mask_text_bool_bi_i_non = mask_text_bool_bi[i_non].float().unsqueeze(0).unsqueeze(0)
                condi_attn_weight_i = condi_attn_weight[i_non] * mask_text_bool_bi_i_non
                condi_attn_weight_i_abs  = torch.abs(condi_attn_weight_i) * self.ratio
                condi_attn_weight_i_ = condi_attn_weight_i_abs.transpose(0,1).unsqueeze(0)
                mask_text_bool_bi_.append(condi_attn_weight_i_)

            # face & attributes
           
            j_human = 0
            mask_human_template_list = list()
            attn_weight_human_list = list()
            for cond_len_human_i in cond_len_human_:
                mask_human_proto = mask_text_bool_bi[j_human + cond_len_non_human: j_human + cond_len_non_human+cond_len_human_i] 
                condi_attn_weight_human_proto = condi_attn_weight[j_human + cond_len_non_human: j_human + cond_len_non_human+cond_len_human_i] 
                mask_human_pos = mask_human_proto.any(dim=0)
                mask_human_template = mask_human_pos.float()
                mask_human_template_list.append(mask_human_template)
                attn_weight_human_list.append(condi_attn_weight_human_proto)
                j_human += cond_len_human_i
            for i_idx, cond_len_human_i  in enumerate(cond_len_human_):
                mask_human_template_minus = mask_human_template_list[:i_idx] + mask_human_template_list[i_idx + 1:]
                if mask_human_template_minus != []:
                    mask_human_template_minus_template = -1.0 * torch.stack(mask_human_template_minus,dim=0).sum(dim=0) / (len(cond_len_human_)-1)
                    mask_human_template = mask_human_template_list[i_idx]
                    mask_human_template_new = (mask_human_template + mask_human_template_minus_template).unsqueeze(0).unsqueeze(0)
                    condi_attn_weight_i_abs  = torch.abs(attn_weight_human_list[i_idx])
                    condi_attn_weight_i_ = condi_attn_weight_i_abs * mask_human_template_new * self.ratio
                    mask_human_template_new_ = condi_attn_weight_i_.transpose(1,2)
                    splits = [1] * cond_len_human_i
                    mask_human_template_new_list = torch.split(mask_human_template_new_, splits, dim=0)
                else:
                    mask_human_template = mask_human_template_list[i_idx].unsqueeze(0).unsqueeze(0)
                    condi_attn_weight_i_abs  = torch.abs(attn_weight_human_list[i_idx])
                    condi_attn_weight_i_ = condi_attn_weight_i_abs * mask_human_template * self.ratio
                    mask_human_template_new_ = condi_attn_weight_i_.transpose(1,2)
                    splits = [1] * cond_len_human_i
                    mask_human_template_new_list = torch.split(mask_human_template_new_, splits, dim=0)
                mask_text_bool_bi_.extend(mask_human_template_new_list)
            # batch no condi
            non_condi_len = mask_text_bool_bi.shape[0] - cond_len
           
            # mask_non_condi_template = (torch.empty(1,mask_text_bool_bi.shape[1], device=mask_text_bool.device).fill_(float('-inf'))).repeat(H * W, 1)
            # mask_non_condi_template = (torch.empty(1,mask_text_bool_bi.shape[1], device=mask_text_bool.device).fill_(1.0)).repeat(H * W, 1)
            mask_non_condi_template = torch.zeros(1,mask_text_bool_bi.shape[1], device=mask_text_bool.device, dtype=T_masks.dtype).repeat(H * W, 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, self.num_heads, 1)
            mask_non_condi_template_list = [mask_non_condi_template] *  non_condi_len
            mask_text_bool_bi_.extend(mask_non_condi_template_list)
            mask_condi_ii = torch.cat(mask_text_bool_bi_,dim=1)
            mask_ii = torch.cat([T_mask.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.num_heads, 1), mask_condi_ii], dim=1)
            mask_.append(mask_ii)
        cross_mask = torch.cat(mask_, dim=0) 
        ##########################################################################################################################
        # break
        x = flash_attention(q, k, v, k_lens=context_lens, mask=cross_mask, cross_mask_bool = self.cross_mask_bool)
        # x = torch.nn.functional.scaled_dot_product_attention(q.reshape(b,-1, *q.shape[1:]).transpose(1,2), k.reshape(b,-1, *k.shape[1:]).transpose(1,2), v.reshape(b,-1, *v.shape[1:]).transpose(1,2), is_causal=False).transpose(1,2) 

        # x = flash_attention(q, k, v, k_lens=context_lens)
        # print(torch.norm(x - x2))
        if is_enable_sequence_parallel():
            x = collect_heads(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_mask_bool = True, 
        cross_mask_bool = True, 
        pooling_type = 'mean', 
        ratio = 0.1,
        RoPE_shift = True
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.self_mask_bool = self_mask_bool
        self.cross_mask_bool = cross_mask_bool
        self.pooling_type = pooling_type
        self.ratio = ratio
        self.RoPE_shift = RoPE_shift

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps,
                                        self_mask_bool = self.self_mask_bool, cross_mask_bool = self.cross_mask_bool, pooling_type = self.pooling_type, ratio = self.ratio, RoPE_shift = self.RoPE_shift)
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps,
                                             self_mask_bool = self.self_mask_bool, cross_mask_bool = self.cross_mask_bool, pooling_type = self.pooling_type, ratio = self.ratio)                                           
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        mask_text_bool,
        cond_length,
        cond_non_human_length,
        cond_length_human,
    ):
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32
        
        # start_time = time.time()
        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, cond_length, cond_non_human_length, cond_length_human
        )
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]
        # end_time = time.time()
        # print(f"Self_attention time:{end_time-start_time}")
        # cross-attention & ffn function

        def cross_attn_ffn(x, context, context_lens, e):
            # start_time = time.time()
            x = x + self.cross_attn(self.norm3(x), context, context_lens, grid_sizes, mask_text_bool, cond_length, cond_length_human)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
            # end_time = time.time()
            # print(f"Cross_attention time:{end_time-start_time}")

            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
    
    def forward(self, x, e):
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x

class Transformer_qkv(nn.Module):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_fixed_seq_len=True,
        sp_degree=1,
        self_mask_bool = True,
        cross_mask_bool = True,
        pooling_type = 'mean',
        ratio = 0.1,
        RoPE_shift =True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_fixed_seq_len = use_fixed_seq_len
        self.sp_degree = sp_degree
        self.self_mask_bool = self_mask_bool
        self.cross_mask_bool = cross_mask_bool
        self.pooling_type = pooling_type
        self.ratio = ratio
        self.RoPE_shift = RoPE_shift

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # blocks
        self.blocks = nn.ModuleList([AttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps,
                    self_mask_bool = self.self_mask_bool, cross_mask_bool = self.cross_mask_bool, pooling_type = self.pooling_type, ratio = self.ratio, RoPE_shift = self.RoPE_shift) for _ in range(num_layers)])
       

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._freqs_initialized = False

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._freqs_initialized = False


    @property
    def freqs(self):
        if not self._freqs_initialized or not hasattr(self, '_freqs'):
            # 动态初始化保障逻辑
            d = self.dim // self.num_heads
            device = self.patch_embedding.weight.device
            
            # 确保在CUDA设备生成张量
            with torch.cuda.device(device):
                self._freqs = torch.cat([
                    rope_params(1024, d-4*(d//6)),
                    rope_params(1024, 2*(d//6)),
                    rope_params(1024, 2*(d//6))
                ], dim=1).to(device)
                
            self._freqs_initialized = True
        return self._freqs
    
    @amp.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        mask_text_emb,
        mask_img_emb,
    ):
        """
        x:              A list of videos each with shape [C, T, H, W].
        t:              [B].
        context:        A list of text embeddings each with shape [L, C].
        """
        # params
        
        # mask_embed = [ cond_value for cond_key, cond_value in mask_img_emb.items()]
        # mask_embed_cat = torch.cat(mask_embed, dim=2)
        # mask_text = [ cond_value for cond_key, cond_value in mask_text_emb.items()]
        # mask_text_bool = torch.stack(mask_text, dim =0) 
        # time_start = time.time()
        mask_img_embed_list = list()
        mask_text_embed_list = list()

        # cond_length = list(map(len, mask_text_emb))
        cond_length = list()
        cond_length_human = list()
        for mask_text_emb_bsi in mask_text_emb:
            length = 0
            cond_length_human_bi = dict()
            for cond_key, cond_value in mask_text_emb_bsi.items():
                if 'human_x_' in cond_key:
                    length += len(cond_value.keys())
                    cond_length_human_bi[cond_key] = len(cond_value.keys())
                else:
                    length += 1
            cond_length.append(length)
            cond_length_human.append(cond_length_human_bi)

        max_len = max(cond_length)

        for (mask_text_emb_, mask_img_emb_) in zip (mask_text_emb, mask_img_emb):
            assert mask_text_emb_.keys() == mask_img_emb_.keys()
            
            mask_text = list()
            for cond_key,  cond_value in mask_text_emb_.items():
                if 'human_x_' in cond_key:
                    mask_human_text = [ cond_human_value for cond_human_key, cond_human_value in cond_value.items()]
                    mask_text.extend(mask_human_text)
                else:
                    mask_text.append(cond_value)
            # mask_text = [ cond_value for cond_key, cond_value in mask_text_emb_.items()]
            if len(mask_text) < max_len:
                dummy_shape = mask_text[0].shape
                dummy = torch.zeros(dummy_shape, dtype=torch.bool, device=mask_text[0].device)
                mask_text += [dummy] * (max_len - len(mask_text))
            mask_text_bool_ = torch.stack(mask_text, dim =0)

            mask_embed = list()
            for cond_key,  cond_value in mask_img_emb_.items():
                if 'human_x_' in cond_key:
                    mask_human_emb = [ cond_human_value for cond_human_key, cond_human_value in cond_value.items()]
                    mask_embed.extend(mask_human_emb)
                else:
                    mask_embed.append(cond_value)
            # mask_embed = [ cond_value for cond_key, cond_value in mask_img_emb_.items()]
            if len(mask_embed) < max_len:
                dummy_shape = mask_embed[0].shape
                dummy = torch.zeros(dummy_shape, dtype=mask_embed[0].dtype, device=mask_embed[0].device)
                mask_embed += [dummy] * (max_len - len(mask_embed))
            mask_embed_cat_ = torch.cat(mask_embed, dim=1)

            mask_img_embed_list.append(mask_embed_cat_)
            mask_text_embed_list.append(mask_text_bool_)
        mask_embed_cat = torch.stack(mask_img_embed_list, dim=0)
        mask_text_bool = torch.stack(mask_text_embed_list, dim=0)

        # mask_embed_cat = mask_embed_cat[:,:,0,:,:].unsqueeze(2)
        # mask_text_bool = mask_text_bool[:,0,:].unsqueeze(1)
        # mask_text_bool = mask_text_bool[:,:,0,:,:].unsqueeze(2)
        
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        T, ori_height, ori_width = x.shape[-3:]
        N_c, ori_mask_height, ori_mask_width = mask_embed_cat.shape[-3:]
        
        x = torch.cat([x,mask_embed_cat], dim=2)        

        if ori_width % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - ori_width % self.patch_size[2]))
        if ori_height % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - ori_height % self.patch_size[1]))
        if ori_mask_width % self.patch_size[2] != 0:
            mask_embed_cat = F.pad(mask_embed_cat, (0, self.patch_size[2] - ori_width % self.patch_size[2]))
        if ori_mask_height % self.patch_size[1] != 0:
            mask_embed_cat = F.pad(mask_embed_cat, (0, 0, 0, self.patch_size[1] - ori_height % self.patch_size[1]))
        
        _, _, ot, oh, ow = x.shape
         
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in x])
        grid_sizes_list = list()
        cond_non_human_length = list()
        for (grid_size, cond_len, cond_human_len) in zip(grid_sizes, cond_length, cond_length_human):
            grid_value = grid_size.tolist()  
            # cond_non_human_len = cond_len - cond_human_len
            cond_human_len = [value for key, value in cond_human_len.items()]
            cond_non_human_len = cond_len - sum(cond_human_len)
            cond_non_human_length.append(cond_non_human_len)
            null_len = N_c - cond_len 
            split_values = [T] + [1] * cond_non_human_len + cond_human_len + [1] * null_len
            grid_size_ = torch.tensor([[split_val, grid_value[1], grid_value[2]] for split_val in split_values], dtype=torch.long)
            grid_sizes_list.append(grid_size_)
        grid_sizes = grid_sizes_list # torch.stack(grid_sizes_list)

        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        # time_end = time.time()
        
        # print(f"YUBEI Stage Time:{time_end-time_start}")
        
        tokens_num = x.shape[1]
        remainder = 0
        if self.use_fixed_seq_len:
            # use fixed seq length
            assert tokens_num <= seq_len, f"{seq_lens=}, {x[0].shape=}"
            padding_num = seq_len - tokens_num
            x = torch.cat([x, x.new_zeros(x.shape[0], padding_num, x.shape[2])], dim=1)
        else:
            remainder = tokens_num % int(self.sp_degree) #兼容sp
            if remainder != 0:
                padding_num = self.sp_degree - remainder
                x = torch.cat([x, x.new_zeros(x.shape[0], padding_num, x.shape[2])], dim=1)
        
        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context]))

        if is_enable_sequence_parallel():
            x = split(x, 1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            mask_text_bool = mask_text_bool,
            cond_length = cond_length,
            cond_non_human_length = cond_non_human_length,
            cond_length_human = cond_length_human,
        )
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, **kwargs)
        
        # head
        x = self.head(x, e)

        if is_enable_sequence_parallel():
            x = gather(x, 1)

        if remainder != 0:
            x = x[:, :-padding_num]
            
        # unpatchify
        x = self.unpatchify(x, tt, th, tw)
        
        x = x[:, :, :, :ori_height, :ori_width]
        
        return x
    
    def unpatchify(self, x, t, h, w):
        c = self.out_dim
        pt, ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nfhwpqrc->ncfphqwr", x)
        out = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        
        return out
    
    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        
        # init output layer
        nn.init.zeros_(self.head.head.weight)

@MODELS.register_module("LumosX_mixfp32")
def lumosx_t2v_mixfp32(from_pretrained=None, **kwargs):
    model = Transformer_qkv(**kwargs)
    print(f"init LumosX model by random")
    return model