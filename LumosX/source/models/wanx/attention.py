import torch
import math
from magi_attention.common import AttnRanges
from magi_attention.functional import  flex_flash_attn_func

from flash_attn import flash_attn_varlen_func

__all__ = [
    'flash_attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    q_ranges = None,
    k_ranges = None,
    attn_type_map = None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    mask=None,
    self_mask_bool = False,
    cross_mask_bool = True
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 1800#256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    def half(x): return x if x.dtype in half_dtypes else x.to(dtype)
    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32
        ).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32
        ).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    # apply attention
    
    if mask is None:
    ############### magi attention ###############
    # q_ranges =AttnRanges().from_ranges([[0,10933],[10933,22620],[22620, 32799],[32799,43732]]).to_tensor(device=q.device)
    # attn_type_map=torch.tensor([0, 0, 0, 0], device=q.device, dtype=torch.int32)  # we support different mask type for different qk ranges.
        if self_mask_bool:
            x, _ = flex_flash_attn_func(q, 
                            k, 
                            v, 
                            q_ranges, 
                            k_ranges, 
                            max_seqlen_q=lq, 
                            max_seqlen_k=lk, 
                            attn_type_map=attn_type_map, 
                            disable_fwd_atomic_reduction=True,)
            splits = [int(q_len) for q_len in q_lens]
            x_ = torch.split(x, splits, dim=0)
            x_new =[torch.nn.functional.pad(x_x, (0, 0, 0, 0, 0, lq-x_x.size(0))) for x_x in x_]
            x = torch.stack(x_new, dim=0)
        else: 
        ############# flash attention ###############
            x = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([
                    q_lens.new_zeros([1]), q_lens
                ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([
                    k_lens.new_zeros([1]), k_lens
                ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )
            splits = [int(q_len) for q_len in q_lens]
            x_ = torch.split(x, splits, dim=0)
            x_new =[torch.nn.functional.pad(x_x, (0, 0, 0, 0, 0, lq-x_x.size(0))) for x_x in x_]
            x = torch.stack(x_new, dim=0) 
    ######################################################################
        # x = torch.nn.functional.scaled_dot_product_attention(q.reshape(b,-1, *q.shape[1:]).transpose(1,2), k.reshape(b,-1, *k.shape[1:]).transpose(1,2), v.reshape(b,-1, *v.shape[1:]).transpose(1,2), is_causal=False ).transpose(1,2) 
    else:
        if cross_mask_bool:
            x = torch.nn.functional.scaled_dot_product_attention(q.reshape(b,-1, *q.shape[1:]).transpose(1,2), k.reshape(b,-1, *k.shape[1:]).transpose(1,2), v.reshape(b,-1, *v.shape[1:]).transpose(1,2), is_causal=False,  attn_mask = mask.transpose(1,2) ).transpose(1,2) 
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q.reshape(b,-1, *q.shape[1:]).transpose(1,2), k.reshape(b,-1, *k.shape[1:]).transpose(1,2), v.reshape(b,-1, *v.shape[1:]).transpose(1,2), is_causal=False ).transpose(1,2) 
    #     # x = torch.nn.functional.scaled_dot_product_attention(q.reshape(b,-1, *q.shape[1:]).transpose(1,2), k.reshape(b,-1, *k.shape[1:]).transpose(1,2), v.reshape(b,-1, *v.shape[1:]).transpose(1,2), is_causal=False,  attn_mask = mask.unsqueeze(1) ).transpose(1,2) 

           
    return x.type(out_dtype)

