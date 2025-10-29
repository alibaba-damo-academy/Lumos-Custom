import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import math

from .attention import flash_attention
from src.utils.utils_ import auto_grad_checkpoint

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
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
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
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):

    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

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
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
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
    ):
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
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


class Transformer(nn.Module):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,
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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim * 3, dim, kernel_size=patch_size, stride=patch_size
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
        self.blocks = nn.ModuleList([AttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
        ) for _ in range(num_layers)])

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
    def forward(self, x, t, context, seq_len, x_deg, x_bg):

        """
        x:              A list of videos each with shape [C, T, H, W].
        t:              [B].
        context:        A list of text embeddings each with shape [L, C].
        seq_len:        [B].
        x_deg:          A list of degree images each with shape [C, T, H, W].
        x_bg:           A list of background images each with shape [C, T, H, W].
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # print(f"x.dim:{x.shape}") [1, 16, 13, 60, 106]
        x = torch.cat([x, x_deg, x_bg], dim=1)

        T, ori_height, ori_width = x.shape[-3:]
        if ori_width % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - ori_width % self.patch_size[2]))
        if ori_height % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - ori_height % self.patch_size[1]))
        

        _, _, ot, oh, ow = x.shape
         
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        # print(f"tt = {tt}, th = {th}, tw = {tw}") tt = 13, th = 30, tw = 53
        
        # embeddings
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 16, 13, 60, 106])
        x = self.patch_embedding(x) 
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 1536, 13, 30, 53])

        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in x])
        x = x.flatten(2).transpose(1, 2)
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 20670, 1536])

        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        # print(f"seq_lens:{seq_lens}") seq_lens:tensor([20670])

        tokens_num = x.shape[1] 
        # print(f"tokens_num:{tokens_num}") tokens_num:20670

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
         
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 20670, 1536])

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )

            # print(f"e.dim:{e.shape}") e.dim:torch.Size([1, 1536])

            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # print(f"e0.dim:{e0.shape}") e0.dim:torch.Size([1, 6, 1536])

            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        # print(f"context.dim:{context.shape}") context.dim:torch.Size([1, 512, 1536])

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
        )

        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, **kwargs)
        
        # head
        x = self.head(x, e)
        
        if remainder != 0:
            x = x[:, :-padding_num]
            
        # unpatchify
        x = self.unpatchify(x, tt, th, tw)
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 16, 13, 60, 106])
        
        x = x[:, :, :, :ori_height, :ori_width]
        # print(f"x.dim:{x.shape}") x.dim:torch.Size([1, 16, 13, 60, 106])
        
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