import torch
import torch.nn as nn

class VisualContextAdapter(nn.Module):
    """
    Visual Context Adapter for processing VAE latent features.
    
    This module takes VAE latent features as input and outputs a learned flattened feature
    with an explicitly set output dimension. It uses a patchify embedding to convert the input
    VAE feature to a downsampled representation.

    Usage:
    ```python
    adapter = VisualContextAdapter(patch_size=(1, 2, 2), in_channels=16, hidden_dim=512, out_dim=4098)
    input_latent = [torch.randn(16, 21, 32, 32) for _ in range(4)]  # 4 latent features
    output_feature = adapter(input_latent)
    ```
    """
    
    def __init__(
        self,
        patch_size=(1, 4, 4),
        in_channels=16,
        hidden_dim=2048,
        out_dim=4098,
        eps=1e-6,
    ):
        """
        Initialize the Visual Context Adapter.
        
        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for latent embedding (t_patch, h_patch, w_patch)
            in_channels (`int`, *optional*, defaults to 16):
                Input latent channels from VAE
            hidden_dim (`int`, *optional*, defaults to 512):
                Hidden dimension of the adapter
            out_dim (`int`, *optional*, defaults to 1024):
                Output dimension of the flattened feature
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.eps = eps
        
        # Patchify embedding to downsample the input
        self.patch_embedding = nn.Conv3d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Normalization and projection to output dimension
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False) # set to False
        self.projection = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim, eps=eps, elementwise_affine=False) # set to False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the module"""
        if isinstance(module, nn.Conv3d):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.flatten(1))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, std=.02)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the adapter.
        
        Args:
            x (List[Tensor] or Tensor):
                If List: List of input latent tensors, each with shape [C_in, F, H, W]
                If Tensor: Batched input with shape [B, C_in, F, H, W]
                
        Returns:
            Tensor: Flattened feature tensor with shape [B, N, out_dim]
                where N is the number of patches
        """
        # Handle both list and tensor inputs
        is_list_input = isinstance(x, list)
        
        if is_list_input:
            # Convert list of tensors to batched tensor
            x = torch.stack([tensor for tensor in x])
        elif x.dim() == 4:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            
        # Apply patch embedding
        x = self.patch_embedding(x)
        
        # Get batch size and grid dimensions
        batch_size = x.size(0)
        grid_sizes = x.shape[2:5]  # [F, H, W] after patching
        
        # Flatten spatial dimensions and transpose to [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        
        # Apply normalization and projection
        x = self.norm1(x)
        x = self.projection(x)
        # Apply final normalization
        x = self.norm2(x)
        
        return x