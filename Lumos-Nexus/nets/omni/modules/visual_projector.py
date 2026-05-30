import torch
import torch.nn as nn

class VAE_Feature_Projector(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 48,
                 hidden_dim: int = 128, mid_channels: int = 64):
        super().__init__()

        self.feature_projector = nn.Sequential(
            nn.Conv2d(in_dim, mid_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_dim, kernel_size=1), 
        )

        self._init_weights()

    def _init_weights(self):       
        for layer in self.feature_projector:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, F, H, W = x.shape
            assert C == 16, f"expect 16 channels, got {C}"
            x_reshaped = x.view(B * F, C, H, W)  # [B*F, 16, H, W]
            x_out = self.feature_projector(x_reshaped)  # [B*F, 48, H/4, W/4]
            x_out = x_out.view(B, F, 48, H//2, W//2)  # [B, F, 48, H/4, W/4]
            x_out = x_out.permute(0, 2, 1, 3, 4)  # [B, 48, F, H/4, W/4]
            return x_out

        raise ValueError(f"VAE_Feature_Projector only supports 5D input [B,C,F,H,W], got {list(x.shape)}")
