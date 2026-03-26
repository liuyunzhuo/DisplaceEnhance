import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@NETWORK_REGISTRY.register()
class SharpenNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB -> YUV (BT.601)
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b

        feat = self.head(y)
        feat = self.body(feat)
        out = self.tail(feat)
        # residual learning on Y only
        y_out = torch.clamp(y + out, 0.0, 1.0)

        # YUV -> RGB
        r_out = y_out + 1.13983 * v
        g_out = y_out - 0.39465 * u - 0.58060 * v
        b_out = y_out + 2.03211 * u
        rgb = torch.cat([r_out, g_out, b_out], dim=1)
        return torch.clamp(rgb, 0.0, 1.0)
