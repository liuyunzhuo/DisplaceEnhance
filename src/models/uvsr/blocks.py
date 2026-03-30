from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "relu6":
        return nn.ReLU6(inplace=True)
    if key in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_batch_norm: bool,
        act: str,
    ) -> None:
        super().__init__()
        bias = not use_batch_norm
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(make_activation(act))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(make_activation(act))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, use_batch_norm: bool, act: str) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool,
        use_batch_norm: bool,
        act: str,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            act=act,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.channel_adjust(x)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


__all__ = ["ConvBlock", "DownBlock", "UpBlock", "make_activation"]
