from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY
from .blocks import ConvBlock, DownBlock, UpBlock


@NETWORK_REGISTRY.register()
class UVSR_YUV_Unet(nn.Module):
    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        encoder_depth: int = 4,
        max_channels: int = 256,
        bilinear: bool = True,
        use_batch_norm: bool = False,
        act: str = "relu",
        residual_y: bool = True,
        residual_uv: bool = True,
        output_clamp: bool = False,
    ) -> None:
        super().__init__()

        if cfg is not None:
            in_channels = int(cfg.get("in_channels", in_channels))
            out_channels = int(cfg.get("out_channels", out_channels))
            base_channels = int(cfg.get("base_channels", cfg.get("first_out_c", base_channels)))
            encoder_depth = int(cfg.get("encoder_depth", encoder_depth))
            max_channels = int(cfg.get("max_channels", max_channels))
            bilinear = bool(cfg.get("bilinear", bilinear))
            use_batch_norm = bool(cfg.get("use_batch_norm", cfg.get("bn", use_batch_norm)))
            act = str(cfg.get("act", act))
            residual_y = bool(cfg.get("residual_y", residual_y))
            residual_uv = bool(cfg.get("residual_uv", residual_uv))
            output_clamp = bool(cfg.get("output_clamp", output_clamp))

        if in_channels != 3 or out_channels != 3:
            raise ValueError("UVSR_YUV_Unet expects 3-channel packed YUV444 input/output.")
        if encoder_depth < 2:
            raise ValueError("encoder_depth must be at least 2 for a U-Net.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.encoder_depth = encoder_depth
        self.max_channels = max_channels
        self.bilinear = bilinear
        self.use_batch_norm = use_batch_norm
        self.act_name = act
        self.residual_y = residual_y
        self.residual_uv = residual_uv
        self.output_clamp = output_clamp

        encoder_channels: list[int] = []
        current_channels = base_channels
        for _ in range(encoder_depth):
            encoder_channels.append(min(current_channels, max_channels))
            current_channels *= 2

        self.stem = ConvBlock(
            in_channels,
            encoder_channels[0],
            use_batch_norm=use_batch_norm,
            act=act,
        )

        self.down_blocks = nn.ModuleList()
        for idx in range(1, len(encoder_channels)):
            self.down_blocks.append(
                DownBlock(
                    encoder_channels[idx - 1],
                    encoder_channels[idx],
                    use_batch_norm=use_batch_norm,
                    act=act,
                )
            )

        bottleneck_channels = min(encoder_channels[-1] * 2, max_channels)
        self.bottleneck = DownBlock(
            encoder_channels[-1],
            bottleneck_channels,
            use_batch_norm=use_batch_norm,
            act=act,
        )

        self.up_blocks = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.up_blocks.append(
                UpBlock(
                    decoder_in_channels,
                    skip_channels,
                    skip_channels,
                    bilinear=bilinear,
                    use_batch_norm=use_batch_norm,
                    act=act,
                )
            )
            decoder_in_channels = skip_channels

        self.y_head = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)
        self.uv_head = nn.Conv2d(encoder_channels[0], 2, kernel_size=1)

    def split_yuv(self, yuv: torch.Tensor, mode: str = "i444p") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if yuv.dim() != 4:
            raise ValueError("split_yuv expects a 4D NCHW tensor.")
        if mode == "i444p":
            if yuv.size(1) != 3:
                raise ValueError("i444p mode expects a 3-channel tensor.")
            return yuv[:, 0:1, :, :], yuv[:, 1:2, :, :], yuv[:, 2:3, :, :]
        raise ValueError(f"Unsupported mode: {mode}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        feat = self.stem(x)
        skips.append(feat)
        for down in self.down_blocks:
            feat = down(feat)
            skips.append(feat)

        feat = self.bottleneck(skips[-1])
        for up, skip in zip(self.up_blocks, reversed(skips)):
            feat = up(feat, skip)
        return feat

    def _merge_output(self, x: torch.Tensor, pred_y: torch.Tensor, pred_uv: torch.Tensor) -> torch.Tensor:
        y = x[:, 0:1, :, :]
        uv = x[:, 1:3, :, :]
        y_out = y + pred_y if self.residual_y else pred_y
        uv_out = uv + pred_uv if self.residual_uv else pred_uv
        out = torch.cat([y_out, uv_out], dim=1)
        if self.output_clamp:
            out = out.clamp(0.0, 1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("UVSR_YUV_Unet expects a 4D NCHW tensor.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but received {x.size(1)}.")

        feat = self.forward_features(x)
        pred_y = self.y_head(feat)
        pred_uv = self.uv_head(feat)
        return self._merge_output(x, pred_y, pred_uv)


__all__ = ["UVSR_YUV_Unet"]
