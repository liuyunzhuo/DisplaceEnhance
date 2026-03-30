from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY
from .blocks import ConvBlock, DownBlock, UpBlock


@NETWORK_REGISTRY.register()
class UVSR_Unet(nn.Module):
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
        residual: bool = True,
        preserve_y_channel: bool = True,
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
            residual = bool(cfg.get("residual", residual))
            preserve_y_channel = bool(cfg.get("preserve_y_channel", preserve_y_channel))
            output_clamp = bool(cfg.get("output_clamp", output_clamp))

        if encoder_depth < 2:
            raise ValueError("encoder_depth must be at least 2 for a U-Net.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.encoder_depth = encoder_depth
        self.max_channels = max_channels
        self.bilinear = bilinear
        self.use_batch_norm = use_batch_norm
        self.act_name = act
        self.residual = residual
        self.preserve_y_channel = preserve_y_channel and in_channels == 3 and out_channels == 3
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

        head_out_channels = 2 if self.preserve_y_channel else out_channels
        self.head = nn.Conv2d(encoder_channels[0], head_out_channels, kernel_size=1)

    def split_yuv(self, yuv: torch.Tensor, mode: str = "i444p") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if yuv.dim() != 4:
            raise ValueError("split_yuv expects a 4D NCHW tensor.")
        b, c, h, w = yuv.shape
        if mode == "i420p":
            if c != 1 or h % 3 != 0 or w % 2 != 0:
                raise ValueError("i420p mode expects packed input with shape [N, 1, H, W].")
            uv_h = h // 3
            uv_w = w // 2
            y = yuv[:, :, : 2 * uv_h, :]
            uv = yuv[:, :, 2 * uv_h :, :].reshape(b, 2, uv_h, uv_w)
            u = uv[:, 0:1, :, :]
            v = uv[:, 1:2, :, :]
            return y, u, v
        if mode == "i444p":
            if c != 3:
                raise ValueError("i444p mode expects a 3-channel tensor.")
            return yuv[:, 0:1, :, :], yuv[:, 1:2, :, :], yuv[:, 2:3, :, :]
        raise ValueError(f"Unsupported mode: {mode}")

    def _merge_output(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.preserve_y_channel:
            y = x[:, 0:1, :, :]
            uv = x[:, 1:3, :, :]
            uv_out = uv + pred if self.residual else pred
            out = torch.cat([y, uv_out], dim=1)
        else:
            out = x + pred if self.residual and x.shape[1] == pred.shape[1] else pred
        if self.output_clamp:
            out = out.clamp(0.0, 1.0)
        return out

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("UVSR_Unet expects a 4D NCHW tensor.")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but received {x.size(1)}."
            )

        feat = self.forward_features(x)
        pred = self.head(feat)
        return self._merge_output(x, pred)


__all__ = ["UVSR_Unet"]
