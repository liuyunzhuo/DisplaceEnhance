from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY
from .blocks import make_activation
from .uvsr_1040w30 import make_fixed_dwt_downsample


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        act: str,
        use_batch_norm: bool,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        bias = not use_batch_norm
        layers: list[nn.Module] = [
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=bias),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(hidden))
        layers.append(make_activation(act))
        layers.append(nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=bias))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        *,
        act: str,
        use_batch_norm: bool,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock(
                    channels,
                    act=act,
                    use_batch_norm=use_batch_norm,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            ]
        )
        bias = not use_batch_norm
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(self.blocks(x))


@NETWORK_REGISTRY.register()
class UVSR_SharedBranchNet(nn.Module):
    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        stem_channels: int = 64,
        shared_blocks: int = 8,
        y_blocks: int = 4,
        y_base_blocks: int = 0,
        y_detail_desktop_blocks: int = 0,
        y_detail_natural_blocks: int = 0,
        y_router_blocks: int = 0,
        uv_blocks: int = 4,
        process_scale: int = 2,
        downsample_type: str = "fixed_dwt_downsample",
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
            base_channels = int(cfg.get("base_channels", base_channels))
            stem_channels = int(cfg.get("stem_channels", cfg.get("first_out_c", stem_channels)))
            shared_blocks = int(cfg.get("shared_blocks", shared_blocks))
            y_blocks = int(cfg.get("y_blocks", y_blocks))
            y_base_blocks = int(cfg.get("y_base_blocks", y_base_blocks))
            y_detail_desktop_blocks = int(
                cfg.get("y_detail_desktop_blocks", cfg.get("y_detail_blocks", y_detail_desktop_blocks))
            )
            y_detail_natural_blocks = int(
                cfg.get("y_detail_natural_blocks", cfg.get("y_detail_blocks", y_detail_natural_blocks))
            )
            y_router_blocks = int(cfg.get("y_router_blocks", cfg.get("y_mask_blocks", y_router_blocks)))
            uv_blocks = int(cfg.get("uv_blocks", uv_blocks))
            process_scale = int(cfg.get("process_scale", process_scale))
            downsample_type = str(cfg.get("downsample_type", downsample_type))
            use_batch_norm = bool(cfg.get("use_batch_norm", cfg.get("bn", use_batch_norm)))
            act = str(cfg.get("act", act))
            residual_y = bool(cfg.get("residual_y", residual_y))
            residual_uv = bool(cfg.get("residual_uv", residual_uv))
            output_clamp = bool(cfg.get("output_clamp", output_clamp))

        if in_channels != 3 or out_channels != 3:
            raise ValueError("UVSR_SharedBranchNet expects 3-channel packed YUV444 input/output.")
        if process_scale != 2:
            raise ValueError("UVSR_SharedBranchNet currently supports only process_scale=2.")
        if y_base_blocks <= 0:
            y_base_blocks = y_blocks
        if y_detail_desktop_blocks <= 0:
            y_detail_desktop_blocks = y_blocks
        if y_detail_natural_blocks <= 0:
            y_detail_natural_blocks = y_blocks
        if y_router_blocks <= 0:
            y_router_blocks = max(1, y_blocks // 2)

        if (
            shared_blocks <= 0
            or y_base_blocks <= 0
            or y_detail_desktop_blocks <= 0
            or y_detail_natural_blocks <= 0
            or y_router_blocks <= 0
            or uv_blocks <= 0
        ):
            raise ValueError("shared/y/uv block counts must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.stem_channels = stem_channels
        self.shared_blocks = shared_blocks
        self.y_blocks = y_blocks
        self.y_base_blocks = y_base_blocks
        self.y_detail_desktop_blocks = y_detail_desktop_blocks
        self.y_detail_natural_blocks = y_detail_natural_blocks
        self.y_router_blocks = y_router_blocks
        self.uv_blocks = uv_blocks
        self.process_scale = process_scale
        self.downsample_type = downsample_type
        self.use_batch_norm = use_batch_norm
        self.act_name = act
        self.residual_y = residual_y
        self.residual_uv = residual_uv
        self.output_clamp = output_clamp
        self.last_router_logits: torch.Tensor | None = None
        self.last_router_weights: torch.Tensor | None = None

        bias = not use_batch_norm
        if self.downsample_type == "fixed_dwt_downsample":
            self.fixed_downsample = make_fixed_dwt_downsample(out_c=12)
            stem_in_channels = 12
        elif self.downsample_type == "learned":
            self.fixed_downsample = None
            stem_in_channels = in_channels
        else:
            raise ValueError(f"Unsupported downsample_type: {self.downsample_type}")

        stem_layers: list[nn.Module] = [
            nn.Conv2d(stem_in_channels, stem_channels, kernel_size=3, padding=1, bias=bias),
        ]
        if use_batch_norm:
            stem_layers.append(nn.BatchNorm2d(stem_channels))
        stem_layers.append(make_activation(act))
        stem_layers.append(nn.Conv2d(stem_channels, base_channels, kernel_size=3, padding=1, bias=bias))
        if use_batch_norm:
            stem_layers.append(nn.BatchNorm2d(base_channels))
        stem_layers.append(make_activation(act))
        self.stem = nn.Sequential(*stem_layers)

        self.shared_trunk = ResidualStack(
            base_channels,
            shared_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )

        self.y_base_adapter = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=bias)
        self.y_detail_desktop_adapter = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=bias)
        self.y_detail_natural_adapter = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=bias)
        self.y_router_adapter = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=bias)
        self.uv_adapter = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=bias)

        self.y_base_branch = ResidualStack(
            base_channels,
            y_base_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )
        self.y_detail_desktop_branch = ResidualStack(
            base_channels,
            y_detail_desktop_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )
        self.y_detail_natural_branch = ResidualStack(
            base_channels,
            y_detail_natural_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )
        self.y_router_branch = ResidualStack(
            base_channels,
            y_router_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )
        self.uv_branch = ResidualStack(
            base_channels,
            uv_blocks,
            act=act,
            use_batch_norm=use_batch_norm,
        )

        self.y_base_head = nn.Conv2d(base_channels, 4, kernel_size=3, padding=1)
        self.y_detail_desktop_head = nn.Conv2d(base_channels, 4, kernel_size=3, padding=1)
        self.y_detail_natural_head = nn.Conv2d(base_channels, 4, kernel_size=3, padding=1)
        self.y_router_head = nn.Conv2d(base_channels, 8, kernel_size=3, padding=1)
        self.uv_head = nn.Conv2d(base_channels, 8, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.process_scale)

    def split_yuv(self, yuv: torch.Tensor, mode: str = "i444p") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if yuv.dim() != 4:
            raise ValueError("split_yuv expects a 4D NCHW tensor.")
        if mode != "i444p":
            raise ValueError(f"Unsupported mode: {mode}")
        if yuv.size(1) != 3:
            raise ValueError("i444p mode expects a 3-channel tensor.")
        return yuv[:, 0:1, :, :], yuv[:, 1:2, :, :], yuv[:, 2:3, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("UVSR_SharedBranchNet expects a 4D NCHW tensor.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but received {x.size(1)}.")
        if x.size(-2) % self.process_scale != 0 or x.size(-1) % self.process_scale != 0:
            raise ValueError(
                f"Input spatial size must be divisible by process_scale={self.process_scale}, "
                f"got {tuple(x.shape[-2:])}."
            )

        if self.fixed_downsample is not None:
            feat = self.fixed_downsample(x)
        else:
            feat = x
        feat = self.stem(feat)
        shared = self.shared_trunk(feat)

        y_base_feat = self.y_base_branch(self.y_base_adapter(shared))
        y_detail_desktop_feat = self.y_detail_desktop_branch(self.y_detail_desktop_adapter(shared))
        y_detail_natural_feat = self.y_detail_natural_branch(self.y_detail_natural_adapter(shared))
        y_router_feat = self.y_router_branch(self.y_router_adapter(shared))
        uv_feat = self.uv_branch(self.uv_adapter(shared))

        pred_y_base = self.pixel_shuffle(self.y_base_head(y_base_feat))
        pred_y_detail_desktop = self.pixel_shuffle(self.y_detail_desktop_head(y_detail_desktop_feat))
        pred_y_detail_natural = self.pixel_shuffle(self.y_detail_natural_head(y_detail_natural_feat))
        router_logits = self.pixel_shuffle(self.y_router_head(y_router_feat))
        self.last_router_logits = router_logits
        router_weights = torch.softmax(router_logits, dim=1)
        self.last_router_weights = router_weights
        pred_y = (
            pred_y_base
            + router_weights[:, 0:1, :, :] * pred_y_detail_desktop
            + router_weights[:, 1:2, :, :] * pred_y_detail_natural
        )
        pred_uv = self.pixel_shuffle(self.uv_head(uv_feat))

        y_in = x[:, 0:1, :, :]
        uv_in = x[:, 1:3, :, :]
        y_out = y_in + pred_y if self.residual_y else pred_y
        uv_out = uv_in + pred_uv if self.residual_uv else pred_uv
        out = torch.cat([y_out, uv_out], dim=1)
        if self.output_clamp:
            out = out.clamp(-128.0, 127.0)
        return out


__all__ = ["UVSR_SharedBranchNet"]
