from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY
from .uvsr_1040w30 import EConvbBNV12, make_bn_bias_layer, make_uvsr_pixshuffle


def make_y_pixshuffle(requires_grad: bool = False) -> nn.ConvTranspose2d:
    weight = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    for i in range(4):
        weight[i, 0, int(i / 2), int(i % 2)] = 1.0

    layer = nn.ConvTranspose2d(
        4,
        1,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=False,
    )
    layer.weight = nn.Parameter(weight, requires_grad=requires_grad)
    return layer


@NETWORK_REGISTRY.register()
class UVSR_1040W30_YUV(nn.Module):
    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        down_out_c: int = 12,
        first_out_c: int = 16,
        hidden_out_c: int = 16,
        mid_channels: int = 32,
        act: str = "relu",
        bn_bias: bool = False,
        residual_y: bool = True,
        residual_uv: bool = True,
        output_clamp: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()

        if cfg is not None:
            in_channels = int(cfg.get("in_channels", in_channels))
            out_channels = int(cfg.get("out_channels", out_channels))
            down_out_c = int(cfg.get("down_out_c", down_out_c))
            first_out_c = int(cfg.get("first_out_c", first_out_c))
            hidden_out_c = int(cfg.get("hidden_out_c", hidden_out_c))
            mid_channels = int(cfg.get("mid_channels", mid_channels))
            act = str(cfg.get("act", act))
            bn_bias = bool(cfg.get("bn_bias", bn_bias))
            residual_y = bool(cfg.get("residual_y", residual_y))
            residual_uv = bool(cfg.get("residual_uv", residual_uv))
            output_clamp = bool(cfg.get("output_clamp", output_clamp))

        if in_channels != 3 or out_channels != 3:
            raise ValueError("UVSR_1040W30_YUV expects 3-channel packed YUV444 input/output.")
        if down_out_c != 12:
            raise ValueError("UVSR_1040W30_YUV currently supports only down_out_c=12.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_out_c = down_out_c
        self.first_out_c = first_out_c
        self.hidden_out_c = hidden_out_c
        self.mid_channels = mid_channels
        self.act_name = act
        self.bn_bias = bn_bias
        self.residual_y = residual_y
        self.residual_uv = residual_uv
        self.output_clamp = output_clamp

        # Reuse the original 1040W30 fixed analysis path.
        from .uvsr_1040w30 import UVSR_1040W30

        ref = UVSR_1040W30(
            in_channels=in_channels,
            out_channels=out_channels,
            down_out_c=down_out_c,
            first_out_c=first_out_c,
            hidden_out_c=hidden_out_c,
            mid_channels=mid_channels,
            act=act,
            bn_bias=bn_bias,
            output_clamp=False,
        )
        self.down_layer = ref.down_layer
        self.skip_layer = ref.skip_layer

        self.first_layer = EConvbBNV12(
            act=act,
            in_c=self.down_out_c,
            mid_c=self.mid_channels,
            out_c=self.first_out_c,
            out_act=True,
        )
        self.hidden_layer = EConvbBNV12(
            act=act,
            in_c=self.first_out_c,
            mid_c=self.mid_channels,
            out_c=self.hidden_out_c,
            out_act=True,
        )

        self.uv_last_layer = EConvbBNV12(
            act=act,
            in_c=self.hidden_out_c,
            mid_c=self.mid_channels,
            out_c=8,
            out_act=False,
        )
        self.y_last_layer = EConvbBNV12(
            act=act,
            in_c=self.hidden_out_c,
            mid_c=self.mid_channels,
            out_c=4,
            out_act=False,
        )
        self.uv_upscale_layer = make_uvsr_pixshuffle(requires_grad=False)
        self.y_upscale_layer = make_y_pixshuffle(requires_grad=False)
        self.bias_layer = make_bn_bias_layer(channels=8, bias_val=128.0) if self.bn_bias else nn.Identity()

    def split_yuv(self, yuv: torch.Tensor, mode: str = "i420p") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if yuv.dim() != 4:
            raise ValueError("split_yuv expects a 4D NCHW tensor.")
        b, c, h, w = yuv.shape
        if mode == "i420p":
            uv_h = int(h / 3)
            uv_w = int(w / 2)
            y = yuv[:, 0:1, 0 : (2 * uv_h), :]
            uv = torch.reshape(yuv[:, 0:1, (2 * uv_h) :, :], (b, c * 2, uv_h, uv_w))
            u = uv[:, 0:1, :, :]
            v = uv[:, 1:2, :, :]
            return y, u, v
        if mode == "i444p":
            return yuv[:, 0:1, :, :], yuv[:, 1:2, :, :], yuv[:, 2:3, :, :]
        raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, yuv: torch.Tensor) -> torch.Tensor:
        if yuv.dim() != 4:
            raise ValueError("UVSR_1040W30_YUV expects a 4D NCHW tensor.")
        if yuv.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but received {yuv.size(1)}.")

        down_fea = self.down_layer(yuv)
        skip_fea = self.skip_layer(yuv)
        fea_first = self.first_layer(down_fea)
        fea_hidden = self.hidden_layer(fea_first)

        uv_last = self.uv_last_layer(fea_hidden)
        uv_mid = uv_last + skip_fea
        uv_mid = self.bias_layer(uv_mid)
        sr_uv = self.uv_upscale_layer(uv_mid)

        y_last = self.y_last_layer(fea_hidden)
        sr_y = self.y_upscale_layer(y_last)

        y_in = yuv[:, 0:1, :, :]
        uv_in = yuv[:, 1:3, :, :]
        y_out = y_in + sr_y if self.residual_y else sr_y
        uv_out = uv_in + sr_uv if self.residual_uv else sr_uv
        out = torch.cat((y_out, uv_out), dim=1)
        if self.output_clamp:
            out = out.clamp(-128.0, 127.0)
        return out


__all__ = ["UVSR_1040W30_YUV"]
