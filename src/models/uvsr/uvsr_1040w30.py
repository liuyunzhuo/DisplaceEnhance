from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from torch import nn

from src.training.registry import NETWORK_REGISTRY
from .blocks import make_activation


def _make_act(act: str | nn.Module) -> nn.Module:
    if isinstance(act, nn.Module):
        return copy.deepcopy(act)
    return make_activation(str(act))


class EConvbBNV12(nn.Module):
    def __init__(
        self,
        *,
        in_c: int,
        mid_c: int,
        out_c: int,
        act: str | nn.Module,
        out_act: bool,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.act_layer = _make_act(act)
        self.out_act = out_act
        self.conv1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=mid_c,
            kernel_size=k,
            stride=1,
            padding=k // 2,
        )
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(
            in_channels=mid_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act_layer(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.out_act:
            out = self.act_layer(out)
        return out


def make_bn_bias_layer(channels: int, bias_val: float) -> nn.BatchNorm2d:
    layer = nn.BatchNorm2d(channels, eps=0.0)
    layer.running_mean = nn.Parameter(torch.zeros(channels, dtype=torch.float32), requires_grad=False)
    layer.running_var = nn.Parameter(torch.ones(channels, dtype=torch.float32), requires_grad=False)
    layer.weight = nn.Parameter(torch.ones(channels, dtype=torch.float32), requires_grad=False)
    layer.bias = nn.Parameter(torch.full((channels,), float(bias_val), dtype=torch.float32), requires_grad=False)
    return layer


def make_uvsr_pixshuffle(requires_grad: bool = False) -> nn.ConvTranspose2d:
    weight = torch.zeros(8, 2, 2, 2, dtype=torch.float32)
    for i in range(4):
        weight[i, 0, int(i / 2), int(i % 2)] = 1.0
        weight[i + 4, 1, int(i / 2), int(i % 2)] = 1.0

    layer = nn.ConvTranspose2d(
        8,
        2,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=False,
    )
    layer.weight = nn.Parameter(weight, requires_grad=requires_grad)
    return layer


@NETWORK_REGISTRY.register()
class UVSR_1040W30(nn.Module):
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
            output_clamp = bool(cfg.get("output_clamp", output_clamp))

        if in_channels != 3 or out_channels != 3:
            raise ValueError("UVSR_1040W30 expects 3-channel packed YUV444 input/output.")
        if down_out_c != 12:
            raise ValueError("UVSR_1040W30 currently supports only down_out_c=12.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_out_c = down_out_c
        self.first_out_c = first_out_c
        self.hidden_out_c = hidden_out_c
        self.mid_channels = mid_channels
        self.act_name = act
        self.bn_bias = bn_bias
        self.output_clamp = output_clamp

        act_layer = make_activation(act)

        self.down_layer = self._get_dwt_layer(self.down_out_c)
        self.skip_layer = self._get_skip_layer()
        self.first_layer = EConvbBNV12(
            act=act_layer,
            in_c=self.down_out_c,
            mid_c=self.mid_channels,
            out_c=self.first_out_c,
            out_act=True,
        )
        self.hidden_layer = EConvbBNV12(
            act=act_layer,
            in_c=self.first_out_c,
            mid_c=self.mid_channels,
            out_c=self.hidden_out_c,
            out_act=True,
        )
        self.last_layer = EConvbBNV12(
            act=act_layer,
            in_c=self.hidden_out_c,
            mid_c=self.mid_channels,
            out_c=8,
            out_act=False,
        )
        self.upscale_layer = make_uvsr_pixshuffle(requires_grad=False)
        self.bias_layer = make_bn_bias_layer(channels=8, bias_val=128.0) if self.bn_bias else nn.Identity()

    def _get_skip_layer(self) -> nn.Conv2d:
        weight = torch.zeros((8, 3, 2 * 2), dtype=torch.float32)
        for i in range(4):
            weight[i, 1, i] = 1.0
            weight[i + 4, 2, i] = 1.0

        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        layer.weight = nn.Parameter(weight.reshape(8, 3, 2, 2), requires_grad=False)
        return layer

    def _get_dwt_layer(self, out_c: int = 12) -> nn.Conv2d:
        weight = torch.zeros((out_c, 3, 2 * 2), dtype=torch.float32)
        dwt_filter = torch.tensor(
            (
                [[0.5, 0.5, 0.5, 0.5]],
                [[-0.5, 0.5, -0.5, 0.5]],
                [[-0.5, -0.5, 0.5, 0.5]],
                [[0.5, -0.5, -0.5, 0.5]],
            ),
            dtype=torch.float32,
        )
        dwt_ft_sz = int(dwt_filter.shape[0])
        weight[:dwt_ft_sz, :1, :] = dwt_filter
        extra_c = (out_c - dwt_ft_sz) // 2
        for i in range(extra_c):
            weight[dwt_ft_sz + i, 1:2, i % 4] = 1.0
            weight[dwt_ft_sz + extra_c + i, 2:3, i % 4] = 1.0

        layer = nn.Conv2d(
            in_channels=3,
            out_channels=out_c,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        layer.weight = nn.Parameter(weight.reshape(out_c, 3, 2, 2), requires_grad=False)
        return layer

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
            raise ValueError("UVSR_1040W30 expects a 4D NCHW tensor.")
        if yuv.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but received {yuv.size(1)}.")

        down_fea = self.down_layer(yuv)
        skip_fea = self.skip_layer(yuv)
        fea_first = self.first_layer(down_fea)
        fea_hidden = self.hidden_layer(fea_first)
        fea_last = self.last_layer(fea_hidden)

        out = fea_last + skip_fea
        out = self.bias_layer(out)
        sr_uv = self.upscale_layer(out)
        sr_out = torch.cat((yuv[:, 0:1, :, :], sr_uv), dim=1)
        if self.output_clamp:
            sr_out = sr_out.clamp(-128.0, 127.0)
        return sr_out


__all__ = ["UVSR_1040W30"]
