from __future__ import annotations

from typing import Tuple

import torch


def _matrix_coefficients(matrix: str) -> Tuple[float, float, float]:
    key = matrix.lower()
    if key == "bt601":
        return 0.2990, 0.5870, 0.1140
    if key == "bt709":
        return 0.2126, 0.7152, 0.0722
    raise ValueError(f"Unsupported matrix: {matrix}")


def luma_bt601(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def packed_yuv444_to_rgb(x: torch.Tensor, matrix: str = "bt709", value_range: str = "limited") -> torch.Tensor:
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError("packed_yuv444_to_rgb expects CHW or NCHW tensor with 3 channels.")

    kr, kg, kb = _matrix_coefficients(matrix)
    y = x[:, 0:1, :, :]
    cb = x[:, 1:2, :, :]
    cr = x[:, 2:3, :, :]

    if value_range.lower() == "limited":
        y = torch.clamp((y * 255.0 - 16.0) / 219.0, 0.0, 1.0)
        cb = (cb * 255.0 - 128.0) / 224.0
        cr = (cr * 255.0 - 128.0) / 224.0
    elif value_range.lower() == "full":
        cb = cb - (128.0 / 255.0)
        cr = cr - (128.0 / 255.0)
    else:
        raise ValueError(f"Unsupported value range: {value_range}")

    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    rgb = torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)
    return rgb.squeeze(0) if squeeze else rgb

