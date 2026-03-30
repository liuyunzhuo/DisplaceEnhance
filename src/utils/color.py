from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


INTERNAL_PRODUCT_MATRICES = {"internal_product", "product"}


def _matrix_coefficients(matrix: str) -> Tuple[float, float, float]:
    key = matrix.lower()
    if key == "bt601":
        return 0.2990, 0.5870, 0.1140
    if key == "bt709":
        return 0.2126, 0.7152, 0.0722
    raise ValueError(f"Unsupported matrix: {matrix}")


def _is_internal_product_matrix(matrix: str) -> bool:
    return matrix.lower() in INTERNAL_PRODUCT_MATRICES


def _split_packed_yuv444_np(
    y: np.ndarray,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if u is not None and v is not None:
        return y, u, v
    if y.ndim != 3 or y.shape[2] != 3:
        raise ValueError("Expected packed HxWx3 YUV array when u and v are omitted.")
    return y[:, :, 0], y[:, :, 1], y[:, :, 2]


def bgr_to_yuv444_product(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert BGR uint8 image to YUV444 planes using the internal product formula.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr_to_yuv444_product expects an HxWx3 BGR array.")

    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    y = 0.213 * r + 0.715 * g + 0.072 * b
    u = -0.117 * r - 0.394 * g + 0.511 * b + 128.0
    v = 0.511 * r - 0.464 * g - 0.047 * b + 128.0

    return (
        np.clip(y, 0, 255).astype(np.uint8),
        np.clip(u, 0, 255).astype(np.uint8),
        np.clip(v, 0, 255).astype(np.uint8),
    )


def rgb_to_yuv444_product(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB uint8 image to YUV444 planes using the internal product formula.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_yuv444_product expects an HxWx3 RGB array.")
    return bgr_to_yuv444_product(rgb[:, :, ::-1])


def yuv444_to_bgr_product(
    y: np.ndarray,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert YUV444 planes or packed HxWx3 YUV to a BGR uint8 image
    using the internal product formula.
    """
    y_plane, u_plane, v_plane = _split_packed_yuv444_np(y, u, v)
    y_plane = y_plane.astype(np.float32)
    u_plane = u_plane.astype(np.float32)
    v_plane = v_plane.astype(np.float32)

    r = y_plane + 0.000771 * (u_plane - 128.0) + 1.540294 * (v_plane - 128.0)
    g = y_plane - 0.183095 * (u_plane - 128.0) - 0.458752 * (v_plane - 128.0)
    b = y_plane + 1.815951 * (u_plane - 128.0) - 0.001045 * (v_plane - 128.0)

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def yuv444_to_rgb_product(
    y: np.ndarray,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert YUV444 planes or packed HxWx3 YUV to an RGB uint8 image
    using the internal product formula.
    """
    return yuv444_to_bgr_product(y, u, v)[:, :, ::-1]


def luma_bt601(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def packed_yuv444_to_rgb_product(x: torch.Tensor) -> torch.Tensor:
    """
    Convert packed YUV444 tensor to RGB using the internal product formula.
    Input/output tensors are normalized to [0, 1].
    """
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError("packed_yuv444_to_rgb_product expects CHW or NCHW tensor with 3 channels.")

    y = x[:, 0:1, :, :] * 255.0
    u = x[:, 1:2, :, :] * 255.0
    v = x[:, 2:3, :, :] * 255.0

    r = y + 0.000771 * (u - 128.0) + 1.540294 * (v - 128.0)
    g = y - 0.183095 * (u - 128.0) - 0.458752 * (v - 128.0)
    b = y + 1.815951 * (u - 128.0) - 0.001045 * (v - 128.0)

    rgb = torch.cat([r, g, b], dim=1).clamp(0.0, 255.0) / 255.0
    return rgb.squeeze(0) if squeeze else rgb


def packed_yuv444_to_rgb(x: torch.Tensor, matrix: str = "bt709", value_range: str = "limited") -> torch.Tensor:
    if _is_internal_product_matrix(matrix):
        return packed_yuv444_to_rgb_product(x)

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
