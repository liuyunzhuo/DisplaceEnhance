import math

import torch


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


@torch.no_grad()
def psnr_per_channel(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("psnr_per_channel expects pred and target to have the same shape.")
    if pred.dim() != 4:
        raise ValueError("psnr_per_channel expects a 4D NCHW tensor.")
    mse = torch.mean((pred - target) ** 2, dim=(0, 2, 3))
    max_tensor = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    safe_mse = torch.clamp(mse, min=torch.finfo(pred.dtype).eps)
    channel_psnr = 20 * torch.log10(max_tensor) - 10 * torch.log10(safe_mse)
    channel_psnr = torch.where(mse == 0, torch.full_like(channel_psnr, 100.0), channel_psnr)
    return channel_psnr
