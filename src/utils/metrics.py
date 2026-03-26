import math

import torch


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)
