from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

def select_channels(x: torch.Tensor, mode: str) -> torch.Tensor:
    key = mode.lower()
    if key in ("all_channels", "direct"):
        return x
    if key in ("uv_channels", "uv_only"):
        if x.size(1) < 3:
            raise ValueError("uv_channels mode expects at least 3 channels.")
        return x[:, 1:3, :, :]
    if key in ("y_channel", "y_only", "preserved_y"):
        return x[:, 0:1, :, :]
    raise ValueError(f"Unsupported loss channel mode: {mode}")


def flatten_channels_as_samples(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("Expected NCHW tensor.")
    n, c, h, w = x.shape
    return x.reshape(n * c, 1, h, w)


def _get_pytorch_msssim() -> Any:
    try:
        from pytorch_msssim import ms_ssim as package_ms_ssim
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "MS-SSIM loss requires the 'pytorch-msssim' package. "
            "Install it with: pip install pytorch-msssim"
        ) from exc
    return package_ms_ssim


def gradient_map(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_gx, pred_gy = gradient_map(pred)
    target_gx, target_gy = gradient_map(target)
    loss_x = torch.abs(pred_gx - target_gx).mean()
    loss_y = torch.abs(pred_gy - target_gy).mean()
    return 0.5 * (loss_x + loss_y)


@dataclass
class LossTerm:
    name: str
    type: str
    weight: float
    mode: str
    single_channel: bool = False
    data_range: float = 1.0
    normalize_before_loss: bool = False
    input_min: float = 0.0
    input_max: float = 1.0


def build_loss_terms(loss_opt: Dict[str, Any]) -> List[LossTerm]:
    terms_opt = loss_opt.get("terms") or []
    if not isinstance(terms_opt, list) or not terms_opt:
        raise ValueError("Composite loss requires a non-empty train.loss.terms list.")

    terms: List[LossTerm] = []
    for idx, term_opt in enumerate(terms_opt, start=1):
        if not isinstance(term_opt, dict):
            raise ValueError(f"train.loss.terms[{idx}] must be a mapping.")
        term_type = str(term_opt.get("type", "")).strip()
        if not term_type:
            raise ValueError(f"train.loss.terms[{idx}] is missing type.")
        term_name = str(term_opt.get("name", f"term_{idx}"))
        terms.append(
            LossTerm(
                name=term_name,
                type=term_type,
                weight=float(term_opt.get("weight", 1.0)),
                mode=str(term_opt.get("mode", "all_channels")),
                single_channel=bool(term_opt.get("single_channel", False)),
                data_range=float(term_opt.get("data_range", 1.0)),
                normalize_before_loss=bool(term_opt.get("normalize_before_loss", False)),
                input_min=float(term_opt.get("input_min", 0.0)),
                input_max=float(term_opt.get("input_max", 1.0)),
            )
        )
    return terms


def normalize_to_unit_interval(x: torch.Tensor, input_min: float, input_max: float) -> torch.Tensor:
    if input_max <= input_min:
        raise ValueError("normalize_to_unit_interval requires input_max > input_min.")
    return torch.clamp((x - input_min) / (input_max - input_min), 0.0, 1.0)


def compute_term_loss(term: LossTerm, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sel = select_channels(pred, term.mode)
    target_sel = select_channels(target, term.mode)
    if term.normalize_before_loss:
        pred_sel = normalize_to_unit_interval(pred_sel, term.input_min, term.input_max)
        target_sel = normalize_to_unit_interval(target_sel, term.input_min, term.input_max)
    term_type = term.type.upper().replace("-", "_")
    term_key = term_type.replace("_", "")

    if term_key == "MSSSIM":
        package_ms_ssim = _get_pytorch_msssim()
        if term.single_channel:
            if pred_sel.dim() != 4:
                raise ValueError("MS-SSIM single_channel mode expects a 4D NCHW tensor.")
            channel_losses = []
            for channel_idx in range(pred_sel.size(1)):
                pred_channel = pred_sel[:, channel_idx : channel_idx + 1, :, :]
                target_channel = target_sel[:, channel_idx : channel_idx + 1, :, :]
                channel_losses.append(1.0 - package_ms_ssim(pred_channel, target_channel, data_range=term.data_range))
            return torch.stack(channel_losses).mean()
        score = package_ms_ssim(pred_sel, target_sel, data_range=term.data_range)
        return 1.0 - score

    if term_key in ("GRADIENTL1", "TVL1", "TVLOSS"):
        return gradient_l1_loss(pred_sel, target_sel)

    raise ValueError(f"Unsupported composite loss term type: {term.type}")
