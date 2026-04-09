from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401,E402
from src.training.config_loader import load_experiment_config  # noqa: E402
from src.training.registry import NETWORK_REGISTRY  # noqa: E402


@dataclass
class LayerStat:
    name: str
    layer_type: str
    output_shape: Tuple[int, ...]
    macs: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile model params and MACs from a training config.")
    parser.add_argument("--opt", required=True, help="Training config path.")
    parser.add_argument("--height", type=int, default=None, help="Input height. If omitted, try to infer from config.")
    parser.add_argument("--width", type=int, default=None, help="Input width. If omitted, try to infer from config.")
    parser.add_argument("--batch_size", type=int, default=1, help="Profile batch size. Default: 1.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Device used for dummy forward.")
    parser.add_argument("--details", action="store_true", help="Print per-layer MAC details.")
    return parser.parse_args()


def _infer_spatial_size(opt: Dict) -> Tuple[int, int]:
    datasets_opt = opt.get("datasets", {})
    train_opt = datasets_opt.get("train", {})
    dataset_list = train_opt.get("datasets")
    if dataset_list:
        dataset_opt = dataset_list[0]
    else:
        dataset_opt = train_opt
    pipeline = dataset_opt.get("pipeline", []) or []
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get("name") == "crop":
            params = step.get("params", {}) or {}
            size = params.get("size")
            if size is not None:
                return int(size), int(size)
        if step.get("name") == "resize":
            params = step.get("params", {}) or {}
            size = params.get("size")
            if size is not None:
                return int(size), int(size)
    return 256, 256


def _count_conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_c, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_c = module.in_channels
    groups = module.groups
    kernel_mul = (in_c // groups) * kernel_h * kernel_w
    return int(batch * out_c * out_h * out_w * kernel_mul)


def _count_conv_transpose2d_macs(module: nn.ConvTranspose2d, output: torch.Tensor) -> int:
    batch, out_c, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_c = module.in_channels
    groups = module.groups
    kernel_mul = (in_c // groups) * kernel_h * kernel_w
    return int(batch * out_c * out_h * out_w * kernel_mul)


def _count_linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    out_features = module.out_features
    in_features = module.in_features
    batch = output.shape[0] if output.dim() > 0 else 1
    return int(batch * in_features * out_features)


def _make_hook(name: str, stats: List[LayerStat]):
    def _hook(module: nn.Module, inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            return
        macs = 0
        if isinstance(module, nn.Conv2d):
            macs = _count_conv2d_macs(module, output)
        elif isinstance(module, nn.ConvTranspose2d):
            macs = _count_conv_transpose2d_macs(module, output)
        elif isinstance(module, nn.Linear):
            macs = _count_linear_macs(module, output)
        stats.append(
            LayerStat(
                name=name,
                layer_type=module.__class__.__name__,
                output_shape=tuple(output.shape),
                macs=macs,
            )
        )

    return _hook


def _format_large_int(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} M"
    if value >= 1_000:
        return f"{value / 1_000:.3f} K"
    return str(value)


def main() -> None:
    args = _parse_args()
    opt = load_experiment_config(args.opt)

    network_opt = dict(opt["network"])
    network_type = network_opt.pop("type")
    network_cls = NETWORK_REGISTRY.get(network_type)
    model = network_cls(**network_opt)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if args.height is None or args.width is None:
        inferred_h, inferred_w = _infer_spatial_size(opt)
        height = args.height or inferred_h
        width = args.width or inferred_w
    else:
        height = args.height
        width = args.width

    batch_size = int(args.batch_size)
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise SystemExit("batch_size, height, and width must be positive.")

    dummy = torch.randn(batch_size, int(network_opt.get("in_channels", 3)), height, width, device=device)

    stats: List[LayerStat] = []
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_make_hook(name or module.__class__.__name__, stats)))

    with torch.no_grad():
        output = model(dummy)

    for hook in hooks:
        hook.remove()

    total_macs = sum(item.macs for item in stats)
    total_params = sum(p.numel() for p in model.parameters())
    per_image_pixels = int(output.shape[-2] * output.shape[-1])
    kmac_per_pixel = (total_macs / batch_size) / per_image_pixels / 1_000.0

    print(f"Config: {args.opt}")
    print(f"Network: {network_type}")
    print(f"Input shape: {(batch_size, dummy.shape[1], height, width)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Params: {_format_large_int(total_params)} ({total_params:,})")
    print(f"MACs / forward: {_format_large_int(total_macs)} ({total_macs:,})")
    print(f"kMAC/pixel: {kmac_per_pixel:.3f}")

    if args.details:
        print("\nPer-layer MACs:")
        for item in stats:
            print(
                f"- {item.name:<40} {item.layer_type:<18} "
                f"shape={item.output_shape!s:<20} macs={_format_large_int(item.macs)}"
            )


if __name__ == "__main__":
    main()
