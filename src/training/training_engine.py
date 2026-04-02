from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.utils.color import packed_yuv444_to_rgb
from src.utils.logger import ExperimentLogger
from src.utils.metrics import psnr, psnr_per_channel


@dataclass
class TrainState:
    iter: int = 0
    epoch: int = 0


def _cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


class TrainingEngine:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
        logger: ExperimentLogger,
        opt: Dict,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.logger = logger
        self.opt = opt
        self.state = TrainState()

        self.train_opt = opt["train"]
        self.val_opt = opt.get("val", {})
        self.save_images = bool(self.val_opt.get("save_img", True))
        self.save_format = str(self.val_opt.get("save_format", "png")).lower()
        self.tensor_range = str(self.val_opt.get("tensor_range", "zero_one")).lower()
        self.visualization_pipeline = (self.val_opt.get("visualization", {}) or {}).get("pipeline", [])

        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "visualization").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "training_state").mkdir(parents=True, exist_ok=True)

    def _save_visuals(
        self,
        pred: torch.Tensor,
        step: int,
        paths: Sequence[str],
    ) -> None:
        out_dir = self.run_dir / "visualization" / f"iter_{step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(pred.size(0)):
            source_name = Path(paths[i]).stem if i < len(paths) else f"sample_{i:03d}"
            base_name = self._build_visual_base_name(source_name)
            if self.save_format == "png":
                visual_pred = self._apply_visualization_pipeline(self._to_normalized_range(pred[i : i + 1]))
                save_image(visual_pred[0], out_dir / f"{base_name}_pred.png")
                continue
            if self.save_format in ("raw_yuv444", "yuv444", "yuv444p"):
                self._save_raw_yuv444(pred[i], out_dir, f"{base_name}_pred")
                continue
            raise ValueError(f"Unsupported val.save_format: {self.save_format}")

    def _build_visual_base_name(self, source_name: str) -> str:
        # Collapse packed/test naming variants like:
        # 0007_TE_1920x1080_420p_yuv444png -> 0007_TE_1920x1080
        # 0007_TE_1920x1080_444p          -> 0007_TE_1920x1080
        patterns = (
            r"_(?:I)?(?:420p|444p)_yuv444png$",
            r"_(?:I)?(?:420p|444p)$",
            r"_444p$",
        )
        base_name = source_name
        for pattern in patterns:
            new_name = re.sub(pattern, "", base_name, flags=re.IGNORECASE)
            if new_name != base_name:
                base_name = new_name
                break
        return base_name

    def _save_raw_yuv444(
        self,
        pred: torch.Tensor,
        out_dir: Path,
        source_name: str,
    ) -> None:
        if pred.dim() != 3 or pred.size(0) != 3:
            raise ValueError("raw_yuv444 saving expects a CHW tensor with 3 channels.")
        normalized = self._to_normalized_range(pred.unsqueeze(0))[0]
        chw = normalized.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        height = int(chw.size(1))
        width = int(chw.size(2))
        y = chw[0].contiguous().numpy().reshape(-1)
        u = chw[1].contiguous().numpy().reshape(-1)
        v = chw[2].contiguous().numpy().reshape(-1)
        payload = np.concatenate([y, u, v]).tobytes()
        out_path = out_dir / f"{source_name}_444p.yuv"
        out_path.write_bytes(payload)

    def _to_normalized_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_range in ("zero_one", "normalized_01", "default"):
            return x
        if self.tensor_range in ("byte_centered", "centered_255"):
            return torch.clamp((x + 128.0) / 255.0, 0.0, 1.0)
        raise ValueError(f"Unsupported val.tensor_range: {self.tensor_range}")

    def _apply_visualization_pipeline(self, pred: torch.Tensor) -> torch.Tensor:
        out = pred
        for step in self.visualization_pipeline:
            if isinstance(step, str):
                name = step
                params: Dict = {}
            else:
                name = step.get("name", "")
                params = step.get("params", {}) or {}

            if name in ("identity", ""):
                continue
            if name == "packed_yuv444_to_rgb":
                matrix = params.get("matrix", "bt601")
                value_range = params.get("value_range", params.get("range", "full"))
                out = packed_yuv444_to_rgb(out, matrix=matrix, value_range=value_range)
                continue
            raise ValueError(f"Unsupported val.visualization step: {name}")
        return out

    def _save_checkpoint(self, step: int, best: bool = False) -> None:
        state = {"iter": step, **self.model.state_dict()}
        name = "best.pt" if best else f"model_{step:08d}.pt"
        torch.save(state, self.run_dir / "models" / name)

        training_state = {"iter": step}
        torch.save(training_state, self.run_dir / "training_state" / f"iter_{step:08d}.state")

    def load_state(self, state: Dict) -> None:
        self.state.iter = int(state.get("iter", 0))

    def _validate(self, step: int) -> Dict[str, float]:
        self.model.network.eval()
        total_l1 = 0.0
        total_psnr = 0.0
        total_y_psnr = 0.0
        total_u_psnr = 0.0
        total_v_psnr = 0.0
        count = 0
        saved_count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                self.model.feed_data(batch)
                lq, pred, gt = self.model.test()
                loss = self.model.compute_loss(pred, gt)
                metric_max_val = 255.0 if self.tensor_range in ("byte_centered", "centered_255") else 1.0
                total_l1 += loss.item()
                total_psnr += psnr(pred, gt, max_val=metric_max_val).item()
                if pred.size(1) >= 3:
                    channel_psnr = psnr_per_channel(pred, gt, max_val=metric_max_val)
                    total_y_psnr += channel_psnr[0].item()
                    total_u_psnr += channel_psnr[1].item()
                    total_v_psnr += channel_psnr[2].item()
                count += 1
                if self.save_images:
                    lq_paths = batch.get("lq_path", [])
                    if isinstance(lq_paths, str):
                        lq_paths = [lq_paths]
                    self._save_visuals(pred, step, lq_paths)
                    saved_count += pred.size(0)
        if count == 0:
            return {
                "val/l1": 0.0,
                "val/psnr": 0.0,
                "val/y_psnr": 0.0,
                "val/u_psnr": 0.0,
                "val/v_psnr": 0.0,
            }
        return {
            "val/l1": total_l1 / count,
            "val/psnr": total_psnr / count,
            "val/y_psnr": total_y_psnr / count,
            "val/u_psnr": total_u_psnr / count,
            "val/v_psnr": total_v_psnr / count,
        }

    def run(self) -> None:
        total_iter = int(self.train_opt["total_iter"])
        print_freq = int(self.opt["logger"].get("print_freq", 100))
        val_freq = int(self.train_opt.get("val_freq", 1000))
        save_freq = int(self.train_opt.get("save_checkpoint_freq", 5000))

        train_iter = _cycle(self.train_loader)
        best_psnr = -1.0

        while self.state.iter < total_iter:
            self.state.iter += 1
            batch = next(train_iter)
            self.model.feed_data(batch)
            losses = self.model.optimize_parameters()
            self.model.step_scheduler()
            losses_with_lr = dict(losses)
            if hasattr(self.model, "get_current_learning_rate"):
                losses_with_lr["lr"] = self.model.get_current_learning_rate()

            if self.state.iter % print_freq == 0:
                self.logger.log_metrics(self.state.iter, {f"train/{k}": v for k, v in losses_with_lr.items()})
                self.logger.info(f"[Iter {self.state.iter}/{total_iter}] {losses_with_lr}")

            if self.state.iter % val_freq == 0:
                val_metrics = self._validate(self.state.iter)
                self.logger.log_metrics(self.state.iter, val_metrics)
                self.logger.info(f"[Val {self.state.iter}] {val_metrics}")
                if val_metrics["val/psnr"] > best_psnr:
                    best_psnr = val_metrics["val/psnr"]
                    self._save_checkpoint(self.state.iter, best=True)

            if self.state.iter % save_freq == 0:
                self._save_checkpoint(self.state.iter)
