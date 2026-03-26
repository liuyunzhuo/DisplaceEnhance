from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.utils.color import packed_yuv444_to_rgb
from src.utils.logger import ExperimentLogger
from src.utils.metrics import psnr


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
        self.visualization_pipeline = (self.val_opt.get("visualization", {}) or {}).get("pipeline", [])

        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "visualization").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "training_state").mkdir(parents=True, exist_ok=True)

    def _save_visuals(
        self,
        pred: torch.Tensor,
        step: int,
        paths: Sequence[str],
        start_index: int,
    ) -> None:
        out_dir = self.run_dir / "visualization" / f"iter_{step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        visual_pred = self._apply_visualization_pipeline(pred)
        for i in range(pred.size(0)):
            sample_index = start_index + i
            source_name = Path(paths[i]).stem if i < len(paths) else f"{sample_index:03d}"
            save_image(visual_pred[i], out_dir / f"{sample_index:03d}_{source_name}.png")

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
        count = 0
        saved_count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                self.model.feed_data(batch)
                lq, pred, gt = self.model.test()
                loss = self.model.compute_loss(pred, gt)
                total_l1 += loss.item()
                total_psnr += psnr(pred, gt).item()
                count += 1
                if self.save_images:
                    lq_paths = batch.get("lq_path", [])
                    if isinstance(lq_paths, str):
                        lq_paths = [lq_paths]
                    self._save_visuals(pred, step, lq_paths, saved_count)
                    saved_count += pred.size(0)
        if count == 0:
            return {"val/l1": 0.0, "val/psnr": 0.0}
        return {"val/l1": total_l1 / count, "val/psnr": total_psnr / count}

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

            if self.state.iter % print_freq == 0:
                self.logger.log_metrics(self.state.iter, {f"train/{k}": v for k, v in losses.items()})
                self.logger.info(f"[Iter {self.state.iter}/{total_iter}] {losses}")

            if self.state.iter % val_freq == 0:
                val_metrics = self._validate(self.state.iter)
                self.logger.log_metrics(self.state.iter, val_metrics)
                self.logger.info(f"[Val {self.state.iter}] {val_metrics}")
                if val_metrics["val/psnr"] > best_psnr:
                    best_psnr = val_metrics["val/psnr"]
                    self._save_checkpoint(self.state.iter, best=True)

            if self.state.iter % save_freq == 0:
                self._save_checkpoint(self.state.iter)
