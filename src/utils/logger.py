from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _env_info() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
    }


class ExperimentLogger:
    def __init__(self, run_dir: Path, use_tensorboard: bool) -> None:
        self.run_dir = run_dir
        self.use_tensorboard = use_tensorboard
        self.tb: Optional[SummaryWriter] = SummaryWriter(run_dir) if use_tensorboard else None
        self.metrics_path = run_dir / "metrics.jsonl"

        self._logger = logging.getLogger("train")
        self._logger.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        self._logger.addHandler(stream_handler)

    def log_config(self, cfg: Dict[str, Any], argv: str) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (self.run_dir / "env.json").write_text(json.dumps(_env_info(), indent=2), encoding="utf-8")
        (self.run_dir / "cmd.txt").write_text(argv, encoding="utf-8")

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        record = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + os.linesep)
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self.tb:
            self.tb.close()
