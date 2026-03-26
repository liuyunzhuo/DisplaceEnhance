import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import ConcatDataset, Dataset

import src.models  # noqa: F401
from src.training import DATASET_REGISTRY, MODEL_REGISTRY
from src.training.config_loader import load_experiment_config
from src.training.datasets import PairedImageDataset, PairedLmdbDataset  # noqa: F401
from src.training.image_restoration_model import ImageRestorationModel  # noqa: F401
from src.training.training_engine import TrainingEngine
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


def _make_run_dir(opt: Dict[str, Any]) -> Path:
    name = opt.get("name", "experiment_run")
    root = Path(opt.get("path", {}).get("experiments_root", "experiments"))
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeat: int) -> None:
        if repeat <= 0:
            raise ValueError("repeat must be greater than 0")
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]


def _build_dataset(dataset_opt: Dict[str, Any]) -> Dataset:
    dataset_cls = DATASET_REGISTRY.get(dataset_opt["type"])
    dataset = dataset_cls(dataset_opt)
    repeat = int(dataset_opt.get("repeat", 1))
    if repeat > 1:
        dataset = RepeatDataset(dataset, repeat)
    return dataset


def _build_train_dataset(dataset_opt: Dict[str, Any]) -> Dataset:
    subdatasets_opt = dataset_opt.get("datasets")
    if not subdatasets_opt:
        return _build_dataset(dataset_opt)
    subdatasets = [_build_dataset(sub_opt) for sub_opt in subdatasets_opt]
    if len(subdatasets) == 1:
        return subdatasets[0]
    return ConcatDataset(subdatasets)


def _create_dataloader(dataset_opt: Dict[str, Any], is_train: bool) -> torch.utils.data.DataLoader:
    if is_train:
        dataset = _build_train_dataset(dataset_opt)
    else:
        dataset = _build_dataset(dataset_opt)

    batch_size = int(dataset_opt.get("batch_size_per_gpu", 8))
    num_workers = int(dataset_opt.get("num_worker_per_gpu", 4))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="configs/sharpen_paired.yaml")
    args = parser.parse_args()

    opt = load_experiment_config(args.opt)
    set_seed(int(opt.get("seed", 42)))

    run_dir = _make_run_dir(opt)
    logger = ExperimentLogger(run_dir, bool(opt.get("logger", {}).get("use_tb_logger", True)))
    logger.log_config(opt, " ".join(sys.argv))
    logger.info(f"Run dir: {run_dir}")

    datasets = opt.get("datasets", {})
    train_loader = _create_dataloader(datasets["train"], is_train=True)
    val_loader = _create_dataloader(datasets["val"], is_train=False)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY.get(opt.get("model_type", "ImageRestorationModel"))
    model = model_cls(opt, device)

    resume_state = opt.get("path", {}).get("resume_state")
    if resume_state:
        state = torch.load(resume_state, map_location=device)
        strict_load = bool(opt.get("path", {}).get("strict_load_model", True))
        model.load_state_dict(state, strict=strict_load)

    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        logger=logger,
        opt=opt,
    )

    if resume_state:
        engine.load_state(state)

    engine.run()
    logger.close()


if __name__ == "__main__":
    main()
