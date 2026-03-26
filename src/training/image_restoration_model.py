from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import src.models  # noqa: F401
from src.utils.color import luma_bt601
from .registry import MODEL_REGISTRY, NETWORK_REGISTRY


@MODEL_REGISTRY.register()
class ImageRestorationModel:
    def __init__(self, opt: Dict, device: torch.device) -> None:
        self.opt = opt
        self.device = device

        network_opt = dict(opt["network"])
        network_type = network_opt.pop("type")
        network_cls = NETWORK_REGISTRY.get(network_type)
        self.network = network_cls(**network_opt).to(device)

        train_opt = opt["train"]
        optimizer_opt = train_opt["optimizer"]
        self.optimizer = Adam(
            self.network.parameters(),
            lr=float(optimizer_opt["lr"]),
            betas=tuple(optimizer_opt.get("betas", [0.9, 0.99])),
            weight_decay=float(optimizer_opt.get("weight_decay", 0.0)),
        )
        for group in self.optimizer.param_groups:
            group["initial_lr"] = float(optimizer_opt["lr"])

        self.scheduler = None
        sched_opt = train_opt.get("scheduler", {})
        if sched_opt.get("type") == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=int(sched_opt.get("T_max", 50)))

        loss_opt = train_opt.get("loss", {})
        loss_type = loss_opt.get("type", "L1")
        if loss_type == "L1":
            self.pixel_loss = nn.L1Loss()
        elif loss_type == "MSE":
            self.pixel_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss.type: {loss_type}")

        self.loss_mode = loss_opt.get("mode", "all_channels")
        self.channel_weights = loss_opt.get("channel_weights")
        self.grad_clip = float(train_opt.get("grad_clip", 0.0))

        self.lq: torch.Tensor | None = None
        self.gt: torch.Tensor | None = None
        self.prediction: torch.Tensor | None = None

    def feed_data(self, data: Dict[str, torch.Tensor]) -> None:
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)

    def _compute_pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_mode in ("all_channels", "direct"):
            if self.channel_weights:
                weights = torch.tensor(self.channel_weights, device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
                diff = torch.abs(pred - target) * weights
                return diff.mean()
            return self.pixel_loss(pred, target)

        if self.loss_mode in ("bt601_luma", "rgb_luma_bt601", "y_channel_bt601"):
            return self.pixel_loss(luma_bt601(pred), luma_bt601(target))

        raise ValueError(f"Unsupported loss.mode: {self.loss_mode}")

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._compute_pixel_loss(pred, target)

    def optimize_parameters(self) -> Dict[str, float]:
        self.network.train()
        assert self.lq is not None and self.gt is not None
        self.optimizer.zero_grad()
        self.prediction = self.network(self.lq)
        loss = self._compute_pixel_loss(self.prediction, self.gt)
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"pixel": loss.item()}

    @torch.no_grad()
    def test(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.network.eval()
        assert self.lq is not None and self.gt is not None
        self.prediction = self.network(self.lq)
        return self.lq, self.prediction, self.gt

    def step_scheduler(self) -> None:
        if self.scheduler:
            self.scheduler.step()

    def state_dict(self) -> Dict:
        return {"model": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict, strict: bool = True) -> None:
        model_state = state.get("model")
        if model_state is None:
            raise KeyError("Checkpoint is missing model weights.")
        self.network.load_state_dict(model_state, strict=strict)

        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
