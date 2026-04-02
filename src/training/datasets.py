from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional, List

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import lmdb
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .registry import DATASET_REGISTRY


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_image_files(root: Path) -> Tuple[Path, ...]:
    return tuple(sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))


def _to_tensor_with_range(img: Image.Image, range_mode: str) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    key = range_mode.lower()
    if key in ("zero_one", "normalized_01", "default"):
        return tensor / 255.0
    if key in ("byte_centered", "centered_255"):
        return tensor - 128.0
    raise ValueError(f"Unsupported to_tensor range_mode: {range_mode}")


def _resolve_meta_path(root_path: Path, meta_info: str) -> Path:
    meta_path = Path(meta_info)
    if meta_path.is_absolute():
        return meta_path
    if meta_path.exists():
        return meta_path
    return root_path / meta_path


def _paired_paths(dataroot_lq: str, dataroot_gt: str) -> Tuple[Tuple[Path, Path], ...]:
    lq_paths = _list_image_files(Path(dataroot_lq))
    gt_paths = _list_image_files(Path(dataroot_gt))
    if len(lq_paths) != len(gt_paths):
        raise ValueError("lq and gt count mismatch")
    gt_map = {p.name: p for p in gt_paths}
    pairs = []
    for lq in lq_paths:
        if lq.name not in gt_map:
            raise ValueError(f"missing gt for {lq.name}")
        pairs.append((lq, gt_map[lq.name]))
    return tuple(pairs)


def _paired_paths_with_meta(
    dataroot_lq: str,
    dataroot_gt: str,
    meta_info_lq: Optional[str],
    meta_info_gt: Optional[str],
) -> Tuple[Tuple[Path, Path], ...]:
    lq_root = Path(dataroot_lq)
    gt_root = Path(dataroot_gt)

    if meta_info_lq:
        lq_paths = _load_images(dataroot_lq, meta_info_lq)
    else:
        lq_paths = _list_image_files(lq_root)

    if meta_info_gt:
        gt_paths = _load_images(dataroot_gt, meta_info_gt)
    else:
        gt_paths = _list_image_files(gt_root)

    if meta_info_lq and meta_info_gt:
        if len(lq_paths) != len(gt_paths):
            raise ValueError("lq and gt meta_info length mismatch")
        return tuple(zip(lq_paths, gt_paths))

    gt_map = {p.name: p for p in gt_paths}
    pairs = []
    for lq in lq_paths:
        if lq.name not in gt_map:
            raise ValueError(f"missing gt for {lq.name}")
        pairs.append((lq, gt_map[lq.name]))
    return tuple(pairs)


def _augment(lq: torch.Tensor, gt: torch.Tensor, use_flip: bool, use_rot: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_flip and random.random() < 0.5:
        lq = torch.flip(lq, dims=[2])
        gt = torch.flip(gt, dims=[2])
    if use_flip and random.random() < 0.5:
        lq = torch.flip(lq, dims=[1])
        gt = torch.flip(gt, dims=[1])
    if use_rot:
        k = random.randint(0, 3)
        if k:
            lq = torch.rot90(lq, k, dims=[1, 2])
            gt = torch.rot90(gt, k, dims=[1, 2])
    return lq, gt


def _load_images(root: str, meta_info: Optional[str] = None) -> Tuple[Path, ...]:
    root_path = Path(root)
    if not meta_info:
        return _list_image_files(root_path)
    meta_path = _resolve_meta_path(root_path, meta_info)
    keys = _read_meta_keys(meta_path)
    paths: List[Path] = []
    for key in keys:
        p = root_path / key
        if not p.exists():
            raise ValueError(f"meta_info entry not found: {p}")
        paths.append(p)
    return tuple(paths)


def _read_meta_keys(meta_path: Path) -> List[str]:
    if not meta_path.exists():
        raise ValueError(f"meta_info.txt not found: {meta_path}")
    keys = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key = line.split(" ", 1)[0]
        keys.append(key)
    if not keys:
        raise ValueError(f"meta_info.txt is empty: {meta_path}")
    return keys


def _bytes_to_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _degrade_from_gt(img: Image.Image, opt: Dict) -> Image.Image:
    # Lightweight configurable degradation pipeline
    steps = opt.get("steps")
    orig_size = img.size
    if not steps:
        # fallback to legacy packed params
        w, h = img.size
        scale = float(opt.get("scale", 2))
        min_scale = float(opt.get("min_scale", 1.0))
        max_scale = float(opt.get("max_scale", scale))
        s = random.uniform(min_scale, max_scale)
        down_w = max(1, int(w / s))
        down_h = max(1, int(h / s))
        img = img.resize((down_w, down_h), resample=Image.BICUBIC)
        if bool(opt.get("use_blur", True)):
            sigma = random.uniform(opt.get("blur_sigma_min", 0.2), opt.get("blur_sigma_max", 1.0))
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = img.resize((w, h), resample=Image.BICUBIC)

        if bool(opt.get("use_noise", True)):
            noise_std = float(opt.get("noise_std", 2.0))
            if noise_std > 0:
                arr = np.array(img).astype("float32")
                noise = np.random.randn(*arr.shape).astype("float32") * noise_std
                arr = np.clip(arr + noise, 0, 255).astype("uint8")
                img = Image.fromarray(arr, mode="RGB")

        if bool(opt.get("use_jpeg", True)):
            q = int(random.uniform(opt.get("jpeg_min", 50), opt.get("jpeg_max", 95)))
            buf = Path(opt.get("jpeg_tmp_path", ".sharpen_tmp.jpg"))
            img.save(buf, format="JPEG", quality=q)
            img = Image.open(buf).convert("RGB")
            try:
                buf.unlink()
            except Exception:
                pass
        return img

    for step in steps:
        name = step.get("name")
        params = step.get("params", {}) or {}
        if name == "blur":
            sigma = random.uniform(params.get("sigma_min", 0.2), params.get("sigma_max", 1.0))
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif name == "downsample":
            w, h = img.size
            scale = float(params.get("scale", 2.0))
            min_scale = float(params.get("scale_min", 1.0))
            max_scale = float(params.get("scale_max", scale))
            s = random.uniform(min_scale, max_scale)
            down_w = max(1, int(w / s))
            down_h = max(1, int(h / s))
            img = img.resize((down_w, down_h), resample=Image.BICUBIC)
        elif name == "upsample":
            if params.get("to_orig", False):
                target_w, target_h = orig_size
            elif "scale" in params:
                w, h = img.size
                s = float(params.get("scale", 1.0))
                target_w = max(1, int(w * s))
                target_h = max(1, int(h * s))
            else:
                w, h = img.size
                target_w = int(params.get("width", w))
                target_h = int(params.get("height", h))
            img = img.resize((target_w, target_h), resample=Image.BICUBIC)
        elif name == "noise":
            noise_std = float(params.get("std", 2.0))
            if noise_std > 0:
                arr = np.array(img).astype("float32")
                noise = np.random.randn(*arr.shape).astype("float32") * noise_std
                arr = np.clip(arr + noise, 0, 255).astype("uint8")
                img = Image.fromarray(arr, mode="RGB")
        elif name == "jpeg":
            q = int(random.uniform(params.get("q_min", 50), params.get("q_max", 95)))
            buf = Path(params.get("tmp_path", ".sharpen_tmp.jpg"))
            img.save(buf, format="JPEG", quality=q)
            img = Image.open(buf).convert("RGB")
            try:
                buf.unlink()
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported lq_op step: {name}")
    return img


def _sharpen_from_lq(img: Image.Image, opt: Dict) -> Image.Image:
    steps = opt.get("steps")
    if not steps:
        # Simple unsharp mask style sharpening
        radius = float(opt.get("radius", 2.0))
        percent = int(round(float(opt.get("percent", 150.0))))
        threshold = int(opt.get("threshold", 3))
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    for step in steps:
        name = step.get("name")
        params = step.get("params", {}) or {}
        if name in ("identity", "copy"):
            img = img.copy()
        elif name == "sharpen":
            radius = float(params.get("radius", 2.0))
            amount = float(params.get("amount", 1.5))
            threshold = int(params.get("threshold", 3))
            percent = int(round(amount * 100.0))
            img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        elif name == "denoise":
            size = int(params.get("size", 3))
            img = img.filter(ImageFilter.MedianFilter(size=size))
        elif name == "contrast":
            factor = float(params.get("factor", 1.0))
            img = ImageEnhance.Contrast(img).enhance(factor)
        else:
            raise ValueError(f"Unsupported gt_op step: {name}")
    return img


def _make_ops() -> Tuple[Callable[[Image.Image, Dict], Image.Image], Callable[[Image.Image, Dict], Image.Image]]:
    # Returns (lq_op, gt_op)
    return _degrade_from_gt, _sharpen_from_lq


@DATASET_REGISTRY.register()
class PairedImageDataset(Dataset):
    def __init__(self, opt: Dict) -> None:
        self.opt = opt
        self.mode = opt.get("mode", "paired")  # paired | gt_only | lq_only | paired_aug
        self.loss_weight = float(opt.get("loss_weight", 1.0))
        self.pipeline = opt.get("pipeline") or [
            {"name": "ops"},
            {"name": "resize"},
            {"name": "to_tensor"},
            {"name": "augment"},
        ]
        self.pairs: Tuple[Tuple[Path, Path], ...] = ()
        self.lq_paths: Tuple[Path, ...] = ()
        self.gt_paths: Tuple[Path, ...] = ()

        if self.mode in ("paired", "paired_aug"):
            meta_info_lq = opt.get("meta_info_lq")
            meta_info_gt = opt.get("meta_info_gt")
            meta_info = opt.get("meta_info")
            if meta_info and not meta_info_lq:
                meta_info_lq = meta_info
            if meta_info and not meta_info_gt:
                meta_info_gt = meta_info
            if meta_info_lq or meta_info_gt:
                self.pairs = _paired_paths_with_meta(
                    opt["dataroot_lq"], opt["dataroot_gt"], meta_info_lq, meta_info_gt
                )
            else:
                self.pairs = _paired_paths(opt["dataroot_lq"], opt["dataroot_gt"])
        elif self.mode == "gt_only":
            self.gt_paths = _load_images(opt["dataroot_gt"], opt.get("meta_info"))
        elif self.mode == "lq_only":
            self.lq_paths = _load_images(opt["dataroot_lq"], opt.get("meta_info"))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        default_size = opt.get("default_size", 256)
        self.resize = transforms.Resize((default_size, default_size))
        self.lq_op, self.gt_op = _make_ops()

    def __len__(self) -> int:
        if self.mode in ("paired", "paired_aug"):
            return len(self.pairs)
        if self.mode == "gt_only":
            return len(self.gt_paths)
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode in ("paired", "paired_aug"):
            lq_path, gt_path = self.pairs[idx]
            lq_img = Image.open(lq_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
        elif self.mode == "gt_only":
            gt_path = self.gt_paths[idx]
            gt_img = Image.open(gt_path).convert("RGB")
            lq_img = None
            lq_path = Path("generated_from_gt")
        else:
            lq_path = self.lq_paths[idx]
            lq_img = Image.open(lq_path).convert("RGB")
            gt_img = None
            gt_path = Path("generated_from_lq")

        lq, gt = self._run_pipeline(lq_img, gt_img)
        return {
            "lq": lq,
            "gt": gt,
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
            "loss_weight": self.loss_weight,
        }

    def _run_pipeline(
        self, lq_img: Optional[Image.Image], gt_img: Optional[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for step in self.pipeline:
            if isinstance(step, str):
                name = step
                params: Dict = {}
            else:
                name = step.get("name", "")
                params = step.get("params", {}) or {}

            if name == "ops":
                lq_op_opt = params.get("lq_op", {})
                gt_op_opt = params.get("gt_op", {})
                if lq_op_opt.get("enable", False):
                    if lq_img is None:
                        if gt_img is None:
                            raise ValueError("lq_from_gt requires gt image")
                        lq_img = self.lq_op(gt_img, lq_op_opt)
                    else:
                        lq_img = self.lq_op(lq_img, lq_op_opt)
                if gt_op_opt.get("enable", False):
                    if "amount" in gt_op_opt:
                        gt_op_opt = dict(gt_op_opt)
                        gt_op_opt["percent"] = float(gt_op_opt["amount"]) * 100.0
                    if gt_img is None:
                        if lq_img is None:
                            raise ValueError("gt_from_lq requires lq image")
                        gt_img = self.gt_op(lq_img, gt_op_opt)
                    else:
                        gt_img = self.gt_op(gt_img, gt_op_opt)
            elif name == "resize":
                if lq_img is None or gt_img is None:
                    raise ValueError("resize requires both lq and gt images")
                if not isinstance(lq_img, Image.Image) or not isinstance(gt_img, Image.Image):
                    raise ValueError("resize must run before to_tensor")
                size = params.get("size")
                if size:
                    resize = transforms.Resize((int(size), int(size)))
                    lq_img = resize(lq_img)
                    gt_img = resize(gt_img)
                else:
                    lq_img = self.resize(lq_img)
                    gt_img = self.resize(gt_img)
            elif name == "crop":
                if lq_img is None and gt_img is None:
                    raise ValueError("crop requires at least one image")
                if (lq_img is not None and not isinstance(lq_img, Image.Image)) or (
                    gt_img is not None and not isinstance(gt_img, Image.Image)
                ):
                    raise ValueError("crop must run before to_tensor")
                size = int(params.get("size", 256))
                is_random = bool(params.get("random", True))
                if is_random:
                    base_img = lq_img if lq_img is not None else gt_img
                    i, j, h, w = transforms.RandomCrop.get_params(base_img, (size, size))
                    if lq_img is not None:
                        lq_img = TF.crop(lq_img, i, j, h, w)
                    if gt_img is not None:
                        gt_img = TF.crop(gt_img, i, j, h, w)
                else:
                    crop = transforms.CenterCrop(size)
                    if lq_img is not None:
                        lq_img = crop(lq_img)
                    if gt_img is not None:
                        gt_img = crop(gt_img)
            elif name == "to_tensor":
                if lq_img is None or gt_img is None:
                    raise ValueError("to_tensor requires both lq and gt images")
                range_mode = str(params.get("range_mode", "zero_one"))
                if isinstance(lq_img, Image.Image):
                    lq_img = _to_tensor_with_range(lq_img, range_mode)
                if isinstance(gt_img, Image.Image):
                    gt_img = _to_tensor_with_range(gt_img, range_mode)
            elif name == "augment":
                if lq_img is None or gt_img is None:
                    raise ValueError("augment requires both lq and gt tensors")
                if isinstance(lq_img, Image.Image) or isinstance(gt_img, Image.Image):
                    raise ValueError("augment must run after to_tensor")
                use_flip = bool(params.get("flip", True))
                use_rot = bool(params.get("rot", True))
                lq_img, gt_img = _augment(lq_img, gt_img, use_flip, use_rot)
            else:
                raise ValueError(f"Unknown pipeline step: {name}")

        if lq_img is None or gt_img is None:
            raise ValueError("pipeline did not produce both lq and gt")
        if isinstance(lq_img, Image.Image) or isinstance(gt_img, Image.Image):
            raise ValueError("pipeline ended before to_tensor")
        return lq_img, gt_img


@DATASET_REGISTRY.register()
class PairedLmdbDataset(Dataset):
    def __init__(self, opt: Dict) -> None:
        self.opt = opt
        self.mode = opt.get("mode", "paired")  # paired | paired_aug | gt_only | lq_only
        self.loss_weight = float(opt.get("loss_weight", 1.0))
        if self.mode not in ("paired", "paired_aug", "gt_only", "lq_only"):
            raise ValueError("PairedLmdbDataset only supports mode: paired / paired_aug / gt_only / lq_only")

        self.pipeline = opt.get("pipeline") or [
            {"name": "ops"},
            {"name": "resize"},
            {"name": "to_tensor"},
            {"name": "augment"},
        ]

        self.lq_root = Path(opt["dataroot_lq"]) if "dataroot_lq" in opt else None
        self.gt_root = Path(opt["dataroot_gt"]) if "dataroot_gt" in opt else None
        meta_name = opt.get("meta_info", "meta_info.txt")

        self.lq_keys: List[str] = []
        self.gt_keys: List[str] = []
        self.keys: List[str] = []
        self.paired_keys: List[Tuple[str, str]] = []

        if self.mode in ("paired", "paired_aug"):
            if self.lq_root is None or self.gt_root is None:
                raise ValueError("paired/paired_aug require both dataroot_lq and dataroot_gt")
            meta_lq = opt.get("meta_info_lq")
            meta_gt = opt.get("meta_info_gt")
            meta_common = opt.get("meta_info")
            if meta_common and not meta_lq:
                meta_lq = meta_common
            if meta_common and not meta_gt:
                meta_gt = meta_common

            if meta_lq:
                lq_meta = _resolve_meta_path(self.lq_root, meta_lq)
                self.lq_keys = _read_meta_keys(lq_meta)
            else:
                lq_meta = self.lq_root / meta_name
                self.lq_keys = _read_meta_keys(lq_meta)

            if meta_gt:
                gt_meta = _resolve_meta_path(self.gt_root, meta_gt)
                self.gt_keys = _read_meta_keys(gt_meta)
            else:
                gt_meta = self.gt_root / meta_name
                self.gt_keys = _read_meta_keys(gt_meta)

            if meta_lq and meta_gt:
                if len(self.lq_keys) != len(self.gt_keys):
                    raise ValueError("lq and gt meta_info length mismatch")
                self.paired_keys = list(zip(self.lq_keys, self.gt_keys))
            else:
                gt_set = set(self.gt_keys)
                pairs = [k for k in self.lq_keys if k in gt_set]
                if len(pairs) != len(self.lq_keys):
                    missing = sorted(set(self.lq_keys) - gt_set)
                    raise ValueError(f"lq meta_info has missing gt keys: {missing[:5]}")
                self.keys = self.lq_keys
        elif self.mode == "gt_only":
            if self.gt_root is None:
                raise ValueError("gt_only requires dataroot_gt")
            gt_meta = self.gt_root / meta_name
            self.gt_keys = _read_meta_keys(gt_meta)
            self.keys = self.gt_keys
        else:
            if self.lq_root is None:
                raise ValueError("lq_only requires dataroot_lq")
            lq_meta = self.lq_root / meta_name
            self.lq_keys = _read_meta_keys(lq_meta)
            self.keys = self.lq_keys
        default_size = opt.get("default_size", 256)
        self.resize = transforms.Resize((default_size, default_size))
        self.lq_op, self.gt_op = _make_ops()

        self._env_lq: Optional[lmdb.Environment] = None
        self._env_gt: Optional[lmdb.Environment] = None

    def _init_lmdb(self) -> None:
        if self.lq_root is not None and self._env_lq is None:
            self._env_lq = lmdb.open(
                str(self.lq_root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        if self.gt_root is not None and self._env_gt is None:
            self._env_gt = lmdb.open(
                str(self.gt_root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self) -> int:
        if self.mode in ("paired", "paired_aug") and self.paired_keys:
            return len(self.paired_keys)
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_lmdb()
        if self.mode in ("paired", "paired_aug") and self.paired_keys:
            lq_key, gt_key = self.paired_keys[idx]
        else:
            key = self.keys[idx]
            lq_key = key
            gt_key = key
        lq_key_b = lq_key.encode("ascii")
        gt_key_b = gt_key.encode("ascii")

        lq_img: Optional[Image.Image] = None
        gt_img: Optional[Image.Image] = None
        lq_path = lq_key
        gt_path = gt_key

        if self.mode in ("paired", "paired_aug"):
            assert self._env_lq is not None
            assert self._env_gt is not None
            with self._env_lq.begin(write=False) as txn_lq:
                lq_buf = txn_lq.get(lq_key_b)
            with self._env_gt.begin(write=False) as txn_gt:
                gt_buf = txn_gt.get(gt_key_b)
            if lq_buf is None or gt_buf is None:
                raise ValueError(f"Missing lmdb key: lq={lq_key} gt={gt_key}")
            lq_img = _bytes_to_rgb(lq_buf)
            gt_img = _bytes_to_rgb(gt_buf)
        elif self.mode == "gt_only":
            assert self._env_gt is not None
            with self._env_gt.begin(write=False) as txn_gt:
                gt_buf = txn_gt.get(gt_key_b)
            if gt_buf is None:
                raise ValueError(f"Missing lmdb key: {gt_key}")
            gt_img = _bytes_to_rgb(gt_buf)
            lq_path = "generated_from_gt"
        else:
            assert self._env_lq is not None
            with self._env_lq.begin(write=False) as txn_lq:
                lq_buf = txn_lq.get(lq_key_b)
            if lq_buf is None:
                raise ValueError(f"Missing lmdb key: {lq_key}")
            lq_img = _bytes_to_rgb(lq_buf)
            gt_path = "generated_from_lq"

        lq, gt = self._run_pipeline(lq_img, gt_img)
        return {
            "lq": lq,
            "gt": gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "loss_weight": self.loss_weight,
        }

    def _run_pipeline(
        self, lq_img: Optional[Image.Image], gt_img: Optional[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for step in self.pipeline:
            if isinstance(step, str):
                name = step
                params: Dict = {}
            else:
                name = step.get("name", "")
                params = step.get("params", {}) or {}

            if name == "ops":
                lq_op_opt = params.get("lq_op", {})
                gt_op_opt = params.get("gt_op", {})
                if lq_op_opt.get("enable", False):
                    if lq_img is None:
                        if gt_img is None:
                            raise ValueError("lq_from_gt requires gt image")
                        lq_img = self.lq_op(gt_img, lq_op_opt)
                    else:
                        lq_img = self.lq_op(lq_img, lq_op_opt)
                if gt_op_opt.get("enable", False):
                    if "amount" in gt_op_opt:
                        gt_op_opt = dict(gt_op_opt)
                        gt_op_opt["percent"] = float(gt_op_opt["amount"]) * 100.0
                    if gt_img is None:
                        if lq_img is None:
                            raise ValueError("gt_from_lq requires lq image")
                        gt_img = self.gt_op(lq_img, gt_op_opt)
                    else:
                        gt_img = self.gt_op(gt_img, gt_op_opt)
            elif name == "resize":
                if lq_img is None or gt_img is None:
                    raise ValueError("resize requires both lq and gt images")
                if not isinstance(lq_img, Image.Image) or not isinstance(gt_img, Image.Image):
                    raise ValueError("resize must run before to_tensor")
                size = params.get("size")
                if size:
                    resize = transforms.Resize((int(size), int(size)))
                    lq_img = resize(lq_img)
                    gt_img = resize(gt_img)
                else:
                    lq_img = self.resize(lq_img)
                    gt_img = self.resize(gt_img)
            elif name == "crop":
                if lq_img is None and gt_img is None:
                    raise ValueError("crop requires at least one image")
                if (lq_img is not None and not isinstance(lq_img, Image.Image)) or (
                    gt_img is not None and not isinstance(gt_img, Image.Image)
                ):
                    raise ValueError("crop must run before to_tensor")
                size = int(params.get("size", 256))
                is_random = bool(params.get("random", True))
                if is_random:
                    base_img = lq_img if lq_img is not None else gt_img
                    i, j, h, w = transforms.RandomCrop.get_params(base_img, (size, size))
                    if lq_img is not None:
                        lq_img = TF.crop(lq_img, i, j, h, w)
                    if gt_img is not None:
                        gt_img = TF.crop(gt_img, i, j, h, w)
                else:
                    crop = transforms.CenterCrop(size)
                    if lq_img is not None:
                        lq_img = crop(lq_img)
                    if gt_img is not None:
                        gt_img = crop(gt_img)
            elif name == "to_tensor":
                if lq_img is None or gt_img is None:
                    raise ValueError("to_tensor requires both lq and gt images")
                range_mode = str(params.get("range_mode", "zero_one"))
                if isinstance(lq_img, Image.Image):
                    lq_img = _to_tensor_with_range(lq_img, range_mode)
                if isinstance(gt_img, Image.Image):
                    gt_img = _to_tensor_with_range(gt_img, range_mode)
            elif name == "augment":
                if lq_img is None or gt_img is None:
                    raise ValueError("augment requires both lq and gt tensors")
                if isinstance(lq_img, Image.Image) or isinstance(gt_img, Image.Image):
                    raise ValueError("augment must run after to_tensor")
                use_flip = bool(params.get("flip", True))
                use_rot = bool(params.get("rot", True))
                lq_img, gt_img = _augment(lq_img, gt_img, use_flip, use_rot)
            else:
                raise ValueError(f"Unknown pipeline step: {name}")

        if lq_img is None or gt_img is None:
            raise ValueError("pipeline did not produce both lq and gt")
        if isinstance(lq_img, Image.Image) or isinstance(gt_img, Image.Image):
            raise ValueError("pipeline ended before to_tensor")
        return lq_img, gt_img
