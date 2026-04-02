from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Dict, Sequence

import lmdb
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the exact YUV444 tensors seen by the dataset loader for debugging."
    )
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=("lmdb", "image"),
        help="Use 'lmdb' for PairedLmdbDataset or 'image' for PairedImageDataset.",
    )
    parser.add_argument("--dataroot_lq", required=True, help="LQ root path.")
    parser.add_argument("--dataroot_gt", required=True, help="GT root path.")
    parser.add_argument("--meta_info", default=None, help="Shared meta_info path.")
    parser.add_argument("--meta_info_lq", default=None, help="LQ meta_info path.")
    parser.add_argument("--meta_info_gt", default=None, help="GT meta_info path.")
    parser.add_argument("--index", type=int, default=0, help="Start index to dump.")
    parser.add_argument("--count", type=int, default=1, help="How many samples to dump.")
    parser.add_argument("--out_dir", required=True, help="Output folder for dumped .yuv files.")
    return parser.parse_args()


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def _read_meta_keys(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"meta_info not found: {path}")
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        keys.append(line.split(" ", 1)[0])
    if not keys:
        raise SystemExit(f"meta_info is empty: {path}")
    return keys


def _resolve_meta_path(root: Path, meta_info: str | None) -> Path | None:
    if not meta_info:
        return None
    meta_path = Path(meta_info)
    if meta_path.is_absolute():
        return meta_path
    if meta_path.exists():
        return meta_path
    return root / meta_path


def _list_image_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}))


def _to_tensor_rgb(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return _to_tensor_rgb(image)


def _load_lmdb_tensor(env: lmdb.Environment, key: str) -> torch.Tensor:
    with env.begin(write=False) as txn:
        value = txn.get(key.encode("ascii"))
    if value is None:
        raise SystemExit(f"Missing LMDB key: {key}")
    image = Image.open(io.BytesIO(value)).convert("RGB")
    return _to_tensor_rgb(image)


def _build_paired_paths(args: argparse.Namespace) -> list[tuple[str, str]]:
    lq_root = Path(args.dataroot_lq)
    gt_root = Path(args.dataroot_gt)
    meta_info_lq = args.meta_info_lq or args.meta_info
    meta_info_gt = args.meta_info_gt or args.meta_info
    meta_lq_path = _resolve_meta_path(lq_root, meta_info_lq)
    meta_gt_path = _resolve_meta_path(gt_root, meta_info_gt)

    if meta_lq_path is not None and meta_gt_path is not None:
        lq_keys = _read_meta_keys(meta_lq_path)
        gt_keys = _read_meta_keys(meta_gt_path)
        if len(lq_keys) != len(gt_keys):
            raise SystemExit("LQ/GT meta_info length mismatch.")
        return [(str(lq_root / lq_key), str(gt_root / gt_key)) for lq_key, gt_key in zip(lq_keys, gt_keys)]

    if meta_lq_path is not None:
        lq_keys = _read_meta_keys(meta_lq_path)
    else:
        lq_keys = [p.name for p in _list_image_files(lq_root)]
    if meta_gt_path is not None:
        gt_keys = _read_meta_keys(meta_gt_path)
    else:
        gt_keys = [p.name for p in _list_image_files(gt_root)]

    if meta_lq_path is not None and meta_gt_path is None:
        gt_set = set(gt_keys)
        pairs: list[tuple[str, str]] = []
        for key in lq_keys:
            if key not in gt_set:
                raise SystemExit(f"Missing GT file for {key}")
            pairs.append((str(lq_root / key), str(gt_root / key)))
        return pairs

    if meta_gt_path is not None and meta_lq_path is None:
        lq_set = set(lq_keys)
        pairs = []
        for key in gt_keys:
            if key not in lq_set:
                raise SystemExit(f"Missing LQ file for {key}")
            pairs.append((str(lq_root / key), str(gt_root / key)))
        return pairs

    if len(lq_keys) != len(gt_keys):
        raise SystemExit("LQ/GT image count mismatch.")
    gt_map = {key: key for key in gt_keys}
    pairs = []
    for key in lq_keys:
        if key not in gt_map:
            raise SystemExit(f"Missing GT file for {key}")
        pairs.append((str(lq_root / key), str(gt_root / gt_map[key])))
    return pairs


def _save_raw_yuv444(tensor: torch.Tensor, out_path: Path) -> None:
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError("Expected CHW tensor with 3 channels.")
    chw = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
    y = chw[0].contiguous().numpy().reshape(-1)
    u = chw[1].contiguous().numpy().reshape(-1)
    v = chw[2].contiguous().numpy().reshape(-1)
    out_path.write_bytes(b"".join([y.tobytes(), u.tobytes(), v.tobytes()]))


def _dump_sample(lq: torch.Tensor, gt: torch.Tensor, lq_path: str, gt_path: str, sample_index: int, out_dir: Path) -> None:

    _, h, w = lq.shape
    lq_tag = _sanitize_name(Path(lq_path).stem)
    gt_tag = _sanitize_name(Path(gt_path).stem)

    lq_out = out_dir / f"{sample_index:04d}_{lq_tag}_lq_{w}x{h}_444p.yuv"
    gt_out = out_dir / f"{sample_index:04d}_{gt_tag}_gt_{w}x{h}_444p.yuv"
    _save_raw_yuv444(lq, lq_out)
    _save_raw_yuv444(gt, gt_out)

    info_out = out_dir / f"{sample_index:04d}_pair_info.txt"
    info_out.write_text(
        "\n".join(
            [
                f"index={sample_index}",
                f"lq_path={lq_path}",
                f"gt_path={gt_path}",
                f"shape={tuple(lq.shape)}",
                f"lq_dump={lq_out.name}",
                f"gt_dump={gt_out.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _build_paired_paths(args)
    total = len(pairs)
    if args.index < 0 or args.index >= total:
        raise SystemExit(f"--index out of range: {args.index}, dataset size: {total}")
    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    end = min(args.index + args.count, total)
    print(f"Dumping samples [{args.index}, {end}) from dataset size {total} ...")
    if args.dataset_type == "lmdb":
        lq_env = lmdb.open(str(Path(args.dataroot_lq)), readonly=True, lock=False, readahead=False, meminit=False)
        gt_env = lmdb.open(str(Path(args.dataroot_gt)), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            for sample_index in range(args.index, end):
                lq_path, gt_path = pairs[sample_index]
                lq_key = Path(lq_path).name
                gt_key = Path(gt_path).name
                lq = _load_lmdb_tensor(lq_env, lq_key)
                gt = _load_lmdb_tensor(gt_env, gt_key)
                _dump_sample(lq, gt, lq_key, gt_key, sample_index, out_dir)
                print(f"Dumped sample {sample_index}")
        finally:
            lq_env.close()
            gt_env.close()
    else:
        for sample_index in range(args.index, end):
            lq_path, gt_path = pairs[sample_index]
            lq = _load_image_tensor(Path(lq_path))
            gt = _load_image_tensor(Path(gt_path))
            _dump_sample(lq, gt, lq_path, gt_path, sample_index, out_dir)
            print(f"Dumped sample {sample_index}")
    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
