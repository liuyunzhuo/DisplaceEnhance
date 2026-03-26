from __future__ import annotations

import argparse
import io
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import yaml
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MATRIX_COEFFICIENTS = {
    "bt601": (0.2990, 0.1140),
    "bt709": (0.2126, 0.0722),
}
DEFAULT_FILENAME_INFO_PATTERN = r"(?P<width>\d+)x(?P<height>\d+)_(?P<pixel_format>(?:yuv|i)?(?:420p|444p|420|444))"
RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


@dataclass
class Frame:
    color_space: str
    data: Any
    width: int
    height: int


@dataclass
class OutputState:
    name: str
    backend: str
    output_path: Path
    meta_info_out: Optional[Path]
    env: Optional[lmdb.Environment]
    map_size: Optional[int]
    write_batch_size: int
    compact: bool
    batch: List[Tuple[bytes, bytes]]
    meta_lines: List[str]
    encode_opt: Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LMDB from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML prep config")
    return parser.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit("Config root must be a YAML mapping.")
    return data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_meta_list(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"meta_info not found: {path}")
    keys: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        keys.append(line.split(" ", 1)[0])
    if not keys:
        raise SystemExit(f"meta_info empty: {path}")
    return keys


def _resolve_meta_path(root: Path, meta_info: Optional[str]) -> Optional[Path]:
    if not meta_info:
        return None
    meta_path = Path(meta_info)
    if not meta_path.is_absolute():
        if meta_path.exists():
            return meta_path
        meta_path = root / meta_path
    return meta_path


def _list_source_files(source_opt: Dict[str, Any]) -> Tuple[Path, ...]:
    root = Path(source_opt["root"])
    if not root.exists():
        raise SystemExit(f"Source folder not found: {root}")

    meta_path = _resolve_meta_path(root, source_opt.get("meta_info"))
    if meta_path is not None and meta_path.exists():
        names = _read_meta_list(meta_path)
        paths: List[Path] = []
        for name in names:
            path = root / name
            if not path.exists():
                raise SystemExit(f"meta_info entry not found: {path}")
            paths.append(path)
        return tuple(paths)

    pattern = source_opt.get("glob")
    decoder_type = (source_opt.get("decoder") or {}).get("type", "image")
    if pattern:
        return tuple(sorted(p for p in root.glob(pattern) if p.is_file()))
    if decoder_type == "image":
        return tuple(sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
    return tuple(sorted(p for p in root.iterdir() if p.is_file()))


def _pair_paths(lq_paths: Sequence[Path], gt_paths: Sequence[Path], pair_by_order: bool) -> Tuple[Tuple[Path, Path], ...]:
    if pair_by_order:
        if len(lq_paths) != len(gt_paths):
            raise SystemExit("pair_by_order requires equal LQ/GT counts.")
        return tuple(zip(lq_paths, gt_paths))

    gt_map = {path.name: path for path in gt_paths}
    pairs: List[Tuple[Path, Path]] = []
    for lq_path in lq_paths:
        gt_path = gt_map.get(lq_path.name)
        if gt_path is None:
            raise SystemExit(f"Missing GT file for {lq_path.name}")
        pairs.append((lq_path, gt_path))
    if len(pairs) != len(gt_paths):
        missing_lq = sorted(set(path.name for path in gt_paths) - set(path.name for path in lq_paths))
        if missing_lq:
            raise SystemExit(f"Missing LQ files for GT entries: {missing_lq[:5]}")
    return tuple(pairs)


def _extract_pair_key(path: Path, pair_key_pattern: str) -> str:
    match = re.search(pair_key_pattern, path.name, flags=re.IGNORECASE)
    if match is None:
        raise SystemExit(
            f"Could not extract pair key from filename: {path.name} using pattern: {pair_key_pattern}"
        )
    if "key" in match.groupdict() and match.group("key") is not None:
        return match.group("key")
    if match.groups():
        return match.group(1)
    return match.group(0)


def _pair_paths_by_key(
    lq_paths: Sequence[Path],
    gt_paths: Sequence[Path],
    pair_key_pattern: str,
) -> Tuple[Tuple[Path, Path], ...]:
    gt_map: Dict[str, Path] = {}
    for gt_path in gt_paths:
        key = _extract_pair_key(gt_path, pair_key_pattern)
        if key in gt_map:
            raise SystemExit(f"Duplicate GT pair key: {key}")
        gt_map[key] = gt_path

    pairs: List[Tuple[Path, Path]] = []
    seen_keys: set[str] = set()
    for lq_path in lq_paths:
        key = _extract_pair_key(lq_path, pair_key_pattern)
        gt_path = gt_map.get(key)
        if gt_path is None:
            raise SystemExit(f"Missing GT file for pair key {key} from {lq_path.name}")
        pairs.append((lq_path, gt_path))
        seen_keys.add(key)

    missing_lq = sorted(set(gt_map.keys()) - seen_keys)
    if missing_lq:
        raise SystemExit(f"Missing LQ files for GT pair keys: {missing_lq[:5]}")
    return tuple(pairs)


def _load_image_frame(path: Path) -> Frame:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    return Frame(color_space="rgb", data=rgb, width=width, height=height)


def _normalize_pixel_format(pixel_format: str) -> str:
    value = pixel_format.lower()
    if value in ("420", "420p", "yuv420p", "i420", "i420p"):
        return "yuv420p"
    if value in ("444", "444p", "yuv444p", "i444", "i444p"):
        return "yuv444p"
    raise SystemExit(f"Unsupported raw_yuv pixel_format: {pixel_format}")


def _infer_raw_yuv_info(path: Path, decoder_opt: Dict[str, Any]) -> Dict[str, Any]:
    pattern = decoder_opt.get("filename_pattern", DEFAULT_FILENAME_INFO_PATTERN)
    match = re.search(pattern, path.name, flags=re.IGNORECASE)
    if match is None:
        raise SystemExit(f"Could not infer raw_yuv info from filename: {path.name}")

    info = match.groupdict()
    result: Dict[str, Any] = {}
    if "width" in info and info["width"] is not None:
        result["width"] = int(info["width"])
    if "height" in info and info["height"] is not None:
        result["height"] = int(info["height"])
    if "pixel_format" in info and info["pixel_format"] is not None:
        result["pixel_format"] = _normalize_pixel_format(info["pixel_format"])
    return result


def _resolve_raw_yuv_decoder(path: Path, decoder_opt: Dict[str, Any]) -> Tuple[str, int, int]:
    inferred: Dict[str, Any] = {}
    if decoder_opt.get("infer_from_filename", False):
        inferred = _infer_raw_yuv_info(path, decoder_opt)

    width = decoder_opt.get("width", inferred.get("width"))
    height = decoder_opt.get("height", inferred.get("height"))
    pixel_format = decoder_opt.get("pixel_format", inferred.get("pixel_format"))

    if width is None or height is None or pixel_format is None:
        raise SystemExit(
            "raw_yuv decoder requires width, height, and pixel_format, "
            "or infer_from_filename: true with a matching filename pattern."
        )

    return _normalize_pixel_format(str(pixel_format)), int(width), int(height)


def _load_raw_yuv_frame(path: Path, decoder_opt: Dict[str, Any]) -> Frame:
    pixel_format, width, height = _resolve_raw_yuv_decoder(path, decoder_opt)
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)

    if pixel_format == "yuv444p":
        plane = width * height
        expected = plane * 3
        if raw.size != expected:
            raise SystemExit(f"{path} size mismatch for yuv444p: expected {expected} bytes, got {raw.size}")
        y = raw[0:plane].reshape(height, width)
        u = raw[plane : plane * 2].reshape(height, width)
        v = raw[plane * 2 : plane * 3].reshape(height, width)
        data = np.stack([y, u, v], axis=2)
        return Frame(color_space="yuv444", data=data, width=width, height=height)

    if pixel_format == "yuv420p":
        if width % 2 != 0 or height % 2 != 0:
            raise SystemExit(f"yuv420p requires even width/height, got {width}x{height} for {path.name}")
        plane = width * height
        chroma_plane = (width // 2) * (height // 2)
        expected = plane + chroma_plane * 2
        if raw.size != expected:
            raise SystemExit(f"{path} size mismatch for yuv420p: expected {expected} bytes, got {raw.size}")
        y = raw[0:plane].reshape(height, width)
        u = raw[plane : plane + chroma_plane].reshape(height // 2, width // 2)
        v = raw[plane + chroma_plane : plane + chroma_plane * 2].reshape(height // 2, width // 2)
        return Frame(color_space="yuv420", data=(y, u, v), width=width, height=height)

    raise SystemExit(f"Unsupported raw_yuv pixel_format: {pixel_format}")


def _load_frame(path: Path, source_opt: Dict[str, Any]) -> Frame:
    decoder_opt = source_opt.get("decoder", {"type": "image"})
    decoder_type = decoder_opt.get("type", "image")
    if decoder_type == "image":
        return _load_image_frame(path)
    if decoder_type == "raw_yuv":
        return _load_raw_yuv_frame(path, decoder_opt)
    raise SystemExit(f"Unsupported decoder type: {decoder_type}")


def _matrix_coefficients(matrix: str) -> Tuple[float, float, float]:
    key = matrix.lower()
    if key not in MATRIX_COEFFICIENTS:
        raise SystemExit(f"Unsupported matrix: {matrix}")
    kr, kb = MATRIX_COEFFICIENTS[key]
    kg = 1.0 - kr - kb
    return kr, kg, kb


def _clip_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array), 0, 255).astype(np.uint8)


def _rgb_to_yuv444(frame: Frame, params: Dict[str, Any]) -> Frame:
    if frame.color_space != "rgb":
        raise SystemExit("rgb_to_yuv444 requires an RGB frame.")
    matrix = params.get("matrix", "bt601")
    value_range = params.get("range", "full").lower()
    kr, kg, kb = _matrix_coefficients(matrix)

    rgb = frame.data.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    y = kr * r + kg * g + kb * b
    cb = 0.5 * (b - y) / (1.0 - kb)
    cr = 0.5 * (r - y) / (1.0 - kr)

    if value_range == "limited":
        y_plane = 16.0 + 219.0 * y
        u_plane = 128.0 + 224.0 * cb
        v_plane = 128.0 + 224.0 * cr
    elif value_range == "full":
        y_plane = 255.0 * y
        u_plane = 128.0 + 255.0 * cb
        v_plane = 128.0 + 255.0 * cr
    else:
        raise SystemExit(f"Unsupported range: {value_range}")

    data = np.stack([_clip_uint8(y_plane), _clip_uint8(u_plane), _clip_uint8(v_plane)], axis=2)
    return Frame(color_space="yuv444", data=data, width=frame.width, height=frame.height)


def _yuv444_to_rgb(frame: Frame, params: Dict[str, Any]) -> Frame:
    if frame.color_space != "yuv444":
        raise SystemExit("yuv444_to_rgb requires a YUV444 frame.")
    matrix = params.get("matrix", "bt601")
    value_range = params.get("range", "full").lower()
    kr, kg, kb = _matrix_coefficients(matrix)

    yuv = frame.data.astype(np.float32)
    y_plane = yuv[:, :, 0]
    u_plane = yuv[:, :, 1]
    v_plane = yuv[:, :, 2]

    if value_range == "limited":
        y = np.clip((y_plane - 16.0) / 219.0, 0.0, 1.0)
        cb = (u_plane - 128.0) / 224.0
        cr = (v_plane - 128.0) / 224.0
    elif value_range == "full":
        y = np.clip(y_plane / 255.0, 0.0, 1.0)
        cb = (u_plane - 128.0) / 255.0
        cr = (v_plane - 128.0) / 255.0
    else:
        raise SystemExit(f"Unsupported range: {value_range}")

    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    rgb = np.stack([_clip_uint8(r * 255.0), _clip_uint8(g * 255.0), _clip_uint8(b * 255.0)], axis=2)
    return Frame(color_space="rgb", data=rgb, width=frame.width, height=frame.height)


def _yuv420_to_yuv444_nn(frame: Frame) -> Frame:
    if frame.color_space != "yuv420":
        raise SystemExit("yuv420_to_yuv444_nn requires a YUV420 frame.")
    y_plane, u_plane, v_plane = frame.data
    u_up = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
    v_up = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
    data = np.stack([y_plane, u_up, v_up], axis=2)
    return Frame(color_space="yuv444", data=data, width=frame.width, height=frame.height)


def _resize_rgb_like(data: np.ndarray, width: int, height: int, resample_name: str) -> np.ndarray:
    if resample_name not in RESAMPLE_MAP:
        raise SystemExit(f"Unsupported resize resample: {resample_name}")
    if data.ndim != 3 or data.shape[2] != 3:
        raise SystemExit("resize expects an HxWx3 array.")

    resized_channels: List[np.ndarray] = []
    for channel_idx in range(3):
        image = Image.fromarray(data[:, :, channel_idx], mode="L")
        resized = image.resize((width, height), resample=RESAMPLE_MAP[resample_name])
        resized_channels.append(np.array(resized, dtype=np.uint8))
    return np.stack(resized_channels, axis=2)


def _resize_frame(frame: Frame, params: Dict[str, Any]) -> Frame:
    width = int(params.get("width", params.get("size", frame.width)))
    height = int(params.get("height", params.get("size", frame.height)))
    resample_name = params.get("resample", "bicubic").lower()

    if frame.color_space in ("rgb", "yuv444"):
        data = _resize_rgb_like(frame.data, width, height, resample_name)
        return Frame(color_space=frame.color_space, data=data, width=width, height=height)

    raise SystemExit("resize currently supports rgb and yuv444 frames. Convert yuv420 to yuv444 first.")


def _crop_frame(frame: Frame, x: int, y: int, size: int) -> Frame:
    if frame.color_space in ("rgb", "yuv444"):
        cropped = frame.data[y : y + size, x : x + size, :]
        return Frame(color_space=frame.color_space, data=cropped, width=size, height=size)

    if frame.color_space == "yuv420":
        if x % 2 != 0 or y % 2 != 0 or size % 2 != 0:
            raise SystemExit("crop on yuv420 requires even x, y, and size.")
        y_plane, u_plane, v_plane = frame.data
        y_crop = y_plane[y : y + size, x : x + size]
        u_crop = u_plane[y // 2 : (y + size) // 2, x // 2 : (x + size) // 2]
        v_crop = v_plane[y // 2 : (y + size) // 2, x // 2 : (x + size) // 2]
        return Frame(color_space="yuv420", data=(y_crop, u_crop, v_crop), width=size, height=size)

    raise SystemExit(f"Unsupported frame color_space: {frame.color_space}")


def _apply_step(frame: Frame, step: Dict[str, Any]) -> Frame:
    name = step.get("name", "")
    params = step.get("params", {}) or {}

    if name == "rgb_to_yuv444":
        return _rgb_to_yuv444(frame, params)
    if name == "yuv444_to_rgb":
        return _yuv444_to_rgb(frame, params)
    if name == "yuv420_to_yuv444_nn":
        return _yuv420_to_yuv444_nn(frame)
    if name == "resize":
        return _resize_frame(frame, params)
    raise SystemExit(f"Unsupported preprocessing step: {name}")


def _apply_pipeline(frame: Frame, steps: Sequence[Dict[str, Any]]) -> Frame:
    current = frame
    for step in steps:
        current = _apply_step(current, step)
    return current


def _build_coords(width: int, height: int, patching_opt: Dict[str, Any], rng: random.Random) -> Iterable[Tuple[int, int, int, int]]:
    mode = patching_opt.get("mode", "none").lower()
    if mode == "none":
        yield (0, 0, width, height)
        return

    size = int(patching_opt["size"])
    if size > width or size > height:
        raise SystemExit(f"Patch size {size} exceeds frame size {width}x{height}.")

    if mode == "grid":
        stride = int(patching_opt.get("stride", size))
        if stride <= 0:
            raise SystemExit("patching.stride must be > 0")
        for y in range(0, height - size + 1, stride):
            for x in range(0, width - size + 1, stride):
                yield (x, y, size, size)
        return

    if mode in ("grid_full", "grid_cover", "grid_exhaustive"):
        stride = int(patching_opt.get("stride", size))
        if stride <= 0:
            raise SystemExit("patching.stride must be > 0")

        x_positions = list(range(0, max(width - size + 1, 1), stride))
        y_positions = list(range(0, max(height - size + 1, 1), stride))
        last_x = width - size
        last_y = height - size
        if not x_positions or x_positions[-1] != last_x:
            x_positions.append(last_x)
        if not y_positions or y_positions[-1] != last_y:
            y_positions.append(last_y)

        x_positions = sorted(set(x_positions))
        y_positions = sorted(set(y_positions))
        for y in y_positions:
            for x in x_positions:
                yield (x, y, size, size)
        return

    if mode == "random":
        num_patches = int(patching_opt.get("num_patches", 1))
        for _ in range(num_patches):
            x = rng.randint(0, width - size)
            y = rng.randint(0, height - size)
            yield (x, y, size, size)
        return

    raise SystemExit(f"Unsupported patching.mode: {mode}")


def _resolve_format(fmt: str, source_path: Path) -> str:
    if fmt != "auto":
        return fmt
    suffix = source_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".webp":
        return "webp"
    return "png"


def _encode_frame(frame: Frame, encode_opt: Dict[str, Any], source_path: Path) -> bytes:
    channel_packing = encode_opt.get("channel_packing", "rgb")
    if channel_packing == "rgb":
        if frame.color_space != "rgb":
            raise SystemExit("RGB packing requires an RGB frame. Add yuv444_to_rgb before packing.")
        packed = frame.data
    elif channel_packing == "yuv444":
        if frame.color_space != "yuv444":
            raise SystemExit("yuv444 packing requires a YUV444 frame.")
        packed = frame.data
    else:
        raise SystemExit(f"Unsupported channel_packing: {channel_packing}")

    fmt = _resolve_format(encode_opt.get("format", "png"), source_path)
    png_compress_level = int(encode_opt.get("png_compress_level", 3))
    jpeg_quality = int(encode_opt.get("jpeg_quality", 90))
    webp_quality = int(encode_opt.get("webp_quality", 90))
    webp_lossless = bool(encode_opt.get("webp_lossless", False))
    image = Image.fromarray(packed, mode="RGB")
    buffer = io.BytesIO()

    if fmt == "png":
        image.save(buffer, format="PNG", compress_level=png_compress_level, optimize=True)
    elif fmt == "jpeg":
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    elif fmt == "webp":
        image.save(buffer, format="WEBP", quality=webp_quality, lossless=webp_lossless, method=6)
    else:
        raise SystemExit(f"Unsupported output format: {fmt}")
    return buffer.getvalue()


def _parse_output_pipeline(branch_opt: Dict[str, Any], packing_opt: Dict[str, Any]) -> Dict[str, Any]:
    if not branch_opt.get("pipeline"):
        raise SystemExit("Each packing.outputs.<branch> must define an explicit pipeline.")

    resolved: Dict[str, Any] = {
        "backend": None,
        "lmdb_name": None,
        "dir_name": None,
        "map_size_gb": None,
        "write_batch_size": None,
        "compact": None,
        "encode_opt": {
            "format": None,
            "channel_packing": None,
            "png_compress_level": int(packing_opt.get("png_compress_level", 3)),
            "jpeg_quality": int(packing_opt.get("jpeg_quality", 90)),
            "webp_quality": int(packing_opt.get("webp_quality", 90)),
            "webp_lossless": bool(packing_opt.get("webp_lossless", False)),
            "filename_mode": "index",
            "filename_suffix": "",
            "append_patch_suffix": False,
        },
    }

    for step in branch_opt["pipeline"]:
        name = step.get("name", "")
        params = step.get("params", {}) or {}

        if name == "store_rgb_png":
            resolved["encode_opt"]["format"] = "png"
            resolved["encode_opt"]["channel_packing"] = "rgb"
            resolved["encode_opt"]["png_compress_level"] = int(params.get("png_compress_level", resolved["encode_opt"]["png_compress_level"]))
        elif name == "store_packed_yuv444_png":
            resolved["encode_opt"]["format"] = "png"
            resolved["encode_opt"]["channel_packing"] = "yuv444"
            resolved["encode_opt"]["png_compress_level"] = int(params.get("png_compress_level", resolved["encode_opt"]["png_compress_level"]))
        elif name == "write_lmdb":
            resolved["backend"] = "lmdb"
            resolved["lmdb_name"] = params.get("lmdb_name")
            resolved["map_size_gb"] = float(params.get("map_size_gb", 0.25))
            resolved["write_batch_size"] = int(params.get("write_batch_size", 512))
            resolved["compact"] = bool(params.get("compact", True))
        elif name == "write_files":
            resolved["backend"] = "files"
            resolved["dir_name"] = params.get("dir_name")
            resolved["encode_opt"]["filename_mode"] = params.get("filename_mode", "index")
            resolved["encode_opt"]["filename_suffix"] = params.get("filename_suffix", "")
            resolved["encode_opt"]["append_patch_suffix"] = bool(params.get("append_patch_suffix", False))
        else:
            raise SystemExit(f"Unsupported output pipeline step: {name}")

    if not resolved["encode_opt"]["format"] or not resolved["encode_opt"]["channel_packing"]:
        raise SystemExit("Output pipeline requires a storage step such as store_rgb_png or store_packed_yuv444_png.")
    if not resolved["backend"]:
        resolved["backend"] = "files"
    if resolved["backend"] == "lmdb" and not resolved["lmdb_name"]:
        raise SystemExit("write_lmdb requires lmdb_name.")
    if resolved["backend"] == "files" and not resolved["dir_name"]:
        resolved["dir_name"] = branch_opt.get("dir_name")
    return resolved


def _dir_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _compact_lmdb(path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.compact_tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    before_size = _dir_size(path)
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    try:
        env.copy(str(tmp_path), compact=True)
    finally:
        env.close()

    meta_path = path / "meta_info.txt"
    if meta_path.exists():
        shutil.copy2(meta_path, tmp_path / "meta_info.txt")
    shutil.rmtree(path)
    tmp_path.rename(path)
    after_size = _dir_size(path)
    print(f"Compacted {path.name}: {_format_size(before_size)} -> {_format_size(after_size)}")


def _flush_batch(state: OutputState, write_batch_size: int) -> None:
    if state.backend != "lmdb":
        return
    if not state.batch or len(state.batch) < state.write_batch_size:
        return
    assert state.env is not None
    assert state.map_size is not None
    state.map_size = _flush_items(state.env, state.batch, state.map_size, state.name)
    state.batch.clear()


def _flush_items(env: lmdb.Environment, items: List[Tuple[bytes, bytes]], cur_size: int, tag: str) -> int:
    if not items:
        return cur_size

    batch_bytes = sum(len(value) for _, value in items)
    while True:
        try:
            with env.begin(write=True) as txn:
                for key, value in items:
                    txn.put(key, value)
            return cur_size
        except lmdb.MapFullError:
            grow_by = max(cur_size // 4, 256 * 1024**2, batch_bytes * 4)
            cur_size += grow_by
            print(f"[{tag}] MapFullError, growing map_size to {_format_size(cur_size)}")
            env.set_mapsize(cur_size)


def _finalize_output(state: OutputState) -> None:
    meta_text = "".join(state.meta_lines)
    if state.backend == "lmdb":
        assert state.env is not None
        assert state.map_size is not None
        state.map_size = _flush_items(state.env, state.batch, state.map_size, state.name)
        state.env.sync()
        state.env.close()
        (state.output_path / "meta_info.txt").write_text(meta_text, encoding="utf-8")
        if state.compact:
            _compact_lmdb(state.output_path)
    else:
        (state.output_path / "meta_info.txt").write_text(meta_text, encoding="utf-8")

    if state.meta_info_out is not None:
        state.meta_info_out.parent.mkdir(parents=True, exist_ok=True)
        state.meta_info_out.write_text(meta_text, encoding="utf-8")


def _format_extension(fmt: str) -> str:
    if fmt == "jpeg":
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt == "webp":
        return ".webp"
    raise SystemExit(f"Unsupported output format: {fmt}")


def _build_output_filename(state: OutputState, source_name: str, key_str: str) -> str:
    filename_mode = state.encode_opt.get("filename_mode", "index")
    filename_suffix = state.encode_opt.get("filename_suffix", "")
    ext = _format_extension(state.encode_opt["format"])

    if filename_mode == "index":
        stem = key_str
    elif filename_mode == "source_stem":
        stem = Path(source_name).stem
    else:
        raise SystemExit(f"Unsupported write_files filename_mode: {filename_mode}")
    return f"{stem}{filename_suffix}{ext}"


def _append_patch_tag_to_filename(filename: str, patch_tag: Optional[str]) -> str:
    if not patch_tag:
        return filename
    path = Path(filename)
    return f"{path.stem}_{patch_tag}{path.suffix}"


def _open_output_states(config: Dict[str, Any]) -> Dict[str, OutputState]:
    packing_opt = config.get("packing", {})
    output_root = Path(packing_opt["output_root"])
    _ensure_dir(output_root)
    meta_info_out_map = packing_opt.get("meta_info_out", {}) or {}
    if not isinstance(meta_info_out_map, dict):
        raise SystemExit("packing.meta_info_out must be a mapping like {lq: path, gt: path}.")

    states: Dict[str, OutputState] = {}
    global_encode = {
        "png_compress_level": int(packing_opt.get("png_compress_level", 3)),
        "jpeg_quality": int(packing_opt.get("jpeg_quality", 90)),
        "webp_quality": int(packing_opt.get("webp_quality", 90)),
        "webp_lossless": bool(packing_opt.get("webp_lossless", False)),
    }

    outputs_opt = packing_opt.get("outputs", {})
    for name in ("lq", "gt"):
        branch_opt = outputs_opt.get(name)
        if not branch_opt:
            continue
        if not branch_opt.get("enabled", True):
            continue
        resolved_branch = _parse_output_pipeline(branch_opt, packing_opt)
        backend = resolved_branch["backend"]
        if backend == "lmdb":
            output_name = resolved_branch["lmdb_name"] or f"{name}.lmdb"
            output_path = output_root / output_name
            map_size = int(float(resolved_branch["map_size_gb"]) * 1024**3)
            env = lmdb.open(str(output_path), map_size=map_size, subdir=True, lock=False, readahead=False, meminit=False)
            output_map_size: Optional[int] = map_size
            write_batch_size = int(resolved_branch["write_batch_size"])
            compact = bool(resolved_branch["compact"])
        elif backend == "files":
            output_name = resolved_branch["dir_name"] or name
            output_path = output_root / output_name
            _ensure_dir(output_path)
            env = None
            output_map_size = None
            write_batch_size = 0
            compact = False
        else:
            raise SystemExit(f"Unsupported output backend: {backend}")
        encode_opt = dict(global_encode)
        encode_opt.update(resolved_branch["encode_opt"])
        meta_info_out = meta_info_out_map.get(name)
        meta_info_out_path = Path(meta_info_out) if meta_info_out else None
        states[name] = OutputState(
            name=name.upper(),
            backend=backend,
            output_path=output_path,
            meta_info_out=meta_info_out_path,
            env=env,
            map_size=output_map_size,
            write_batch_size=write_batch_size,
            compact=compact,
            batch=[],
            meta_lines=[],
            encode_opt=encode_opt,
        )
    if not states:
        raise SystemExit("No outputs enabled under packing.outputs.")
    return states


def _sample_source_name(paths: Dict[str, Path], branch_name: str) -> str:
    if branch_name in paths:
        return paths[branch_name].name
    if "source" in paths:
        return paths["source"].name
    return f"{branch_name}.dat"


def _ensure_same_size(branch_frames: Dict[str, Frame]) -> Tuple[int, int]:
    sizes = {(frame.width, frame.height) for frame in branch_frames.values()}
    if len(sizes) != 1:
        raise SystemExit(f"Branch sizes do not match after preprocessing: {sorted(sizes)}")
    width, height = next(iter(sizes))
    return width, height


def _process_sample(
    branch_frames: Dict[str, Frame],
    source_paths: Dict[str, Path],
    patching_opt: Dict[str, Any],
    output_states: Dict[str, OutputState],
    rng: random.Random,
    key_index: int,
) -> int:
    width, height = _ensure_same_size(branch_frames)
    coords = list(_build_coords(width, height, patching_opt, rng))
    multi_patch = len(coords) > 1
    for patch_idx, (x, y, patch_width, patch_height) in enumerate(coords, start=1):
        patch_tag = f"s{patch_idx:03d}" if multi_patch else None
        key = f"{key_index:08d}".encode("ascii")
        key_str = key.decode("ascii")
        for branch_name, frame in branch_frames.items():
            if patch_width == width and patch_height == height:
                patch_frame = frame
            else:
                patch_frame = _crop_frame(frame, x, y, patch_width)
            state = output_states.get(branch_name)
            if state is None:
                continue
            source_name = _sample_source_name(source_paths, branch_name)
            payload = _encode_frame(patch_frame, state.encode_opt, source_paths.get(branch_name, source_paths.get("source", Path(source_name))))
            meta_source = source_name if not patch_tag else f"{source_name} patch={patch_tag}"
            if state.backend == "lmdb":
                state.batch.append((key, payload))
                state.meta_lines.append(
                    f"{key_str} ({patch_frame.height},{patch_frame.width},3) {meta_source}\n"
                )
                _flush_batch(state, state.write_batch_size)
            else:
                filename = _build_output_filename(state, source_name, key_str)
                if state.encode_opt.get("append_patch_suffix", False):
                    filename = _append_patch_tag_to_filename(filename, patch_tag)
                (state.output_path / filename).write_bytes(payload)
                state.meta_lines.append(
                    f"{filename} ({patch_frame.height},{patch_frame.width},3) {meta_source}\n"
                )
        key_index += 1
    return key_index


def _prepare_paired_samples(config: Dict[str, Any]) -> Tuple[Tuple[Tuple[Path, Path], ...], Dict[str, Any], Dict[str, Any]]:
    dataset_opt = config["dataset"]
    lq_opt = dataset_opt["lq"]
    gt_opt = dataset_opt["gt"]
    lq_paths = _list_source_files(lq_opt)
    gt_paths = _list_source_files(gt_opt)
    if not lq_paths or not gt_paths:
        raise SystemExit("No input files found for paired dataset.")
    pair_by_order = bool(dataset_opt.get("pair_by_order", False))
    pair_key_pattern = dataset_opt.get("pair_key_pattern")
    if pair_key_pattern:
        pairs = _pair_paths_by_key(lq_paths, gt_paths, pair_key_pattern)
    else:
        pairs = _pair_paths(lq_paths, gt_paths, pair_by_order)
    return pairs, lq_opt, gt_opt


def _prepare_single_samples(config: Dict[str, Any]) -> Tuple[Tuple[Path, ...], Dict[str, Any]]:
    source_opt = config["dataset"]["source"]
    paths = _list_source_files(source_opt)
    if not paths:
        raise SystemExit("No input files found for single-source dataset.")
    return paths, source_opt


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    dataset_opt = config.get("dataset", {})
    dataset_mode = dataset_opt.get("mode", "paired").lower()
    packing_opt = config.get("packing", {})
    patching_opt = config.get("patching", {"mode": "none"})
    seed = int(config.get("seed", 1234))
    rng = random.Random(seed)
    show_progress = bool(packing_opt.get("show_progress", True))

    output_states = _open_output_states(config)
    key_index = 0

    if dataset_mode == "paired":
        pairs, lq_opt, gt_opt = _prepare_paired_samples(config)
        progress = tqdm(total=len(pairs), desc="Preparing paired data", unit="sample") if show_progress and tqdm is not None else None
        for lq_path, gt_path in pairs:
            lq_frame = _apply_pipeline(_load_frame(lq_path, lq_opt), lq_opt.get("pipeline", []))
            gt_frame = _apply_pipeline(_load_frame(gt_path, gt_opt), gt_opt.get("pipeline", []))
            key_index = _process_sample(
                branch_frames={"lq": lq_frame, "gt": gt_frame},
                source_paths={"lq": lq_path, "gt": gt_path},
                patching_opt=patching_opt,
                output_states=output_states,
                rng=rng,
                key_index=key_index,
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(f"patches={key_index}")
        if progress is not None:
            progress.close()
    elif dataset_mode == "single":
        paths, source_opt = _prepare_single_samples(config)
        branches_opt = dataset_opt.get("branches", {})
        if not branches_opt:
            raise SystemExit("single dataset mode requires dataset.branches.")
        progress = tqdm(total=len(paths), desc="Preparing single-source data", unit="sample") if show_progress and tqdm is not None else None
        for source_path in paths:
            source_frame = _load_frame(source_path, source_opt)
            branch_frames: Dict[str, Frame] = {}
            for branch_name in ("lq", "gt"):
                branch_opt = branches_opt.get(branch_name)
                if not branch_opt or not branch_opt.get("enabled", True):
                    continue
                branch_frames[branch_name] = _apply_pipeline(source_frame, branch_opt.get("pipeline", []))
            key_index = _process_sample(
                branch_frames=branch_frames,
                source_paths={"source": source_path},
                patching_opt=patching_opt,
                output_states=output_states,
                rng=rng,
                key_index=key_index,
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(f"patches={key_index}")
        if progress is not None:
            progress.close()
    else:
        raise SystemExit(f"Unsupported dataset.mode: {dataset_mode}")

    for state in output_states.values():
        _finalize_output(state)

    output_root = Path(packing_opt["output_root"])
    print(f"Done. Wrote {key_index} samples to {output_root}")


if __name__ == "__main__":
    main()
