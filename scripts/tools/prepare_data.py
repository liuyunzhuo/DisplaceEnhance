from __future__ import annotations

import argparse
import io
import random
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import lmdb
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.color import rgb_to_yuv444_product, yuv444_to_rgb_product  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MATRIX_COEFFICIENTS = {
    "bt601": (0.2990, 0.1140),
    "bt709": (0.2126, 0.0722),
}
RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}
DEFAULT_RAW_INFO_PATTERN = (
    r"^(?P<key>.+?)_"
    r"(?P<width>\d+)x(?P<height>\d+)_"
    r"(?P<pixel_format>(?:nv12|(?:yuv|i)?(?:420p|444p|420|444)))"
    r"\.yuv$"
)


@dataclass
class Frame:
    color_space: str
    data: Any
    width: int
    height: int


@dataclass
class SourceEntry:
    source_name: str
    path: Path
    key: str
    raw_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    key: str
    frames: dict[str, Frame]
    source_entries: dict[str, SourceEntry]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputState:
    name: str
    backend: str
    frame_name: str
    output_path: Path
    format_name: str
    filename_pattern: str
    key_pattern: str
    start_index: int
    png_compress_level: int
    map_size: Optional[int] = None
    env: Optional[lmdb.Environment] = None
    write_batch_size: int = 512
    compact: bool = False
    batch: list[tuple[bytes, bytes]] = field(default_factory=list)
    meta_lines: list[str] = field(default_factory=list)
    counter: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run operator-based data preparation from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit("Config root must be a YAML mapping.")
    return data


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_source_files(source_opt: dict[str, Any], base_dir: Path) -> tuple[Path, ...]:
    root = _resolve_path(base_dir, source_opt["root"])
    if not root.exists():
        raise SystemExit(f"Source folder not found: {root}")

    glob_pattern = source_opt.get("glob")
    decoder_opt = source_opt.get("decoder", {}) or {}
    decoder_type = str(decoder_opt.get("type", "image")).lower()
    if glob_pattern:
        return tuple(sorted(p for p in root.glob(glob_pattern) if p.is_file()))
    if decoder_type == "image":
        return tuple(sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
    return tuple(sorted(p for p in root.iterdir() if p.is_file()))


def _extract_key_with_pattern(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.name, flags=re.IGNORECASE)
    if match is None:
        raise SystemExit(f"Could not extract key from filename: {path.name} using pattern: {pattern}")
    if "key" in match.groupdict() and match.group("key") is not None:
        return match.group("key")
    if match.groups():
        return match.group(1)
    return match.group(0)


def _normalize_pixel_format(pixel_format: str) -> str:
    value = pixel_format.lower()
    if value == "nv12":
        return "nv12"
    if value in ("420", "420p", "yuv420p", "i420", "i420p"):
        return "yuv420p"
    if value in ("444", "444p", "yuv444p", "i444", "i444p"):
        return "yuv444p"
    raise SystemExit(f"Unsupported raw_yuv pixel_format: {pixel_format}")


def _parse_raw_info(path: Path, decoder_opt: dict[str, Any]) -> dict[str, Any]:
    pattern = decoder_opt.get("info_pattern", decoder_opt.get("filename_pattern", DEFAULT_RAW_INFO_PATTERN))
    match = re.search(pattern, path.name, flags=re.IGNORECASE)
    if match is None:
        raise SystemExit(f"Could not parse raw_yuv info from filename: {path.name}")

    info = match.groupdict()
    result: dict[str, Any] = {}
    key = info.get("key")
    if key:
        result["key"] = key
    width = info.get("width")
    height = info.get("height")
    pixel_format = info.get("pixel_format")
    if width is not None:
        result["width"] = int(width)
    if height is not None:
        result["height"] = int(height)
    if pixel_format is not None:
        result["pixel_format"] = _normalize_pixel_format(pixel_format)
    if "width" not in result and decoder_opt.get("width") is not None:
        result["width"] = int(decoder_opt["width"])
    if "height" not in result and decoder_opt.get("height") is not None:
        result["height"] = int(decoder_opt["height"])
    if "pixel_format" not in result and decoder_opt.get("pixel_format") is not None:
        result["pixel_format"] = _normalize_pixel_format(str(decoder_opt["pixel_format"]))
    if "width" not in result or "height" not in result or "pixel_format" not in result:
        raise SystemExit(
            "raw_yuv decoder requires width, height, and pixel_format, "
            "either from info_pattern or explicit decoder fields."
        )
    return result


def _build_source_entries(
    source_name: str,
    source_opt: dict[str, Any],
    base_dir: Path,
) -> tuple[SourceEntry, ...]:
    decoder_opt = source_opt.get("decoder", {}) or {}
    decoder_type = str(decoder_opt.get("type", "image")).lower()
    key_pattern = source_opt.get("key_pattern")

    entries: list[SourceEntry] = []
    for path in _list_source_files(source_opt, base_dir):
        raw_info: dict[str, Any] = {}
        if decoder_type == "raw_yuv":
            raw_info = _parse_raw_info(path, decoder_opt)
        if key_pattern:
            key = _extract_key_with_pattern(path, key_pattern)
        elif raw_info.get("key"):
            key = str(raw_info["key"])
        else:
            key = path.stem
        entries.append(SourceEntry(source_name=source_name, path=path, key=key, raw_info=raw_info))
    if not entries:
        raise SystemExit(f"No source files found for source '{source_name}'.")
    return tuple(entries)


def _load_image_frame(path: Path, decoder_opt: dict[str, Any]) -> Frame:
    color_space = str(decoder_opt.get("color_space", "rgb")).lower()
    if color_space not in ("rgb", "yuv444"):
        raise SystemExit(f"Unsupported image decoder color_space: {color_space}")
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    return Frame(color_space=color_space, data=array, width=width, height=height)


def _load_raw_yuv_frame(entry: SourceEntry) -> Frame:
    info = entry.raw_info
    pixel_format = str(info["pixel_format"])
    width = int(info["width"])
    height = int(info["height"])
    raw = np.frombuffer(entry.path.read_bytes(), dtype=np.uint8)

    if pixel_format == "nv12":
        if width % 2 != 0 or height % 2 != 0:
            raise SystemExit(f"nv12 requires even width/height: {entry.path.name}")
        plane = width * height
        chroma_bytes = plane // 2
        expected = plane + chroma_bytes
        if raw.size != expected:
            raise SystemExit(f"{entry.path.name} size mismatch for nv12: expected {expected}, got {raw.size}")
        y = raw[:plane].reshape(height, width)
        uv = raw[plane:].reshape(height // 2, width)
        u = uv[:, 0::2]
        v = uv[:, 1::2]
        return Frame(color_space="yuv420", data=(y, u, v), width=width, height=height)

    if pixel_format == "yuv420p":
        if width % 2 != 0 or height % 2 != 0:
            raise SystemExit(f"yuv420p requires even width/height: {entry.path.name}")
        plane = width * height
        chroma_plane = (width // 2) * (height // 2)
        expected = plane + chroma_plane * 2
        if raw.size != expected:
            raise SystemExit(f"{entry.path.name} size mismatch for yuv420p: expected {expected}, got {raw.size}")
        y = raw[:plane].reshape(height, width)
        u = raw[plane : plane + chroma_plane].reshape(height // 2, width // 2)
        v = raw[plane + chroma_plane : plane + chroma_plane * 2].reshape(height // 2, width // 2)
        return Frame(color_space="yuv420", data=(y, u, v), width=width, height=height)

    if pixel_format == "yuv444p":
        plane = width * height
        expected = plane * 3
        if raw.size != expected:
            raise SystemExit(f"{entry.path.name} size mismatch for yuv444p: expected {expected}, got {raw.size}")
        y = raw[:plane].reshape(height, width)
        u = raw[plane : plane * 2].reshape(height, width)
        v = raw[plane * 2 : plane * 3].reshape(height, width)
        return Frame(color_space="yuv444", data=np.stack([y, u, v], axis=2), width=width, height=height)

    raise SystemExit(f"Unsupported raw_yuv pixel_format: {pixel_format}")


def _load_frame(entry: SourceEntry, source_opt: dict[str, Any]) -> Frame:
    decoder_opt = source_opt.get("decoder", {}) or {}
    decoder_type = str(decoder_opt.get("type", "image")).lower()
    if decoder_type == "image":
        return _load_image_frame(entry.path, decoder_opt)
    if decoder_type == "raw_yuv":
        return _load_raw_yuv_frame(entry)
    raise SystemExit(f"Unsupported decoder type: {decoder_type}")


@dataclass
class JobSources:
    mode: str
    sources_opt: dict[str, Any]
    built_entries: dict[str, tuple[SourceEntry, ...]]
    pair_by_order: bool
    keyed_maps: Optional[dict[str, dict[str, SourceEntry]]] = None
    common_keys: Optional[tuple[str, ...]] = None
    total: int = 0


def _prepare_job_sources(job_opt: dict[str, Any], base_dir: Path) -> JobSources:
    sources_opt = job_opt.get("sources")
    if not isinstance(sources_opt, dict) or not sources_opt:
        raise SystemExit("Each job must define a non-empty 'sources' mapping.")

    mode = str(job_opt.get("mode", "single")).lower()
    built: dict[str, tuple[SourceEntry, ...]] = {
        source_name: _build_source_entries(source_name, source_opt, base_dir)
        for source_name, source_opt in sources_opt.items()
    }

    if mode == "single":
        if len(built) != 1:
            raise SystemExit("mode: single requires exactly one source.")
        source_name, entries = next(iter(built.items()))
        total = len(entries)
        return JobSources(
            mode=mode,
            sources_opt=sources_opt,
            built_entries=built,
            pair_by_order=False,
            total=total,
        )

    if mode != "paired":
        raise SystemExit(f"Unsupported job mode: {mode}")

    pair_by_order = bool(job_opt.get("pair_by_order", False))
    source_names = list(built.keys())
    if pair_by_order:
        counts = {name: len(entries) for name, entries in built.items()}
        if len(set(counts.values())) != 1:
            raise SystemExit(f"pair_by_order requires equal source counts, got: {counts}")
        total = int(next(iter(counts.values())))
        return JobSources(
            mode=mode,
            sources_opt=sources_opt,
            built_entries=built,
            pair_by_order=True,
            total=total,
        )

    keyed_maps: dict[str, dict[str, SourceEntry]] = {}
    for source_name, entries in built.items():
        mapping: dict[str, SourceEntry] = {}
        for entry in entries:
            if entry.key in mapping:
                raise SystemExit(f"Duplicate key '{entry.key}' in source '{source_name}'")
            mapping[entry.key] = entry
        keyed_maps[source_name] = mapping

    common_keys = set(next(iter(keyed_maps.values())).keys())
    for mapping in keyed_maps.values():
        common_keys &= set(mapping.keys())
    if not common_keys:
        raise SystemExit("No common keys found across paired sources.")

    for source_name, mapping in keyed_maps.items():
        missing = sorted(set(mapping.keys()) - common_keys)
        if missing:
            print(f"Warning: source '{source_name}' has {len(missing)} unmatched files.")

    ordered_keys = tuple(sorted(common_keys))
    return JobSources(
        mode=mode,
        sources_opt=sources_opt,
        built_entries=built,
        pair_by_order=False,
        keyed_maps=keyed_maps,
        common_keys=ordered_keys,
        total=len(ordered_keys),
    )


def _iter_initial_samples(job_sources: JobSources) -> Iterable[Sample]:
    sources_opt = job_sources.sources_opt
    if job_sources.mode == "single":
        source_name, entries = next(iter(job_sources.built_entries.items()))
        source_opt = sources_opt[source_name]
        for entry in entries:
            frame = _load_frame(entry, source_opt)
            yield Sample(key=entry.key, frames={source_name: frame}, source_entries={source_name: entry})
        return

    source_names = list(job_sources.built_entries.keys())
    if job_sources.pair_by_order:
        total = job_sources.total
        for idx in range(total):
            sample_entries = {name: job_sources.built_entries[name][idx] for name in source_names}
            sample_key = next(iter(sample_entries.values())).key
            frames = {name: _load_frame(sample_entries[name], sources_opt[name]) for name in source_names}
            yield Sample(key=sample_key, frames=frames, source_entries=sample_entries)
        return

    assert job_sources.keyed_maps is not None
    assert job_sources.common_keys is not None
    for key in job_sources.common_keys:
        sample_entries = {name: job_sources.keyed_maps[name][key] for name in source_names}
        frames = {name: _load_frame(sample_entries[name], sources_opt[name]) for name in source_names}
        yield Sample(key=key, frames=frames, source_entries=sample_entries)


def _matrix_coefficients(matrix: str) -> tuple[float, float, float]:
    key = matrix.lower()
    if key not in MATRIX_COEFFICIENTS:
        raise SystemExit(f"Unsupported matrix: {matrix}")
    kr, kb = MATRIX_COEFFICIENTS[key]
    kg = 1.0 - kr - kb
    return kr, kg, kb


def _clip_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array), 0, 255).astype(np.uint8)


def _rgb_to_yuv444(frame: Frame, matrix: str, value_range: str) -> Frame:
    if frame.color_space != "rgb":
        raise SystemExit("rgb_to_yuv444 requires an RGB frame.")
    if matrix.lower() in ("internal_product", "product"):
        y, u, v = rgb_to_yuv444_product(frame.data)
        packed = np.stack([y, u, v], axis=2)
        return Frame(color_space="yuv444", data=packed, width=frame.width, height=frame.height)

    kr, kg, kb = _matrix_coefficients(matrix)
    rgb = frame.data.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    y = kr * r + kg * g + kb * b
    cb = 0.5 * (b - y) / (1.0 - kb)
    cr = 0.5 * (r - y) / (1.0 - kr)

    if value_range.lower() == "limited":
        y_plane = 16.0 + 219.0 * y
        u_plane = 128.0 + 224.0 * cb
        v_plane = 128.0 + 224.0 * cr
    elif value_range.lower() == "full":
        y_plane = 255.0 * y
        u_plane = 128.0 + 255.0 * cb
        v_plane = 128.0 + 255.0 * cr
    else:
        raise SystemExit(f"Unsupported range: {value_range}")

    packed = np.stack([_clip_uint8(y_plane), _clip_uint8(u_plane), _clip_uint8(v_plane)], axis=2)
    return Frame(color_space="yuv444", data=packed, width=frame.width, height=frame.height)


def _yuv444_to_rgb(frame: Frame, matrix: str, value_range: str) -> Frame:
    if frame.color_space != "yuv444":
        raise SystemExit("yuv444_to_rgb requires a yuv444 frame.")
    if matrix.lower() in ("internal_product", "product"):
        rgb = yuv444_to_rgb_product(frame.data)
        return Frame(color_space="rgb", data=rgb, width=frame.width, height=frame.height)

    kr, kg, kb = _matrix_coefficients(matrix)
    yuv = frame.data.astype(np.float32)
    y_plane = yuv[:, :, 0]
    u_plane = yuv[:, :, 1]
    v_plane = yuv[:, :, 2]

    if value_range.lower() == "limited":
        y = np.clip((y_plane - 16.0) / 219.0, 0.0, 1.0)
        cb = (u_plane - 128.0) / 224.0
        cr = (v_plane - 128.0) / 224.0
    elif value_range.lower() == "full":
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
    if frame.color_space == "yuv444":
        return frame
    if frame.color_space != "yuv420":
        raise SystemExit("yuv420_to_yuv444_nn requires a YUV420 frame.")
    y, u420, v420 = frame.data
    u = np.repeat(np.repeat(u420, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
    v = np.repeat(np.repeat(v420, 2, axis=0), 2, axis=1)[: frame.height, : frame.width]
    return Frame(color_space="yuv444", data=np.stack([y, u, v], axis=2), width=frame.width, height=frame.height)


def _resize_rgb_like(data: np.ndarray, width: int, height: int, resample_name: str) -> np.ndarray:
    if resample_name not in RESAMPLE_MAP:
        raise SystemExit(f"Unsupported resize resample: {resample_name}")
    resized_channels: list[np.ndarray] = []
    for channel_idx in range(3):
        image = Image.fromarray(data[:, :, channel_idx], mode="L")
        resized = image.resize((width, height), resample=RESAMPLE_MAP[resample_name])
        resized_channels.append(np.array(resized, dtype=np.uint8))
    return np.stack(resized_channels, axis=2)


def _resize_frame(frame: Frame, width: int, height: int, resample_name: str) -> Frame:
    if frame.color_space in ("rgb", "yuv444"):
        resized = _resize_rgb_like(frame.data, width, height, resample_name)
        return Frame(color_space=frame.color_space, data=resized, width=width, height=height)
    raise SystemExit("resize currently supports rgb and yuv444 frames.")


def _pad_offsets(total: int, size: int, align: str) -> int:
    if align == "center":
        return max(0, (total - size) // 2)
    if align == "top_left":
        return 0
    raise SystemExit(f"Unsupported pad_align: {align}")


def _resolve_pad_color(frame: Frame, pad_color: Optional[list[int] | tuple[int, int, int]]) -> tuple[int, int, int]:
    if pad_color is not None:
        if len(pad_color) != 3:
            raise SystemExit("pad_color must be a 3-element list like [0,0,0].")
        return int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
    if frame.color_space == "rgb":
        return 0, 0, 0
    if frame.color_space in ("yuv444", "yuv420"):
        return 0, 128, 128
    raise SystemExit(f"Unsupported frame color_space for padding: {frame.color_space}")


def _pad_frame(frame: Frame, target_w: int, target_h: int, align: str, pad_color: tuple[int, int, int]) -> Frame:
    if frame.width == target_w and frame.height == target_h:
        return frame
    if frame.width > target_w or frame.height > target_h:
        raise SystemExit("pad_frame expects target size >= source size.")

    offset_x = _pad_offsets(target_w, frame.width, align)
    offset_y = _pad_offsets(target_h, frame.height, align)

    if frame.color_space == "rgb":
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:, :, 0] = pad_color[0]
        canvas[:, :, 1] = pad_color[1]
        canvas[:, :, 2] = pad_color[2]
        canvas[offset_y : offset_y + frame.height, offset_x : offset_x + frame.width, :] = frame.data
        return Frame(color_space="rgb", data=canvas, width=target_w, height=target_h)

    if frame.color_space == "yuv444":
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:, :, 0] = pad_color[0]
        canvas[:, :, 1] = pad_color[1]
        canvas[:, :, 2] = pad_color[2]
        canvas[offset_y : offset_y + frame.height, offset_x : offset_x + frame.width, :] = frame.data
        return Frame(color_space="yuv444", data=canvas, width=target_w, height=target_h)

    if frame.color_space == "yuv420":
        if target_w % 2 != 0 or target_h % 2 != 0:
            raise SystemExit("pad_frame for yuv420 requires even target width/height.")
        if offset_x % 2 != 0:
            offset_x -= 1
        if offset_y % 2 != 0:
            offset_y -= 1
        y_plane, u_plane, v_plane = frame.data
        y_canvas = np.full((target_h, target_w), pad_color[0], dtype=np.uint8)
        u_canvas = np.full((target_h // 2, target_w // 2), pad_color[1], dtype=np.uint8)
        v_canvas = np.full((target_h // 2, target_w // 2), pad_color[2], dtype=np.uint8)
        y_canvas[offset_y : offset_y + frame.height, offset_x : offset_x + frame.width] = y_plane
        u_canvas[
            offset_y // 2 : offset_y // 2 + u_plane.shape[0],
            offset_x // 2 : offset_x // 2 + u_plane.shape[1],
        ] = u_plane
        v_canvas[
            offset_y // 2 : offset_y // 2 + v_plane.shape[0],
            offset_x // 2 : offset_x // 2 + v_plane.shape[1],
        ] = v_plane
        return Frame(color_space="yuv420", data=(y_canvas, u_canvas, v_canvas), width=target_w, height=target_h)

    raise SystemExit(f"Unsupported frame color_space for padding: {frame.color_space}")


def _rotate_frame(frame: Frame, k: int) -> Frame:
    if frame.color_space in ("rgb", "yuv444"):
        rotated = np.rot90(frame.data, k=k)
        height, width = rotated.shape[:2]
        return Frame(color_space=frame.color_space, data=rotated, width=width, height=height)
    if frame.color_space == "yuv420":
        y_plane, u_plane, v_plane = frame.data
        y_rot = np.rot90(y_plane, k=k)
        u_rot = np.rot90(u_plane, k=k)
        v_rot = np.rot90(v_plane, k=k)
        height, width = y_rot.shape[:2]
        return Frame(color_space="yuv420", data=(y_rot, u_rot, v_rot), width=width, height=height)
    raise SystemExit(f"Unsupported frame color_space for rotate: {frame.color_space}")


def _crop_frame(frame: Frame, x: int, y: int, width: int, height: int) -> Frame:
    if frame.color_space in ("rgb", "yuv444"):
        cropped = frame.data[y : y + height, x : x + width, :]
        return Frame(color_space=frame.color_space, data=cropped, width=width, height=height)
    if frame.color_space == "yuv420":
        if x % 2 != 0 or y % 2 != 0 or width % 2 != 0 or height % 2 != 0:
            raise SystemExit("crop on yuv420 requires even x, y, width, and height.")
        y_plane, u_plane, v_plane = frame.data
        y_crop = y_plane[y : y + height, x : x + width]
        u_crop = u_plane[y // 2 : (y + height) // 2, x // 2 : (x + width) // 2]
        v_crop = v_plane[y // 2 : (y + height) // 2, x // 2 : (x + width) // 2]
        return Frame(color_space="yuv420", data=(y_crop, u_crop, v_crop), width=width, height=height)
    raise SystemExit(f"Unsupported frame color_space: {frame.color_space}")


def _build_positions(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size > length:
        return []
    positions = list(range(0, max(length - crop_size + 1, 1), stride))
    last_pos = length - crop_size
    if not positions or positions[-1] != last_pos:
        positions.append(last_pos)
    return sorted(set(positions))


def _iter_crop_coords(
    width: int,
    height: int,
    crop_width: int,
    crop_height: int,
    mode: str,
    stride_x: int,
    stride_y: int,
    num_random: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    if mode == "grid":
        if crop_width > width or crop_height > height:
            return tuple()
        return tuple(
            (x, y)
            for y in range(0, height - crop_height + 1, stride_y)
            for x in range(0, width - crop_width + 1, stride_x)
        )
    if mode == "grid_full":
        x_positions = _build_positions(width, crop_width, stride_x)
        y_positions = _build_positions(height, crop_height, stride_y)
        return tuple((x, y) for y in y_positions for x in x_positions)
    if mode == "center":
        if crop_width > width or crop_height > height:
            return tuple()
        return (((width - crop_width) // 2, (height - crop_height) // 2),)
    if mode == "random":
        return tuple(
            (rng.randint(0, width - crop_width), rng.randint(0, height - crop_height))
            for _ in range(num_random)
        )
    raise SystemExit(f"Unsupported crop mode: {mode}")


def _op_copy_frame(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    src = str(op["src"])
    dst = str(op.get("dst", src))
    for sample in samples:
        if src not in sample.frames:
            raise SystemExit(f"copy_frame missing source frame: {src}")
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        cloned.frames[dst] = sample.frames[src]
        yield cloned


def _op_rgb_to_yuv444(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    frame_name = str(op["frame"])
    out_name = str(op.get("out", frame_name))
    matrix = str(op.get("matrix", "internal_product"))
    value_range = str(op.get("range", "full"))
    for sample in samples:
        frame = sample.frames.get(frame_name)
        if frame is None:
            raise SystemExit(f"rgb_to_yuv444 missing frame: {frame_name}")
        converted = _rgb_to_yuv444(frame, matrix=matrix, value_range=value_range)
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        cloned.frames[out_name] = converted
        yield cloned


def _op_yuv420_to_yuv444_nn(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    frame_name = str(op["frame"])
    out_name = str(op.get("out", frame_name))
    for sample in samples:
        frame = sample.frames.get(frame_name)
        if frame is None:
            raise SystemExit(f"yuv420_to_yuv444_nn missing frame: {frame_name}")
        converted = _yuv420_to_yuv444_nn(frame)
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        cloned.frames[out_name] = converted
        yield cloned


def _op_yuv444_to_rgb(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    frame_name = str(op["frame"])
    out_name = str(op.get("out", frame_name))
    matrix = str(op.get("matrix", "internal_product"))
    value_range = str(op.get("range", "full"))
    for sample in samples:
        frame = sample.frames.get(frame_name)
        if frame is None:
            raise SystemExit(f"yuv444_to_rgb missing frame: {frame_name}")
        converted = _yuv444_to_rgb(frame, matrix=matrix, value_range=value_range)
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        cloned.frames[out_name] = converted
        yield cloned


def _op_resize(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    width_value = op.get("width", op.get("size"))
    height_value = op.get("height", op.get("size", width_value))
    if width_value is None or height_value is None:
        raise SystemExit("resize requires width/height or size.")
    target_width = int(width_value)
    target_height = int(height_value)
    resample = str(op.get("resample", "bicubic")).lower()
    frames_opt = op.get("frames")
    for sample in samples:
        frame_names = list(sample.frames.keys()) if frames_opt in (None, "all") else list(frames_opt)
        if not frame_names:
            raise SystemExit("resize op did not resolve any target frames.")
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        for frame_name in frame_names:
            frame = sample.frames.get(frame_name)
            if frame is None:
                raise SystemExit(f"resize missing frame: {frame_name}")
            cloned.frames[frame_name] = _resize_frame(frame, target_width, target_height, resample)
        yield cloned


def _op_resize_if_ratio_close(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    width_value = op.get("width")
    height_value = op.get("height")
    if width_value is None or height_value is None:
        raise SystemExit("resize_if_ratio_close requires width and height.")
    target_width = int(width_value)
    target_height = int(height_value)
    if target_width <= 0 or target_height <= 0:
        raise SystemExit("resize_if_ratio_close requires positive width/height.")
    tolerance = float(op.get("tolerance", 0.05))
    allow_upscale = bool(op.get("allow_upscale", True))
    resample = str(op.get("resample", "bicubic")).lower()
    frames_opt = op.get("frames")

    target_ratio = target_width / target_height
    for sample in samples:
        frame_names = list(sample.frames.keys()) if frames_opt in (None, "all") else list(frames_opt)
        if not frame_names:
            raise SystemExit("resize_if_ratio_close did not resolve any target frames.")
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        for frame_name in frame_names:
            frame = sample.frames.get(frame_name)
            if frame is None:
                raise SystemExit(f"resize_if_ratio_close missing frame: {frame_name}")
            if frame.height <= 0:
                raise SystemExit("resize_if_ratio_close encountered invalid frame height.")
            if not allow_upscale and (frame.width < target_width or frame.height < target_height):
                continue
            ratio = frame.width / frame.height
            if abs(ratio / target_ratio - 1.0) <= tolerance:
                cloned.frames[frame_name] = _resize_frame(frame, target_width, target_height, resample)
        yield cloned


def _op_rotate_if_portrait(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    direction = str(op.get("direction", "clockwise")).lower()
    frames_opt = op.get("frames")
    if direction in ("clockwise", "cw"):
        k = 3
    elif direction in ("counterclockwise", "ccw"):
        k = 1
    else:
        raise SystemExit(f"Unsupported rotate_if_portrait direction: {direction}")

    for sample in samples:
        frame_names = list(sample.frames.keys()) if frames_opt in (None, "all") else list(frames_opt)
        if not frame_names:
            raise SystemExit("rotate_if_portrait did not resolve any target frames.")
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        for frame_name in frame_names:
            frame = sample.frames.get(frame_name)
            if frame is None:
                raise SystemExit(f"rotate_if_portrait missing frame: {frame_name}")
            if frame.height > frame.width:
                cloned.frames[frame_name] = _rotate_frame(frame, k=k)
        yield cloned


def _op_merge_yuv(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    _ = rng
    y_from = str(op["y_from"])
    uv_from = str(op["uv_from"])
    out_name = str(op.get("out", "merged"))
    for sample in samples:
        y_frame = sample.frames.get(y_from)
        uv_frame = sample.frames.get(uv_from)
        if y_frame is None or uv_frame is None:
            raise SystemExit(f"merge_yuv missing frames: y_from={y_from}, uv_from={uv_from}")
        if y_frame.color_space != "yuv444" or uv_frame.color_space != "yuv444":
            raise SystemExit("merge_yuv requires yuv444 frames on both inputs.")
        if y_frame.width != uv_frame.width or y_frame.height != uv_frame.height:
            raise SystemExit(
                f"merge_yuv size mismatch: {y_frame.width}x{y_frame.height} vs {uv_frame.width}x{uv_frame.height}"
            )
        merged = np.stack(
            [y_frame.data[:, :, 0], uv_frame.data[:, :, 1], uv_frame.data[:, :, 2]],
            axis=2,
        )
        cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
        cloned.frames[out_name] = Frame(color_space="yuv444", data=merged, width=y_frame.width, height=y_frame.height)
        yield cloned


def _op_crop(samples: Iterable[Sample], op: dict[str, Any], rng: random.Random) -> Iterable[Sample]:
    crop_width = int(op.get("width", op.get("crop_width", 1920)))
    crop_height = int(op.get("height", op.get("crop_height", 1080)))
    mode = str(op.get("mode", "grid_full")).lower()
    stride_x = int(op.get("stride_x", crop_width))
    stride_y = int(op.get("stride_y", crop_height))
    num_random = int(op.get("num_random", 1))
    small_image = str(op.get("small_image", "skip")).lower()
    pad_align = str(op.get("pad_align", "center")).lower()
    pad_color = op.get("pad_color")
    reference = str(op.get("reference", op.get("frame", ""))).strip()
    frames_opt = op.get("frames")
    for sample in samples:
        frame_names = list(sample.frames.keys()) if frames_opt in (None, "all") else list(frames_opt)
        if not frame_names:
            raise SystemExit("crop op did not resolve any target frames.")
        ref_name = reference or frame_names[0]
        if ref_name not in sample.frames:
            raise SystemExit(f"crop reference frame not found: {ref_name}")
        ref_frame = sample.frames[ref_name]
        for frame_name in frame_names:
            frame = sample.frames.get(frame_name)
            if frame is None:
                raise SystemExit(f"crop missing frame: {frame_name}")
            if frame.width != ref_frame.width or frame.height != ref_frame.height:
                raise SystemExit(f"crop requires aligned frame sizes, but {frame_name} differs from {ref_name}")
        if ref_frame.width < crop_width or ref_frame.height < crop_height:
            if small_image == "skip":
                continue
            if small_image != "pad":
                raise SystemExit(f"Unsupported small_image mode: {small_image}")
            resolved_pad = _resolve_pad_color(ref_frame, pad_color)
            padded = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
            for frame_name in frame_names:
                frame = sample.frames[frame_name]
                crop_w = min(frame.width, crop_width)
                crop_h = min(frame.height, crop_height)
                if frame.color_space == "yuv420":
                    if crop_w % 2 != 0:
                        crop_w -= 1
                    if crop_h % 2 != 0:
                        crop_h -= 1
                x = max(0, (frame.width - crop_w) // 2)
                y = max(0, (frame.height - crop_h) // 2)
                if frame.color_space == "yuv420":
                    if x % 2 != 0:
                        x -= 1
                    if y % 2 != 0:
                        y -= 1
                cropped = _crop_frame(frame, x, y, crop_w, crop_h)
                padded.frames[frame_name] = _pad_frame(
                    cropped,
                    crop_width,
                    crop_height,
                    pad_align,
                    resolved_pad,
                )
            padded.meta["patch_index"] = 1
            padded.meta["patch_x"] = 0
            padded.meta["patch_y"] = 0
            yield padded
            continue

        coords = _iter_crop_coords(
            width=ref_frame.width,
            height=ref_frame.height,
            crop_width=crop_width,
            crop_height=crop_height,
            mode=mode,
            stride_x=stride_x,
            stride_y=stride_y,
            num_random=num_random,
            rng=rng,
        )
        for patch_idx, (x, y) in enumerate(coords, start=1):
            cloned = replace(sample, frames=dict(sample.frames), meta=dict(sample.meta))
            cloned.meta["patch_index"] = patch_idx
            cloned.meta["patch_x"] = x
            cloned.meta["patch_y"] = y
            for frame_name in frame_names:
                cloned.frames[frame_name] = _crop_frame(sample.frames[frame_name], x, y, crop_width, crop_height)
            yield cloned


OP_REGISTRY = {
    "copy_frame": _op_copy_frame,
    "rgb_to_yuv444": _op_rgb_to_yuv444,
    "yuv420_to_yuv444_nn": _op_yuv420_to_yuv444_nn,
    "yuv444_to_rgb": _op_yuv444_to_rgb,
    "resize": _op_resize,
    "resize_if_ratio_close": _op_resize_if_ratio_close,
    "rotate_if_portrait": _op_rotate_if_portrait,
    "merge_yuv": _op_merge_yuv,
    "crop": _op_crop,
}


def _wrap_count(iterable: Iterable[Sample], label: str) -> Iterable[Sample]:
    count = 0
    for item in iterable:
        count += 1
        yield item
    print(f"  {label}: {count} sample(s)")


def _run_pipeline(samples: Iterable[Sample], pipeline: list[dict[str, Any]], rng: random.Random) -> Iterable[Sample]:
    current: Iterable[Sample] = samples
    for idx, op in enumerate(pipeline, start=1):
        if not isinstance(op, dict):
            raise SystemExit(f"pipeline[{idx}] must be a mapping.")
        op_name = str(op.get("op", "")).strip()
        if not op_name:
            raise SystemExit(f"pipeline[{idx}] is missing 'op'.")
        func = OP_REGISTRY.get(op_name)
        if func is None:
            raise SystemExit(f"Unsupported pipeline op: {op_name}")
        current = func(current, op, rng)
        current = _wrap_count(current, f"pipeline[{idx}] {op_name}")
    return current


def _default_extension(format_name: str) -> str:
    if format_name in ("rgb_png", "packed_yuv444_png"):
        return ".png"
    if format_name == "raw_yuv444":
        return ".yuv"
    raise SystemExit(f"Unsupported output format: {format_name}")


def _context_for_output(sample: Sample, frame: Frame, state: OutputState) -> dict[str, Any]:
    context = {
        "key": sample.key,
        "index": state.start_index + state.counter,
        "width": frame.width,
        "height": frame.height,
        "patch_index": sample.meta.get("patch_index", 1),
        "patch_x": sample.meta.get("patch_x", 0),
        "patch_y": sample.meta.get("patch_y", 0),
    }
    for source_name, entry in sample.source_entries.items():
        context[f"source_{source_name}_name"] = entry.path.name
        context[f"source_{source_name}_stem"] = entry.path.stem
    return context


def _render_pattern(pattern: str, context: dict[str, Any]) -> str:
    try:
        return pattern.format(**context)
    except KeyError as exc:
        raise SystemExit(f"Missing format key in pattern '{pattern}': {exc}") from exc


def _encode_frame(frame: Frame, format_name: str, png_compress_level: int) -> bytes:
    if format_name == "rgb_png":
        if frame.color_space != "rgb":
            raise SystemExit("rgb_png output requires an RGB frame.")
        image = Image.fromarray(frame.data, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=png_compress_level, optimize=True)
        return buffer.getvalue()

    if format_name == "packed_yuv444_png":
        if frame.color_space != "yuv444":
            raise SystemExit("packed_yuv444_png output requires a yuv444 frame.")
        image = Image.fromarray(frame.data, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=png_compress_level, optimize=True)
        return buffer.getvalue()

    if format_name == "raw_yuv444":
        if frame.color_space != "yuv444":
            raise SystemExit("raw_yuv444 output requires a yuv444 frame.")
        y = frame.data[:, :, 0].reshape(-1)
        u = frame.data[:, :, 1].reshape(-1)
        v = frame.data[:, :, 2].reshape(-1)
        return b"".join([y.tobytes(), u.tobytes(), v.tobytes()])

    raise SystemExit(f"Unsupported output format: {format_name}")


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


def _flush_items(env: lmdb.Environment, items: list[tuple[bytes, bytes]], cur_size: int, tag: str) -> int:
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


def _open_outputs(job_opt: dict[str, Any], base_dir: Path) -> dict[str, OutputState]:
    outputs_opt = job_opt.get("outputs")
    if not isinstance(outputs_opt, dict) or not outputs_opt:
        raise SystemExit("Each job must define a non-empty 'outputs' mapping.")

    states: dict[str, OutputState] = {}
    for output_name, output_opt in outputs_opt.items():
        backend = str(output_opt.get("backend", "files")).lower()
        frame_name = str(output_opt.get("frame", output_name))
        output_path = _resolve_path(base_dir, output_opt["output_path"])
        format_name = str(output_opt.get("format", "rgb_png"))
        filename_pattern = str(output_opt.get("filename_pattern", "{key}"))
        key_pattern = str(output_opt.get("key_pattern", filename_pattern))
        start_index = int(output_opt.get("start_index", 0))
        png_compress_level = int(output_opt.get("png_compress_level", 3))

        if backend == "files":
            _ensure_dir(output_path)
            env = None
            map_size = None
            write_batch_size = 0
            compact = False
        elif backend == "lmdb":
            _ensure_dir(output_path.parent)
            map_size = int(float(output_opt.get("map_size_gb", 0.5)) * 1024**3)
            env = lmdb.open(str(output_path), map_size=map_size, subdir=True, lock=False, readahead=False, meminit=False)
            write_batch_size = int(output_opt.get("write_batch_size", 512))
            compact = bool(output_opt.get("compact", False))
        else:
            raise SystemExit(f"Unsupported output backend: {backend}")

        states[output_name] = OutputState(
            name=output_name,
            backend=backend,
            frame_name=frame_name,
            output_path=output_path,
            format_name=format_name,
            filename_pattern=filename_pattern,
            key_pattern=key_pattern,
            start_index=start_index,
            png_compress_level=png_compress_level,
            map_size=map_size,
            env=env,
            write_batch_size=write_batch_size,
            compact=compact,
        )
    return states


def _write_sample_outputs(sample: Sample, output_states: dict[str, OutputState]) -> None:
    for state in output_states.values():
        frame = sample.frames.get(state.frame_name)
        if frame is None:
            raise SystemExit(f"Output '{state.name}' refers to missing frame: {state.frame_name}")
        context = _context_for_output(sample, frame, state)
        rendered_name = _render_pattern(state.filename_pattern, context)
        if "." not in Path(rendered_name).name:
            rendered_name = f"{rendered_name}{_default_extension(state.format_name)}"
        rendered_key = _render_pattern(state.key_pattern, context)
        if state.backend == "files" and "." not in Path(rendered_key).name:
            rendered_key = f"{rendered_key}{_default_extension(state.format_name)}"

        payload = _encode_frame(frame, format_name=state.format_name, png_compress_level=state.png_compress_level)
        meta_key = rendered_name if state.backend == "files" else rendered_key
        state.meta_lines.append(f"{meta_key} ({frame.height},{frame.width},3) {sample.key}\n")

        if state.backend == "files":
            (state.output_path / rendered_name).write_bytes(payload)
        else:
            assert state.env is not None
            assert state.map_size is not None
            state.batch.append((rendered_key.encode("ascii"), payload))
            if len(state.batch) >= state.write_batch_size:
                state.map_size = _flush_items(state.env, state.batch, state.map_size, state.name.upper())
                state.batch.clear()
        state.counter += 1


def _finalize_outputs(output_states: dict[str, OutputState]) -> None:
    for state in output_states.values():
        if state.backend == "lmdb":
            assert state.env is not None
            assert state.map_size is not None
            state.map_size = _flush_items(state.env, state.batch, state.map_size, state.name.upper())
            state.env.sync()
            state.env.close()
            (state.output_path / "meta_info.txt").write_text("".join(state.meta_lines), encoding="utf-8")
            if state.compact:
                _compact_lmdb(state.output_path)
        else:
            (state.output_path / "meta_info.txt").write_text("".join(state.meta_lines), encoding="utf-8")


def _run_job(job_opt: dict[str, Any], base_dir: Path, job_index: int, total_jobs: int) -> None:
    job_name = str(job_opt.get("name", f"job_{job_index}"))
    seed = int(job_opt.get("seed", 1234))
    rng = random.Random(seed)
    print(f"[{job_index}/{total_jobs}] Running job: {job_name}")

    job_sources = _prepare_job_sources(job_opt, base_dir)
    print(f"  loaded {job_sources.total} initial sample(s)")
    pipeline = job_opt.get("pipeline", []) or []
    samples_iter = _iter_initial_samples(job_sources)
    samples_iter = _run_pipeline(samples_iter, pipeline, rng)

    outputs = _open_outputs(job_opt, base_dir)
    wrote_count = 0
    for sample in samples_iter:
        wrote_count += 1
        _write_sample_outputs(sample, outputs)
        if wrote_count % 100 == 0:
            print(f"  wrote {wrote_count} sample(s)")
    if wrote_count == 0:
        raise SystemExit(f"Job '{job_name}' produced no samples.")
    print(f"  wrote {wrote_count} sample(s)")
    _finalize_outputs(outputs)
    print(f"[{job_index}/{total_jobs}] Finished job: {job_name}")


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("Config must contain a non-empty 'jobs' list.")

    print(f"Using config: {config_path}")
    for idx, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise SystemExit(f"jobs[{idx}] must be a mapping.")
        _run_job(job, config_dir, idx, len(jobs))
    print("All jobs completed.")


if __name__ == "__main__":
    main()
