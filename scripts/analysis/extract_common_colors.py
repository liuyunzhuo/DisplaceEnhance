from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the most common colors from a folder of images.")
    parser.add_argument("--img_dir", required=True, help="Image folder")
    parser.add_argument("--top_k", type=int, default=24, help="Number of colors to report")
    parser.add_argument(
        "--quantize_step",
        type=int,
        default=8,
        help="Bucket size for color quantization. Larger values merge nearby colors.",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=512,
        help="Resize long side before counting to speed up extraction",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output text path. Default: <img_dir>/common_colors.txt",
    )
    parser.add_argument(
        "--palette_png",
        default=None,
        help="Optional palette PNG path. Default: <img_dir>/common_colors.png",
    )
    return parser.parse_args()


def _iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _resize_for_count(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, resample=Image.BILINEAR)


def _quantize_color(rgb: Tuple[int, int, int], step: int) -> Tuple[int, int, int]:
    if step <= 1:
        return rgb
    return tuple(min(255, int(round(channel / step) * step)) for channel in rgb)


def _to_hex(rgb: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _write_palette_png(path: Path, colors: Sequence[Tuple[Tuple[int, int, int], int]]) -> None:
    if not colors:
        return
    swatch_w = 160
    swatch_h = 72
    image = Image.new("RGB", (swatch_w * len(colors), swatch_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for idx, (rgb, _) in enumerate(colors):
        x0 = idx * swatch_w
        x1 = x0 + swatch_w
        draw.rectangle((x0, 0, x1, swatch_h), fill=tuple(rgb))
        text = _to_hex(rgb)
        text_color = (0, 0, 0) if sum(rgb) > 382 else (255, 255, 255)
        draw.text((x0 + 10, swatch_h - 24), text, fill=text_color)
    image.save(path, format="PNG", compress_level=3, optimize=True)


def main() -> None:
    args = _parse_args()
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise SystemExit(f"Folder not found: {img_dir}")
    if args.top_k <= 0:
        raise SystemExit("top_k must be > 0")
    if args.quantize_step <= 0:
        raise SystemExit("quantize_step must be > 0")

    image_paths = list(_iter_images(img_dir, recursive=args.recursive))
    if not image_paths:
        raise SystemExit(f"No images found in: {img_dir}")

    counter: Counter[Tuple[int, int, int]] = Counter()
    total_pixels = 0

    for path in image_paths:
        with Image.open(path) as image:
            image = _resize_for_count(image.convert("RGB"), args.max_side)
            pixels = image.getdata()
            for rgb in pixels:
                counter[_quantize_color(rgb, args.quantize_step)] += 1
            total_pixels += image.width * image.height

    common = counter.most_common(args.top_k)
    out_path = Path(args.out) if args.out else (img_dir / "common_colors.txt")
    palette_path = Path(args.palette_png) if args.palette_png else (img_dir / "common_colors.png")

    lines: List[str] = []
    lines.append(f"images={len(image_paths)}")
    lines.append(f"pixels={total_pixels}")
    lines.append(f"quantize_step={args.quantize_step}")
    lines.append("")
    for idx, (rgb, count) in enumerate(common, start=1):
        ratio = (count / total_pixels) * 100.0 if total_pixels else 0.0
        lines.append(f"{idx:02d} {_to_hex(rgb)} rgb={rgb} count={count} ratio={ratio:.2f}%")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_palette_png(palette_path, common)

    print(f"Done. Wrote text summary to {out_path}")
    print(f"Done. Wrote palette PNG to {palette_path}")


if __name__ == "__main__":
    main()
