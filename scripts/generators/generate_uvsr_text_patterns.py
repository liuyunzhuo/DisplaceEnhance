from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Color = Tuple[int, int, int]

COMMON_PALETTE: Sequence[Color] = (
    (112, 48, 160),
    (0, 0, 0),
    (255, 192, 0),
    (192, 0, 0),
    (0, 176, 240),
    (232, 232, 232),
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (168, 168, 168),
    (0, 112, 192),
    (144, 208, 80),
    (0, 32, 96),
    (64, 112, 192),
    (255, 0, 255),
    (120, 112, 112),
    (240, 128, 48),
    (64, 80, 104),
    (255, 48, 208),
    (160, 160, 160),
    (255, 240, 0),
    (248, 248, 248),
    (232, 0, 0),
    (64, 88, 112),
    (8, 32, 96),
    (152, 208, 80),
    (0, 168, 232),
    (120, 120, 120),
    (255, 0, 232),
    (0, 120, 192),
    (248, 48, 200),
    (232, 128, 56),
    (80, 32, 112),
    (48, 184, 184),
    (152, 160, 176),
    (255, 0, 40),
    (144, 48, 168),
    (0, 72, 144),
    (192, 152, 128),
    (88, 48, 144),
)

DEFAULT_TEXT_ITEMS: Sequence[str] = (
    "\u6d4b\u8bd5",       # 测试
    "\u84dd\u8272",       # 蓝色
    "\u7eff\u8272",       # 绿色
    "\u6e05\u6670\u5ea6", # 清晰度
    "\u8fb9\u7f18",       # 边缘
    "\u6587\u5b57",       # 文字
    "\u83dc\u5355",       # 菜单
    "\u663e\u793a",       # 显示
    "\u6309\u94ae",       # 按钮
    "TEST",
    "MENU",
    "SAVE",
    "LOAD",
    "A1",
    "RGB",
    "123",
    "2026",
    ",.",
    "!?",
    "[]",
    "()",
    "\uff0c\u3002",       # ，。
    "\uff01\uff1f",       # ！？
    "\u3010\u3011",       # 【】
    "\uff08\uff09",       # （）
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic UVSR text-pattern images.")
    parser.add_argument("--out_dir", default="data/synth_uvsr_patterns", help="Output root")
    parser.add_argument("--count", type=int, default=32, help="Number of samples to generate")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--text", default=None, help="Single text token override")
    parser.add_argument("--texts", nargs="+", default=None, help="Multiple text tokens to cycle through")
    parser.add_argument(
        "--with_uvsr_pairs",
        action="store_true",
        help="Also export packed-YUV GT/LQ pairs under gt/ and lq/ subfolders",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_meta(folder: Path, names: Sequence[str]) -> None:
    (folder / "meta_info.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in (
        "msyh.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "NotoSansCJKsc-Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _jitter_color(color: Color, rng: random.Random, amount: int = 8) -> Color:
    return tuple(max(0, min(255, channel + rng.randint(-amount, amount))) for channel in color)


def _luma(color: Color) -> float:
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def _channel_spread(color: Color) -> int:
    return max(color) - min(color)


def _contrast_score(a: Color, b: Color) -> float:
    return abs(_luma(a) - _luma(b)) + 0.25 * sum(abs(x - y) for x, y in zip(a, b))


def _is_blue_green(color: Color) -> bool:
    r, g, b = color
    return g > r or b > r


def _is_neutral(color: Color) -> bool:
    return _channel_spread(color) < 28


def _pick_panel_text_colors(bg: Color, rng: random.Random) -> Sequence[Color]:
    contrast_candidates = [c for c in COMMON_PALETTE if _contrast_score(c, bg) > 70]
    blue_green_candidates = [c for c in contrast_candidates if _is_blue_green(c)]
    neutral_candidates = [c for c in contrast_candidates if _is_neutral(c)]
    accent_candidates = [c for c in contrast_candidates if not _is_blue_green(c) and not _is_neutral(c)]
    if not blue_green_candidates:
        blue_green_candidates = contrast_candidates or list(COMMON_PALETTE)
    if not accent_candidates:
        accent_candidates = contrast_candidates or list(COMMON_PALETTE)
    if not neutral_candidates:
        neutral_candidates = contrast_candidates or list(COMMON_PALETTE)

    picked: List[Color] = []
    pool_plan = [
        blue_green_candidates,
        blue_green_candidates,
        blue_green_candidates,
        accent_candidates,
        accent_candidates,
        neutral_candidates,
    ]
    for pool in pool_plan:
        picked.append(_jitter_color(pool[rng.randrange(len(pool))], rng, 4))
    return tuple(picked)


def _build_image_text_items(text_items: Sequence[str], rng: random.Random, target_count: int = 48) -> Sequence[str]:
    if not text_items:
        return ("TEST",)

    base = list(text_items)
    generated: List[str] = []
    punctuation = [",.", "!?", "[]", "()", "\uff0c\u3002", "\uff01\uff1f", "\u3010\u3011", "\uff08\uff09"]
    while len(generated) < target_count:
        rng.shuffle(base)
        generated.extend(base)
        generated.append(str(rng.randint(0, 9999)))
        generated.append(f"{rng.randint(1, 99):02d}:{rng.randint(0, 59):02d}")
        generated.append(punctuation[rng.randrange(len(punctuation))])
    return tuple(generated[:target_count])


def _build_panel_colors(rng: random.Random, num_panels: int) -> List[Color]:
    colors = list(COMMON_PALETTE)
    rng.shuffle(colors)
    palette: List[Color] = []
    for idx in range(num_panels):
        palette.append(_jitter_color(colors[idx % len(colors)], rng, 8))
    return palette


def _panel_layout(width: int, height: int, rng: random.Random) -> List[Tuple[int, int, int, int, Color]]:
    top = 0
    bottom = height
    panel_min = max(86, width // 24)
    panel_max = max(128, width // 16)

    widths: List[int] = []
    total = 0
    while total < width:
        remaining = width - total
        panel_w = min(remaining, rng.randint(panel_min, panel_max))
        if remaining < panel_w * 1.35:
            panel_w = remaining
        widths.append(panel_w)
        total += panel_w

    if total != width:
        widths[-1] += width - total

    colors = _build_panel_colors(rng, len(widths))
    panels: List[Tuple[int, int, int, int, Color]] = []
    x = 0
    for panel_w, color in zip(widths, colors):
        panels.append((x, top, x + panel_w, bottom, color))
        x += panel_w
    return panels


def _draw_background(width: int, height: int, rng: random.Random) -> tuple[Image.Image, List[Tuple[int, int, int, int, Color]]]:
    image = Image.new("RGB", (width, height), (32, 32, 34))
    draw = ImageDraw.Draw(image)
    panels = _panel_layout(width, height, rng)
    pixels = image.load()

    for x0, y0, x1, y1, color in panels:
        draw.rectangle((x0, y0, x1, y1), fill=color)
        for x in range(x0, x1):
            shade = 0.95 + 0.05 * math.sin((x - x0) / max(1, (x1 - x0)) * math.pi)
            line_color = tuple(max(0, min(255, int(channel * shade))) for channel in color)
            for y in range(y0, y1):
                pixels[x, y] = line_color
    return image, panels


def _draw_text_overlay(
    image: Image.Image,
    panels: Sequence[Tuple[int, int, int, int, Color]],
    text_items: Sequence[str],
    rng: random.Random,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font_sizes = [16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32]

    for panel_idx, (x0, y0, x1, y1, bg) in enumerate(panels):
        panel_w = x1 - x0
        inner_x0 = x0 + 6
        inner_x1 = x1 - 6
        inner_y0 = y0 + 20
        inner_y1 = y1 - 20
        cols = 1 if panel_w < 108 else 2
        x_positions = [int(inner_x0 + (idx + 1) * (inner_x1 - inner_x0) / (cols + 1)) for idx in range(cols)]
        row_count = len(font_sizes)
        y_positions = [int(inner_y0 + (idx + 0.5) * (inner_y1 - inner_y0) / row_count) for idx in range(row_count)]
        color_cycle = _pick_panel_text_colors(bg, rng)

        for row, (y, font_size) in enumerate(zip(y_positions, font_sizes)):
            font = _try_font(font_size)
            color = color_cycle[row % len(color_cycle)]
            text = text_items[(panel_idx + row) % len(text_items)]
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            for x_center in x_positions:
                x = int(x_center - text_w / 2)
                yy = int(y - text_h / 2)
                draw.text((x, yy), text, font=font, fill=(*color, 255))


def _rgb_to_yuv444_limited_bt709(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]
    kr = 0.2126
    kb = 0.0722
    kg = 1.0 - kr - kb
    y = kr * r + kg * g + kb * b
    cb = 0.5 * (b - y) / (1.0 - kb)
    cr = 0.5 * (r - y) / (1.0 - kr)
    y_plane = np.clip(np.round(16.0 + 219.0 * y), 0, 255).astype(np.uint8)
    u_plane = np.clip(np.round(128.0 + 224.0 * cb), 0, 255).astype(np.uint8)
    v_plane = np.clip(np.round(128.0 + 224.0 * cr), 0, 255).astype(np.uint8)
    return np.stack([y_plane, u_plane, v_plane], axis=2)


def _yuv444_to_yuv420(yuv444: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = yuv444[:, :, 0]
    u = yuv444[:, :, 1]
    v = yuv444[:, :, 2]
    h, w = y.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    y = y[:h2, :w2]
    u = u[:h2, :w2]
    v = v[:h2, :w2]
    u420 = u.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)).round().astype(np.uint8)
    v420 = v.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)).round().astype(np.uint8)
    return y, u420, v420


def _yuv420_to_yuv444_nn(y: np.ndarray, u420: np.ndarray, v420: np.ndarray) -> np.ndarray:
    h, w = y.shape
    u = np.repeat(np.repeat(u420, 2, axis=0), 2, axis=1)[:h, :w]
    v = np.repeat(np.repeat(v420, 2, axis=0), 2, axis=1)[:h, :w]
    return np.stack([y, u, v], axis=2)


def _save_packed_png(path: Path, yuv444: np.ndarray) -> None:
    Image.fromarray(yuv444, mode="RGB").save(path, format="PNG", compress_level=3, optimize=True)


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    if args.texts:
        text_items = tuple(args.texts)
    elif args.text:
        text_items = (args.text,)
    else:
        text_items = DEFAULT_TEXT_ITEMS

    out_root = Path(args.out_dir)
    _ensure_dir(out_root)

    gt_dir = out_root / "gt"
    lq_dir = out_root / "lq"
    if args.with_uvsr_pairs:
        _ensure_dir(gt_dir)
        _ensure_dir(lq_dir)

    names: List[str] = []
    for idx in range(args.count):
        image_text_items = _build_image_text_items(text_items, rng)
        image, panels = _draw_background(args.width, args.height, rng)
        _draw_text_overlay(image, panels, image_text_items, rng)
        rgb = np.array(image, dtype=np.uint8)

        name = f"pattern_{idx:04d}.png"
        names.append(name)
        image.save(out_root / name, format="PNG", compress_level=3, optimize=True)

        if args.with_uvsr_pairs:
            yuv444 = _rgb_to_yuv444_limited_bt709(rgb)
            y, u420, v420 = _yuv444_to_yuv420(yuv444)
            lq_yuv444 = _yuv420_to_yuv444_nn(y, u420, v420)
            _save_packed_png(gt_dir / name, yuv444)
            _save_packed_png(lq_dir / name, lq_yuv444)

    _write_meta(out_root, names)
    if args.with_uvsr_pairs:
        _write_meta(gt_dir, names)
        _write_meta(lq_dir, names)

    print(f"Done. Generated {args.count} samples under {out_root}")


if __name__ == "__main__":
    main()
