from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GridPreset:
    font_size: int
    cols: int
    rows: int


@dataclass(frozen=True)
class ColorPairConfig:
    name: str
    background_color: tuple[int, int, int]
    text_color: tuple[int, int, int]
    background_jitter: tuple[int, int, int] = (0, 0, 0)
    text_jitter: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class FontCatalog:
    all_fonts: tuple[Path, ...]
    non_times_fonts: tuple[Path, ...]
    preferred_fonts: tuple[Path, ...]
    other_fonts: tuple[Path, ...]

    def pick(self, rng: random.Random, prefer_ms_yahei: bool | None = None) -> Path:
        if prefer_ms_yahei is True and self.preferred_fonts:
            return rng.choice(self.preferred_fonts)
        if prefer_ms_yahei is False and self.other_fonts:
            return rng.choice(self.other_fonts)
        if self.preferred_fonts and rng.random() < MS_YAHEI_RATIO:
            return rng.choice(self.preferred_fonts)
        pool = self.other_fonts or self.non_times_fonts or self.all_fonts
        return rng.choice(pool)


# ========================
# Config
# ========================

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
ASSETS_DIR = SCRIPTS_DIR / "assets"
CHINESE_CHAR_FILE = ASSETS_DIR / "chars" / "3500常用汉字.txt"
FONTS_DIR = ASSETS_DIR / "fonts"
OUTPUT_DIR = Path("output/png")
FILENAME_TAG = "TR"
LOG_FILENAME = "generation_log.txt"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
START_INDEX = 9200
RANDOM_SEED = 42
PNG_COMPRESS_LEVEL = 3

DEFAULT_CHINESE_CHARS = "测试显示增强蓝黑数据生成"
FONT_EXTENSIONS = {".ttf", ".ttc", ".otf"}
MS_YAHEI_KEYWORD = "微软雅黑"
MS_YAHEI_RATIO = 0.40
ENGLISH_TIMES_NEW_ROMAN_RATIO = 0.60
TIMES_NEW_ROMAN_FONT_FILENAMES: Sequence[str] = (
    "times.ttf",
    "timesbd.ttf",
    "timesi.ttf",
    "timesbi.ttf",
)
TIMES_NEW_ROMAN_FONT_FILENAMES_LOWER = {name.lower() for name in TIMES_NEW_ROMAN_FONT_FILENAMES}

DIGITS = "0123456789"
ENGLISH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_CHARS = ".,;:!?\"'()[]{}<>-_=+*/&^%$#@~`|\\/"
MIX_WEIGHTS = {
    "chinese": 10,
    "english": 6,
    "digit": 3,
    "symbol": 1,
}

SAMPLE_PLAN: Sequence[tuple[str, int]] = (
    ("digit", 3),
    ("chinese", 8),
    ("english", 8),
    ("symbol", 4),
    ("mix", 4),
)

GRID_PRESETS: dict[str, Sequence[GridPreset]] = {
    "digit": (
        GridPreset(font_size=20, cols=118, rows=45),
        GridPreset(font_size=40, cols=80, rows=30),
        GridPreset(font_size=60, cols=60, rows=23),
    ),
    "chinese": (
        GridPreset(font_size=20, cols=73, rows=38),
        GridPreset(font_size=40, cols=53, rows=23),
        GridPreset(font_size=60, cols=32, rows=18),
    ),
    "english": (
        GridPreset(font_size=20, cols=113, rows=42),
        GridPreset(font_size=40, cols=70, rows=28),
        GridPreset(font_size=60, cols=55, rows=21),
    ),
    "symbol": (
        GridPreset(font_size=20, cols=113, rows=42),
        GridPreset(font_size=40, cols=70, rows=28),
        GridPreset(font_size=60, cols=55, rows=21),
    ),
    "mix": (
        GridPreset(font_size=20, cols=73, rows=38),
        GridPreset(font_size=40, cols=53, rows=23),
        GridPreset(font_size=60, cols=32, rows=18),
    ),
}

COLOR_PAIR = ColorPairConfig(
    name="black_blue",
    background_color=(0, 0, 0),
    text_color=(17, 6, 165),
    background_jitter=(6, 6, 6),
    text_jitter=(15, 15, 15),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RGB PNG training data.")
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH, help="Image width.")
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT, help="Image height.")
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help="Output folder for RGB PNG images.",
    )
    parser.add_argument("--start_index", type=int, default=START_INDEX, help="Starting sample index.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


def load_chinese_chars(chinese_file: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            lines = chinese_file.read_text(encoding=encoding).splitlines()
            chars = [line.strip()[0] for line in lines if line.strip()]
            unique_chars = "".join(dict.fromkeys(chars))
            if unique_chars:
                return unique_chars
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    return DEFAULT_CHINESE_CHARS


def load_font_catalog(fonts_dir: Path) -> FontCatalog:
    font_paths: list[Path] = []
    non_times_fonts: list[Path] = []
    preferred_fonts: list[Path] = []

    if not fonts_dir.exists():
        raise FileNotFoundError(f"Font directory not found: {fonts_dir}")

    for font_path in sorted(fonts_dir.iterdir()):
        if font_path.suffix.lower() not in FONT_EXTENSIONS:
            continue
        try:
            ImageFont.truetype(str(font_path), size=32)
        except OSError:
            continue
        font_paths.append(font_path)
        if font_path.name.lower() not in TIMES_NEW_ROMAN_FONT_FILENAMES_LOWER:
            non_times_fonts.append(font_path)
        if MS_YAHEI_KEYWORD in font_path.stem:
            preferred_fonts.append(font_path)

    if not font_paths:
        raise RuntimeError(f"No usable Chinese fonts found under: {fonts_dir}")

    preferred_set = set(preferred_fonts)
    other_fonts = [font for font in non_times_fonts if font not in preferred_set]
    return FontCatalog(
        all_fonts=tuple(font_paths),
        non_times_fonts=tuple(non_times_fonts),
        preferred_fonts=tuple(preferred_fonts),
        other_fonts=tuple(other_fonts),
    )


def load_optional_font_paths(font_paths: Sequence[Path]) -> tuple[Path, ...]:
    usable_paths: list[Path] = []
    for font_path in font_paths:
        if not font_path.exists():
            continue
        try:
            ImageFont.truetype(str(font_path), size=32)
        except OSError:
            continue
        usable_paths.append(font_path)
    return tuple(usable_paths)


def resolve_font_paths(fonts_dir: Path, font_filenames: Sequence[str]) -> tuple[Path, ...]:
    return tuple(fonts_dir / filename for filename in font_filenames)


def randomize_color(
    base_color: tuple[int, int, int],
    spread: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    return tuple(
        max(0, min(255, channel + rng.randint(-delta, delta)))
        for channel, delta in zip(base_color, spread)
    )


def choose_char(char_type: str, chinese_chars: str, rng: random.Random) -> str:
    if char_type == "digit":
        return rng.choice(DIGITS)
    if char_type == "chinese":
        return rng.choice(chinese_chars)
    if char_type == "english":
        return rng.choice(ENGLISH_CHARS)
    if char_type == "symbol":
        return rng.choice(SYMBOL_CHARS)
    if char_type == "mix":
        bucket = rng.choices(tuple(MIX_WEIGHTS.keys()), weights=tuple(MIX_WEIGHTS.values()), k=1)[0]
        return choose_char(bucket, chinese_chars, rng)
    raise ValueError(f"Unsupported char type: {char_type}")


def choose_grid_preset(char_type: str, rng: random.Random) -> GridPreset:
    presets = GRID_PRESETS[char_type]
    return rng.choice(tuple(presets))


def load_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(font_path), size=font_size)
    except OSError:
        return ImageFont.load_default()


def build_preference_plan(
    total_samples: int,
    preferred_ratio: float,
    has_preferred_fonts: bool,
    has_other_fonts: bool,
    rng: random.Random,
) -> list[bool]:
    if not has_preferred_fonts:
        return [False] * total_samples
    if not has_other_fonts:
        return [True] * total_samples

    preferred_count = round(total_samples * preferred_ratio)
    preferred_flags = [True] * preferred_count + [False] * max(0, total_samples - preferred_count)
    rng.shuffle(preferred_flags)
    return preferred_flags


def pick_font_for_sample(
    char_type: str,
    font_catalog: FontCatalog,
    times_new_roman_fonts: Sequence[Path],
    rng: random.Random,
    prefer_ms_yahei: bool,
    prefer_times_new_roman: bool,
) -> Path:
    if char_type == "english" and times_new_roman_fonts and prefer_times_new_roman:
        return rng.choice(tuple(times_new_roman_fonts))
    if char_type == "english":
        pool = font_catalog.non_times_fonts or font_catalog.all_fonts
        return rng.choice(pool)
    return font_catalog.pick(rng, prefer_ms_yahei=prefer_ms_yahei)


def render_sample(
    width: int,
    height: int,
    char_type: str,
    preset: GridPreset,
    color_pair: ColorPairConfig,
    font_path: Path,
    chinese_chars: str,
    rng: random.Random,
) -> tuple[Image.Image, tuple[int, int, int], tuple[int, int, int], int, int]:
    background_color = randomize_color(color_pair.background_color, color_pair.background_jitter, rng)
    text_color = randomize_color(color_pair.text_color, color_pair.text_jitter, rng)

    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    font = load_font(font_path, preset.font_size)

    cols = max(1, preset.cols + rng.randint(-5, 0))
    rows = max(1, preset.rows + rng.randint(-2, 0))
    cell_width = width / cols
    cell_height = height / rows
    jitter_x = max(1, int(cell_width * 0.08))
    jitter_y = max(1, int(cell_height * 0.08))

    for row in range(rows):
        for col in range(cols):
            text = choose_char(char_type, chinese_chars, rng)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            cell_x = int(col * cell_width)
            cell_y = int(row * cell_height)
            cell_w = max(1, int((col + 1) * cell_width) - cell_x)
            cell_h = max(1, int((row + 1) * cell_height) - cell_y)

            text_x = cell_x + (cell_w - text_width) // 2 + rng.randint(-jitter_x, jitter_x)
            text_y = cell_y + (cell_h - text_height) // 2 + rng.randint(-jitter_y, jitter_y)
            text_x = min(max(text_x, 0), max(0, width - text_width))
            text_y = min(max(text_y, 0), max(0, height - text_height))
            draw.text((text_x, text_y), text, fill=text_color, font=font)

    return image, background_color, text_color, cols, rows


def build_output_filename(sample_idx: int, width: int, height: int) -> str:
    return f"{sample_idx}_{FILENAME_TAG}_{width}x{height}.png"


def write_generation_log(log_path: Path, lines: Sequence[str]) -> None:
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_training_data(
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    output_dir: str | Path = OUTPUT_DIR,
    start_index: int = START_INDEX,
    seed: int = RANDOM_SEED,
) -> None:
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / LOG_FILENAME

    chinese_chars = load_chinese_chars(CHINESE_CHAR_FILE)
    font_catalog = load_font_catalog(FONTS_DIR)
    times_new_roman_fonts = load_optional_font_paths(
        resolve_font_paths(FONTS_DIR, TIMES_NEW_ROMAN_FONT_FILENAMES)
    )
    font_usage: Counter[str] = Counter()
    total_english_samples = sum(count for char_type, count in SAMPLE_PLAN if char_type == "english")
    total_non_english_samples = sum(count for char_type, count in SAMPLE_PLAN if char_type != "english")
    ms_yahei_preference_plan = iter(
        build_preference_plan(
            total_samples=total_non_english_samples,
            preferred_ratio=MS_YAHEI_RATIO,
            has_preferred_fonts=bool(font_catalog.preferred_fonts),
            has_other_fonts=bool(font_catalog.other_fonts),
            rng=rng,
        )
    )
    english_times_preference_plan = iter(
        build_preference_plan(
            total_samples=total_english_samples,
            preferred_ratio=ENGLISH_TIMES_NEW_ROMAN_RATIO,
            has_preferred_fonts=bool(times_new_roman_fonts),
            has_other_fonts=bool(font_catalog.all_fonts),
            rng=rng,
        )
    )
    log_lines = [
        f"seed={seed}",
        f"start_index={start_index}",
        f"image_size={width}x{height}",
        f"output_dir={output_path}",
        f"filename_tag={FILENAME_TAG}",
        f"ms_yahei_ratio_non_english={MS_YAHEI_RATIO}",
        f"english_times_new_roman_ratio={ENGLISH_TIMES_NEW_ROMAN_RATIO}",
        f"times_new_roman_fonts={[font.name for font in times_new_roman_fonts]}",
        (
            "color_pair="
            f"name:{COLOR_PAIR.name},"
            f"bg_base:{COLOR_PAIR.background_color},"
            f"text_base:{COLOR_PAIR.text_color},"
            f"bg_jitter:{COLOR_PAIR.background_jitter},"
            f"text_jitter:{COLOR_PAIR.text_jitter}"
        ),
        "samples:",
    ]

    sample_idx = start_index
    for char_type, count in SAMPLE_PLAN:
        for _ in range(count):
            preset = choose_grid_preset(char_type, rng)
            if char_type == "english":
                font_path = pick_font_for_sample(
                    char_type=char_type,
                    font_catalog=font_catalog,
                    times_new_roman_fonts=times_new_roman_fonts,
                    rng=rng,
                    prefer_ms_yahei=False,
                    prefer_times_new_roman=next(english_times_preference_plan),
                )
            else:
                font_path = pick_font_for_sample(
                    char_type=char_type,
                    font_catalog=font_catalog,
                    times_new_roman_fonts=times_new_roman_fonts,
                    rng=rng,
                    prefer_ms_yahei=next(ms_yahei_preference_plan),
                    prefer_times_new_roman=False,
                )
            image, background_color, text_color, cols, rows = render_sample(
                width=width,
                height=height,
                char_type=char_type,
                preset=preset,
                color_pair=COLOR_PAIR,
                font_path=font_path,
                chinese_chars=chinese_chars,
                rng=rng,
            )
            filename = build_output_filename(sample_idx, width, height)
            image.save(
                output_path / filename,
                format="PNG",
                compress_level=PNG_COMPRESS_LEVEL,
                optimize=True,
            )
            log_lines.append(
                " | ".join(
                    [
                        f"filename={filename}",
                        f"char_type={char_type}",
                        f"font={font_path.name}",
                        f"font_size={preset.font_size}",
                        f"cols={cols}",
                        f"rows={rows}",
                        f"background={background_color}",
                        f"text={text_color}",
                    ]
                )
            )
            font_usage[font_path.name] += 1
            print(
                f"generated {filename} | font={font_path.name} | "
                f"bg={background_color} | text={text_color}"
            )
            sample_idx += 1

    write_generation_log(log_path, log_lines)
    total_images = sum(font_usage.values())
    yahei_images = sum(count for name, count in font_usage.items() if MS_YAHEI_KEYWORD in name)
    times_new_roman_images = sum(
        count for name, count in font_usage.items() if name.lower().startswith("times")
    )
    ratio = yahei_images / total_images if total_images else 0.0
    english_ratio = times_new_roman_images / total_english_samples if total_english_samples else 0.0
    print(
        f"finished {total_images} RGB PNG files in {output_path}. "
        f"Microsoft YaHei usage: {yahei_images}/{total_images} ({ratio:.1%}). "
        f"Times New Roman usage in english samples: {times_new_roman_images}/{total_english_samples} "
        f"({english_ratio:.1%}). "
        f"log saved to {log_path}"
    )


def main() -> None:
    args = parse_args()
    generate_training_data(
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
        start_index=args.start_index,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
