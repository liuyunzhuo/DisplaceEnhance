from __future__ import annotations

import re
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_training_data import (
    CHINESE_CHAR_FILE,
    ENGLISH_TIMES_NEW_ROMAN_RATIO,
    FILENAME_TAG,
    FONTS_DIR,
    GridPreset,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOG_FILENAME,
    MS_YAHEI_KEYWORD,
    PNG_COMPRESS_LEVEL,
    TIMES_NEW_ROMAN_FONT_FILENAMES,
    TIMES_NEW_ROMAN_FONT_FILENAMES_LOWER,
    ColorPairConfig,
    build_output_filename,
    build_preference_plan,
    choose_char,
    load_chinese_chars,
    load_font_catalog,
    load_font,
    load_optional_font_paths,
    pick_font_for_sample,
    randomize_color,
    resolve_font_paths,
    write_generation_log,
)


# ========================
# Config
# ========================

SCRIPTS_DIR = SCRIPT_DIR.parent
ASSETS_DIR = SCRIPTS_DIR / "assets"
COLOR_CONFIG_FILE = ASSETS_DIR / "colors" / "问题颜色.txt"
OUTPUT_DIR = Path("samples/problem_colors_1920x1080_v1")
META_INFO_FILENAME = "meta_info.txt"

START_INDEX = 9500
RANDOM_SEED = 42
SAMPLES_PER_COLOR_PAIR = 30

BACKGROUND_JITTER = (6, 6, 6)
TEXT_JITTER = (15, 15, 15)

SAMPLE_PLAN: Sequence[tuple[str, int]] = (
    ("digit", 3),
    ("chinese", 9),
    ("english", 9),
    ("symbol", 4),
    ("mix", 5),
)

COLOR_LINE_PATTERN = re.compile(
    r"^(#[0-9A-Fa-f]{6})\s+rgb=\((\d+),\s*(\d+),\s*(\d+)\)"
)

# Problem-color samples in the current corpus tend to have smaller text with
# slightly tighter spacing, so we bias this generator toward denser grids and
# smaller font sizes than the default training-data presets.
PROBLEM_GRID_PRESETS: dict[str, Sequence[GridPreset]] = {
    "digit": (
        GridPreset(font_size=12, cols=194, rows=72),
        GridPreset(font_size=16, cols=166, rows=62),
        GridPreset(font_size=20, cols=136, rows=51),
        GridPreset(font_size=24, cols=112, rows=42),
    ),
    "chinese": (
        GridPreset(font_size=16, cols=110, rows=58),
        GridPreset(font_size=18, cols=96, rows=50),
        GridPreset(font_size=20, cols=84, rows=44),
        GridPreset(font_size=24, cols=69, rows=36),
    ),
    "english": (
        GridPreset(font_size=12, cols=178, rows=66),
        GridPreset(font_size=16, cols=150, rows=56),
        GridPreset(font_size=20, cols=128, rows=48),
        GridPreset(font_size=24, cols=108, rows=40),
    ),
    "symbol": (
        GridPreset(font_size=12, cols=178, rows=66),
        GridPreset(font_size=16, cols=150, rows=56),
        GridPreset(font_size=20, cols=128, rows=48),
        GridPreset(font_size=24, cols=108, rows=40),
    ),
    "mix": (
        GridPreset(font_size=16, cols=110, rows=58),
        GridPreset(font_size=18, cols=96, rows=50),
        GridPreset(font_size=20, cols=84, rows=44),
        GridPreset(font_size=24, cols=69, rows=36),
    ),
}
PROBLEM_GRID_PRESET_WEIGHTS: Sequence[int] = (5, 3, 2, 1)
PROBLEM_MS_YAHEI_RATIO = 0.75
PROBLEM_SMALL_NON_ENGLISH_MAX_FONT_SIZE = 18
PROBLEM_CLEAR_FONT_KEYWORDS: Sequence[str] = ("微软雅黑", "细黑")
PROBLEM_COL_VARIATION = (-2, 0)
PROBLEM_ROW_VARIATION = (-1, 0)
PROBLEM_CELL_JITTER_RATIO = 0.05


@dataclass(frozen=True)
class ProblemColorEntry:
    label: str
    text_color: tuple[int, int, int]
    background_color: tuple[int, int, int]

    def to_color_pair(self) -> ColorPairConfig:
        return ColorPairConfig(
            name=self.label,
            background_color=self.background_color,
            text_color=self.text_color,
            background_jitter=BACKGROUND_JITTER,
            text_jitter=TEXT_JITTER,
        )


def parse_problem_color_entries(config_path: Path) -> list[ProblemColorEntry]:
    text = config_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    entries: list[ProblemColorEntry] = []

    for block_index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Invalid color block #{block_index}: {block!r}")

        label = lines[0]
        color_matches = []
        for line in lines[1:]:
            match = COLOR_LINE_PATTERN.match(line)
            if match:
                color_matches.append(match)

        if len(color_matches) < 2:
            raise ValueError(f"Missing colors in block #{block_index}: {block!r}")

        text_color = tuple(int(color_matches[0].group(i)) for i in range(2, 5))
        background_color = tuple(int(color_matches[1].group(i)) for i in range(2, 5))
        entries.append(
            ProblemColorEntry(
                label=label,
                text_color=text_color,
                background_color=background_color,
            )
        )

    return entries


def build_meta_info(meta_info_path: Path, names: Sequence[str]) -> None:
    meta_info_path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def choose_problem_grid_preset(char_type: str, rng: random.Random) -> GridPreset:
    presets = tuple(PROBLEM_GRID_PRESETS[char_type])
    if len(presets) != len(PROBLEM_GRID_PRESET_WEIGHTS):
        raise ValueError(
            f"Preset count mismatch for {char_type}: "
            f"{len(presets)} presets vs {len(PROBLEM_GRID_PRESET_WEIGHTS)} weights."
        )
    return rng.choices(presets, weights=PROBLEM_GRID_PRESET_WEIGHTS, k=1)[0]


def build_problem_clear_font_pool(font_catalog) -> tuple[Path, ...]:
    clear_fonts: list[Path] = []
    for font_path in font_catalog.non_times_fonts or font_catalog.all_fonts:
        if any(keyword in font_path.stem for keyword in PROBLEM_CLEAR_FONT_KEYWORDS):
            clear_fonts.append(font_path)
    return tuple(clear_fonts)


def pick_problem_font_for_sample(
    char_type: str,
    preset: GridPreset,
    font_catalog,
    times_new_roman_fonts: Sequence[Path],
    clear_non_english_fonts: Sequence[Path],
    rng: random.Random,
    prefer_ms_yahei: bool,
    prefer_times_new_roman: bool,
) -> Path:
    if char_type == "english":
        return pick_font_for_sample(
            char_type=char_type,
            font_catalog=font_catalog,
            times_new_roman_fonts=times_new_roman_fonts,
            rng=rng,
            prefer_ms_yahei=False,
            prefer_times_new_roman=prefer_times_new_roman,
        )

    if char_type in {"chinese", "mix"} and preset.font_size <= PROBLEM_SMALL_NON_ENGLISH_MAX_FONT_SIZE:
        clear_pool = tuple(clear_non_english_fonts)
        if clear_pool:
            preferred_clear_fonts = tuple(
                font_path for font_path in clear_pool if MS_YAHEI_KEYWORD in font_path.stem
            )
            other_clear_fonts = tuple(
                font_path for font_path in clear_pool if MS_YAHEI_KEYWORD not in font_path.stem
            )
            if prefer_ms_yahei and preferred_clear_fonts:
                return rng.choice(preferred_clear_fonts)
            if other_clear_fonts:
                return rng.choice(other_clear_fonts)
            return rng.choice(clear_pool)

    return pick_font_for_sample(
        char_type=char_type,
        font_catalog=font_catalog,
        times_new_roman_fonts=times_new_roman_fonts,
        rng=rng,
        prefer_ms_yahei=prefer_ms_yahei,
        prefer_times_new_roman=False,
    )


def render_problem_sample(
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

    cols = max(1, preset.cols + rng.randint(*PROBLEM_COL_VARIATION))
    rows = max(1, preset.rows + rng.randint(*PROBLEM_ROW_VARIATION))
    cell_width = width / cols
    cell_height = height / rows
    jitter_x = max(1, int(cell_width * PROBLEM_CELL_JITTER_RATIO))
    jitter_y = max(1, int(cell_height * PROBLEM_CELL_JITTER_RATIO))

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


def generate_problem_color_training_data() -> None:
    rng = random.Random(RANDOM_SEED)
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    entries = parse_problem_color_entries(COLOR_CONFIG_FILE)
    chinese_chars = load_chinese_chars(CHINESE_CHAR_FILE)
    font_catalog = load_font_catalog(FONTS_DIR)
    times_new_roman_fonts = load_optional_font_paths(
        resolve_font_paths(FONTS_DIR, TIMES_NEW_ROMAN_FONT_FILENAMES)
    )
    clear_non_english_fonts = build_problem_clear_font_pool(font_catalog)

    total_per_pair = sum(count for _, count in SAMPLE_PLAN)
    if total_per_pair != SAMPLES_PER_COLOR_PAIR:
        raise ValueError(
            f"SAMPLE_PLAN totals {total_per_pair} samples per color pair, "
            f"expected {SAMPLES_PER_COLOR_PAIR}."
        )

    total_english_samples = sum(
        count for char_type, count in SAMPLE_PLAN if char_type == "english"
    ) * len(entries)
    total_non_english_samples = sum(
        count for char_type, count in SAMPLE_PLAN if char_type != "english"
    ) * len(entries)

    ms_yahei_preference_plan = iter(
        build_preference_plan(
            total_samples=total_non_english_samples,
            preferred_ratio=PROBLEM_MS_YAHEI_RATIO,
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

    sample_idx = START_INDEX
    font_usage: dict[str, int] = {}
    english_times_new_roman_count = 0
    output_names: list[str] = []
    log_lines = [
        f"seed={RANDOM_SEED}",
        f"start_index={START_INDEX}",
        f"image_size={IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        f"output_dir={output_path}",
        f"filename_tag={FILENAME_TAG}",
        f"color_config_file={COLOR_CONFIG_FILE}",
        f"samples_per_color_pair={SAMPLES_PER_COLOR_PAIR}",
        f"color_pair_count={len(entries)}",
        f"sample_plan={list(SAMPLE_PLAN)}",
        f"background_jitter={BACKGROUND_JITTER}",
        f"text_jitter={TEXT_JITTER}",
        f"problem_grid_preset_weights={list(PROBLEM_GRID_PRESET_WEIGHTS)}",
        f"problem_ms_yahei_ratio_non_english={PROBLEM_MS_YAHEI_RATIO}",
        f"problem_small_non_english_max_font_size={PROBLEM_SMALL_NON_ENGLISH_MAX_FONT_SIZE}",
        f"problem_clear_font_keywords={list(PROBLEM_CLEAR_FONT_KEYWORDS)}",
        f"problem_clear_non_english_fonts={[font.name for font in clear_non_english_fonts]}",
        f"problem_col_variation={PROBLEM_COL_VARIATION}",
        f"problem_row_variation={PROBLEM_ROW_VARIATION}",
        f"problem_cell_jitter_ratio={PROBLEM_CELL_JITTER_RATIO}",
        (
            "problem_grid_presets="
            + str(
                {
                    char_type: [
                        {
                            "font_size": preset.font_size,
                            "cols": preset.cols,
                            "rows": preset.rows,
                        }
                        for preset in presets
                    ]
                    for char_type, presets in PROBLEM_GRID_PRESETS.items()
                }
            )
        ),
        f"english_times_new_roman_ratio={ENGLISH_TIMES_NEW_ROMAN_RATIO}",
        f"times_new_roman_fonts={[font.name for font in times_new_roman_fonts]}",
        "samples:",
    ]

    for pair_index, entry in enumerate(entries, start=1):
        color_pair = entry.to_color_pair()
        log_lines.append(
            (
                f"color_pair[{pair_index}] label={entry.label} "
                f"background={entry.background_color} text={entry.text_color}"
            )
        )
        for char_type, count in SAMPLE_PLAN:
            for _ in range(count):
                preset = choose_problem_grid_preset(char_type, rng)
                if char_type == "english":
                    font_path = pick_problem_font_for_sample(
                        char_type=char_type,
                        preset=preset,
                        font_catalog=font_catalog,
                        times_new_roman_fonts=times_new_roman_fonts,
                        clear_non_english_fonts=clear_non_english_fonts,
                        rng=rng,
                        prefer_ms_yahei=False,
                        prefer_times_new_roman=next(english_times_preference_plan),
                    )
                else:
                    font_path = pick_problem_font_for_sample(
                        char_type=char_type,
                        preset=preset,
                        font_catalog=font_catalog,
                        times_new_roman_fonts=times_new_roman_fonts,
                        clear_non_english_fonts=clear_non_english_fonts,
                        rng=rng,
                        prefer_ms_yahei=next(ms_yahei_preference_plan),
                        prefer_times_new_roman=False,
                    )

                image, background_color, text_color, cols, rows = render_problem_sample(
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    char_type=char_type,
                    preset=preset,
                    color_pair=color_pair,
                    font_path=font_path,
                    chinese_chars=chinese_chars,
                    rng=rng,
                )
                filename = build_output_filename(sample_idx, IMAGE_WIDTH, IMAGE_HEIGHT)
                image.save(
                    output_path / filename,
                    format="PNG",
                    compress_level=PNG_COMPRESS_LEVEL,
                    optimize=True,
                )

                output_names.append(filename)
                font_usage[font_path.name] = font_usage.get(font_path.name, 0) + 1
                if char_type == "english" and font_path.name.lower() in TIMES_NEW_ROMAN_FONT_FILENAMES_LOWER:
                    english_times_new_roman_count += 1
                log_lines.append(
                    " | ".join(
                        [
                            f"filename={filename}",
                            f"pair_label={entry.label}",
                            f"pair_index={pair_index}",
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
                print(
                    f"generated {filename} | pair={entry.label} | "
                    f"font={font_path.name} | bg={background_color} | text={text_color}"
                )
                sample_idx += 1

    build_meta_info(output_path / META_INFO_FILENAME, output_names)
    write_generation_log(output_path / LOG_FILENAME, log_lines)

    total_images = len(output_names)
    yahei_images = sum(
        count for name, count in font_usage.items() if MS_YAHEI_KEYWORD in name
    )
    yahei_ratio = yahei_images / total_images if total_images else 0.0
    english_ratio = (
        english_times_new_roman_count / total_english_samples if total_english_samples else 0.0
    )
    print(
        f"finished {total_images} RGB PNG files in {output_path}. "
        f"Microsoft YaHei usage: {yahei_images}/{total_images} ({yahei_ratio:.1%}). "
        f"Times New Roman usage in english samples: {english_times_new_roman_count}/{total_english_samples} "
        f"({english_ratio:.1%})."
    )


if __name__ == "__main__":
    generate_problem_color_training_data()
