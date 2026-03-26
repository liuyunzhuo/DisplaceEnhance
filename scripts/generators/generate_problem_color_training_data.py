from __future__ import annotations

import re
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_training_data import (
    CHINESE_CHAR_FILE,
    ENGLISH_TIMES_NEW_ROMAN_RATIO,
    FILENAME_TAG,
    FONTS_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOG_FILENAME,
    MS_YAHEI_KEYWORD,
    MS_YAHEI_RATIO,
    PNG_COMPRESS_LEVEL,
    TIMES_NEW_ROMAN_FONT_FILENAMES,
    TIMES_NEW_ROMAN_FONT_FILENAMES_LOWER,
    ColorPairConfig,
    build_output_filename,
    build_preference_plan,
    choose_grid_preset,
    load_chinese_chars,
    load_font_catalog,
    load_optional_font_paths,
    pick_font_for_sample,
    render_sample,
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
        f"ms_yahei_ratio_non_english={MS_YAHEI_RATIO}",
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
