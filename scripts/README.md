# Scripts Layout

## Folders

- `assets/`
  Shared non-code resources.
- `assets/chars/`
  Character lists such as `3500常用汉字.txt`.
- `assets/colors/`
  Color reference files such as `常用颜色.txt` and `问题颜色.txt`.
- `assets/fonts/`
  Project-local font files used by the generators.
- `generators/`
  Synthetic data generation scripts.
- `analysis/`
  Analysis helpers such as color extraction.
- `tools/`
  Dataset preparation and `meta_info` utilities.
- `maintenance/`
  Cleanup helpers.

## Common Commands

```bash
python scripts/generators/generate_training_data.py
python scripts/generators/generate_problem_color_training_data.py
python scripts/generators/generate_uvsr_text_patterns.py
python scripts/analysis/extract_common_colors.py --img_dir samples/problem_colors_1920x1080_v1
python scripts/tools/make_meta_info.py --img_dir samples/problem_colors_1920x1080_v1
python scripts/tools/prepare_lmdb_from_config.py --config configs/data_prep/paired_rgb_to_lmdb.yaml
```
