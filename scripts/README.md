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
python scripts/tools/prepare_data.py --config configs/data_preparation/prepare_data_crop_div2k_1080p.yaml
python scripts/tools/prepare_data.py --config configs/data_preparation/prepare_data_topaz_full.yaml
python scripts/tools/prepare_data.py --config configs/data_preparation/paired_rgb_to_lmdb.yaml
python scripts/tools/prepare_data.py --config configs/data_preparation/desktop_yuv420_identity_lmdb.yaml
python scripts/tools/prepare_data.py --config configs/data_preparation/uvsr_yuv420_to_packed_yuv444_lmdb.yaml
python scripts/tools/prepare_data.py --config configs/data_preparation/uvsr_test_yuv_to_packed_png.yaml
python scripts/generators/generate_training_data.py
python scripts/generators/generate_problem_color_training_data.py
python scripts/generators/generate_uvsr_text_patterns.py
python scripts/analysis/extract_common_colors.py --img_dir samples/problem_colors_1920x1080_v1
python scripts/tools/make_meta_info.py --img_dir samples/problem_colors_1920x1080_v1
```

`prepare_data.py` now uses an operator pipeline config:

- `jobs`: run multiple preparation jobs in sequence
- `sources`: declare one or more named inputs
- `pipeline`: compose reusable ops such as `crop` (supports `small_image: pad`), `rotate_if_portrait`, `resize`, `resize_if_ratio_close` (supports `allow_upscale: false`), `rgb_to_yuv444`, `yuv444_to_rgb`, `yuv420_to_yuv444_nn`, `merge_yuv`
- `outputs`: write named frames to files or LMDB
