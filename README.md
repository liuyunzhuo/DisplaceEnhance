# DisplaceEnhance

DisplaceEnhance is a lightweight image enhancement training project built around a configurable paired-image pipeline and a luminance-focused sharpening network.

## Current Layout

```text
DisplaceEnhance/
|- train.py                  # Generic training entrypoint
|- configs/                  # Experiment configs
|- scripts/                  # Dataset helper scripts
`- src/
   |- training/              # Active training framework
   |- models/                # Network definitions
   `- utils/                 # Logging, metrics, reproducibility
```

## Training

Run training with:

```bash
python train.py --opt configs/sharpen_paired.yaml
```

Available example configs:

- `configs/sharpen_paired.yaml`
- `configs/sharpen_gt_only.yaml`
- `configs/sharpen_lq_only.yaml`
- `configs/sharpen_paired_aug.yaml`
- `configs/sharpen_paired_lmdb.yaml`
- `configs/sharpen_paired_lmdb_desktop_lq_only_mix.yaml`
- `configs/data_prep/paired_rgb_to_lmdb.yaml`
- `configs/data_prep/desktop_yuv420_identity_lmdb.yaml`
- `configs/data_prep/uvsr_yuv420_to_packed_yuv444_lmdb.yaml`
- `configs/data_prep/uvsr_test_yuv_to_packed_png.yaml`

## Dataset Pipeline

Each dataset config under `datasets.train` or `datasets.val` defines a `pipeline` list.

Supported steps:

1. `ops`
   Generates or modifies paired samples.
   - `lq_op`: operation applied to the input image
   - `gt_op`: operation applied to the target image

2. `resize`
   Resize both images to a fixed square size.
   - `size`: integer

3. `crop`
   Crop either or both images to a fixed square size.
   - `size`: integer
   - `random`: `true` or `false`

4. `to_tensor`
   Convert PIL images to PyTorch tensors.

5. `augment`
   Tensor-space flip and rotation augmentation.
   - `flip`: `true` or `false`
   - `rot`: `true` or `false`

Order rules:

- `resize` and `crop` must run before `to_tensor`
- `augment` must run after `to_tensor`
- `ops` must run before `to_tensor`
- For `gt_only` and `lq_only`, `crop` can run before `ops`

If the order is invalid, the dataset will raise an error.

## Data Modes

- `paired`: input and target images already exist as aligned pairs
- `paired_aug`: input and target images exist, but each side can be modified by `ops`
- `gt_only`: only target images exist, input images are generated on the fly
- `lq_only`: only input images exist, target images are generated on the fly

## Model Wrapper

The training framework now uses a generic single-network wrapper:

- `model_type: "ImageRestorationModel"` for the main path
- `network.type` selects the concrete network class
- `train.loss.mode` controls whether loss is computed on all channels or on RGB luma only

To add another network such as UVSR:

- place the network module under `src/models/`
- register it with `NETWORK_REGISTRY`
- point `network.type` at that registered class name

For sharpen-style RGB training, use:

```yaml
train:
  loss:
    type: "L1"
    mode: "bt601_luma"
```

For UVSR-style packed YUV training, use:

```yaml
train:
  loss:
    type: "L1"
    mode: "all_channels"
```

## Mixing Train Datasets

`datasets.train` can now be either a single dataset config or a loader config with a `datasets` list.

Example:

```yaml
datasets:
  train:
    batch_size_per_gpu: 8
    num_worker_per_gpu: 4
    datasets:
      - type: "PairedLmdbDataset"
        dataroot_lq: "data/train/lq.lmdb"
        dataroot_gt: "data/train/gt.lmdb"
        repeat: 1
      - type: "PairedImageDataset"
        dataroot_lq: "data/train/desktop_identity"
        dataroot_gt: "data/train/desktop_identity"
        repeat: 2
```

Notes:

- `repeat` can be used to upweight a smaller dataset.
- All mixed train datasets should output the same tensor shape after their pipelines.
- For identity-style desktop data, you can point `dataroot_lq` and `dataroot_gt` to the same folder so the model sees `LQ == GT`.
- `meta_info` can live inside the project, for example `datasets/train/bvi_dvc/gt/meta_info.txt`, which is useful when you want fine-grained control over which files participate in training or validation.

## LMDB Tools

Helper scripts live under `scripts/`:

- `scripts/tools/make_meta_info.py`: build `meta_info.txt` from an image folder
- `scripts/tools/prepare_lmdb_from_config.py`: config-driven LMDB prep for paired or single-source data
- `scripts/maintenance/clean_pycache.bat`: remove `__pycache__` folders and `.pyc` files
- `scripts/generators/`: training-data and preview generators
- `scripts/assets/`: shared fonts and text/color config files

Config-driven LMDB prep example:

```bash
python scripts/tools/prepare_lmdb_from_config.py --config configs/data_prep/paired_rgb_to_lmdb.yaml
```

The config-driven prep tool supports:

- `dataset.mode: paired` for normal LQ/GT datasets
- `dataset.mode: single` for one-sided data, such as desktop scenes where only LQ is available
- `yuv420_to_yuv444_nn`
- `yuv444_to_rgb`
- `rgb_to_yuv444`
- `resize`
- patch generation via `patching.mode: none | grid | grid_full | random`
- raw YUV width, height, and pixel format can be inferred from filenames

For `yuv444_to_rgb` and `rgb_to_yuv444`, `matrix` can be `bt601`, `bt709`, or `internal_product`.
`internal_product` uses the project-specific full-range product conversion formula and ignores `range`.

`grid_full` works like grid cropping, but also adds the last row and last column patches so edge areas are not dropped when the image size is not divisible by stride.
It does not pad or fill missing pixels with zeros. Instead, it moves the final patch start position back to `width - size` or `height - size`, so the border area is covered by overlapping crops taken entirely from the original image.

The prep tool can also store packed YUV444 PNGs for UVSR-style training:

- `channel_packing: "rgb"` means PNG channels are true RGB
- `channel_packing: "yuv444"` means PNG channels store `Y`, `U`, `V` directly in `R`, `G`, `B`

To keep the config explicit, output storage is described with an output pipeline.
Typical output steps are:

- `store_rgb_png`
- `store_packed_yuv444_png`
- `write_lmdb`
- `write_files`

Example:

```yaml
packing:
  meta_info_out:
    lq: "datasets/train/example/lq/meta_info.txt"
  outputs:
    lq:
      pipeline:
        - name: "store_packed_yuv444_png"
          params:
            png_compress_level: 3
        - name: "write_lmdb"
          params:
            lmdb_name: "lq.lmdb"
            map_size_gb: 0.25
            write_batch_size: 512
            compact: true
```

This makes the intent clearer:

- preprocessing pipeline decides what the sample becomes
- output pipeline decides how that sample is stored
- `store_packed_yuv444_png` does not convert RGB to YUV. It simply stores the current YUV444 planes into PNG `R/G/B` channels as `Y/U/V`.
- LMDB-only options such as `map_size_gb`, `write_batch_size`, and `compact` belong inside `write_lmdb.params`.
- `packing.meta_info_out` is optional and lets you write extra project-side `meta_info.txt` files for training-time dataset curation while still keeping a local `meta_info.txt` inside each output folder.

For example:

- the sample is stored as a PNG
- channel 0 is `Y`
- channel 1 is `U`
- channel 2 is `V`

If you want to keep the original source filename and append a suffix such as `_yuv444png`, use:

```yaml
- name: "write_files"
  params:
    dir_name: "lq"
    filename_mode: "source_stem"
    filename_suffix: "_yuv444png"
    append_patch_suffix: true
```

This turns an input like `0013_TE_1920x1080_I444p.yuv` into `0013_TE_1920x1080_I444p_yuv444png.png`.
If one source image produces multiple patches, enabling `append_patch_suffix: true` will produce names such as `_s001`, `_s002`, which makes it easier to trace cropped outputs back to the original source sample.

If you want validation images to be converted from packed YUV444 back to RGB before saving, configure it explicitly in the train config:

```yaml
val:
  save_img: true
  visualization:
    pipeline:
      - name: "packed_yuv444_to_rgb"
        params:
          matrix: "bt709"
          value_range: "limited"
```

If your packed YUV444 data uses the internal product conversion instead of standard BT.601 / BT.709, use:

```yaml
      - name: "packed_yuv444_to_rgb"
        params:
          matrix: "internal_product"
```

If no visualization pipeline is provided, validation image saving keeps the packed channel layout instead of guessing how to convert it back to RGB.

For one-sided identity data, you can save only `lq.lmdb` and use `mode: lq_only` during training. The example config `configs/sharpen_paired_lmdb_desktop_lq_only_mix.yaml` uses an `identity` target op so `gt` is copied from `lq` on the fly.

For identity-style desktop training, you can point both output branches at the same single source by defining both `dataset.branches.lq` and `dataset.branches.gt` from one input stream.
