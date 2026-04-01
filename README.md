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
- `configs/data_preparation/paired_rgb_to_lmdb.yaml`
- `configs/data_preparation/desktop_yuv420_identity_lmdb.yaml`
- `configs/data_preparation/uvsr_yuv420_to_packed_yuv444_lmdb.yaml`
- `configs/data_preparation/uvsr_test_yuv_to_packed_png.yaml`
- `configs/data_preparation/prepare_data_crop_div2k_1080p.yaml`
- `configs/data_preparation/prepare_data_topaz_full.yaml`

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

## Data Prep

Helper scripts live under `scripts/`:

- `scripts/tools/make_meta_info.py`: build `meta_info.txt` from an image folder
- `scripts/tools/prepare_data.py`: single config-driven entrypoint for operator-based data preparation
- `scripts/maintenance/clean_pycache.bat`: remove `__pycache__` folders and `.pyc` files
- `scripts/generators/`: training-data and preview generators
- `scripts/assets/`: shared fonts and text/color config files

Run data preparation with:

```bash
python scripts/tools/prepare_data.py --config configs/data_preparation/paired_rgb_to_lmdb.yaml
```

`prepare_data.py` uses an operator pipeline config with:

- `jobs`: run one or more preparation jobs in sequence
- `sources`: declare named inputs such as RGB PNG, packed YUV444 PNG, or raw YUV
- `pipeline`: compose reusable ops such as `crop`, `resize`, `rgb_to_yuv444`, `yuv444_to_rgb`, `yuv420_to_yuv444_nn`, `merge_yuv`
- `outputs`: write named frames to files or LMDB

Current example configs include:

- `configs/data_preparation/paired_rgb_to_lmdb.yaml`
- `configs/data_preparation/desktop_yuv420_identity_lmdb.yaml`
- `configs/data_preparation/uvsr_yuv420_to_packed_yuv444_lmdb.yaml`
- `configs/data_preparation/uvsr_test_yuv_to_packed_png.yaml`
- `configs/data_preparation/prepare_data_crop_div2k_1080p.yaml`
- `configs/data_preparation/prepare_data_topaz_full.yaml`
