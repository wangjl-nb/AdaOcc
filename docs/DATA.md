# Data preparation

AdaOcc uses a fixed repository-relative runtime root: `data/OccScanNet`. The repository does not include data, generated labels, depth PNGs, or PKL annotations.

## Expected layout

Arrange local files under this ignored tree:

```text
data/
└── OccScanNet/
    ├── train_occscannet_mini.pkl
    ├── val_occscannet_mini.pkl
    ├── test_occscannet_mini.pkl
    ├── train_subscenes.txt                         # needed only if regenerating PKLs
    ├── val_subscenes.txt                           # needed only if regenerating PKLs
    ├── gathered_data/
    │   └── <scene>/
    │       └── <frame>.pkl
    ├── posed_images/
    │   └── <scene>/
    │       └── <frame>.jpg
    ├── gts_camvisbits/
    │   └── <scene>/
    │       └── <frame>/
    │           └── labels.npz
    └── depth_splatssc_stage1_ftdav2_vitb_20m_full/  # optional precomputed depth
        └── <scene>/
            └── <frame>.png
```

The three PKLs are indexes. Their paths are relative to `data/OccScanNet`.

## Mini annotation PKLs

Generate the mini PKLs from prepared OccScanNet split files when needed. AdaOcc follows the ISO OccScanNet-mini reference setup: `iso/config/iso_occscannet_mini.yaml` selects `OccScanNet_mini`, and `iso/scripts/train_iso.py` uses `train_scenes_sample=4639` and `val_scenes_sample=2007`.

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite
```

The script reads `train_subscenes.txt` and `val_subscenes.txt`, builds `train/val/test_occscannet_mini.pkl`, and preserves the camera/lidar coordinate convention used by the baseline. The test split mirrors the mini validation split unless you provide a separate output/source policy.

## Labels

The label path is:

```text
gts_camvisbits/{token}/labels.npz
```

where `token` is already `<scene>/<frame>`. Do **not** construct `gts_camvisbits/{scene}/{token}/labels.npz`.

Required `labels.npz` keys:

- `semantics`
- `mask_lidar`
- `mask_camera`
- `raw_semantics`
- `voxel_origin`
- `voxel_size`

Generate/verify labels:

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --verify-only
```

`raw_semantics` preserves OccScanNet's native `[y, x, z]` indexing. AdaOcc converts raw coordinates with `coords[:, [1, 0, 2]]`; do not transpose `raw_semantics` during generation.

## Online vs precomputed depth

Default public reproduction uses online DepthAnything (`ADAOCC_ONLINE_DEPTH=1`, the default), so precomputed depth PNGs are not required.

For optional precomputed-depth training/eval, generate or provide:

```text
depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png
```

Generate the tree from mini PKLs and posed images with:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --verify-only
```

The generation script writes to the `depth_path` stored in each mini pkl camera record. By default this is `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png`. Existing files are skipped unless `--overwrite` is used. Use `--limit N` for a small smoke run.

These PNGs are binary depth containers. Each pixel stores one `float32` metric-depth value in meters. The file is written by casting the depth map to little-endian float32 (`<f4`) and viewing each 4-byte float as RGBA `uint8` channels (`H x W x 4`). On load, AdaOcc reads the raw RGBA bytes and views them back as `<f4`. Do not convert, resize, color-map, or re-save these PNGs as normal images.

Verify the final data tree with:

```bash
python scripts/check_assets.py --precomputed-depth --verify-depth-png
```

Use precomputed-depth mode by setting `ADAOCC_ONLINE_DEPTH=0` for both training and evaluation.

## Asset check

```bash
python scripts/check_assets.py --online-depth
```

The checker reports exact missing pkl/lidar/image/label/depth/pretrain paths. `--verify-depth-png` decodes a small sample by default (`--max-depth-checks 16`) while still checking every pkl-referenced depth path exists. Use `--max-depth-checks 0` only if you want an exhaustive decode pass.
