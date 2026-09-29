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
    ├── train_occscannet_full.pkl                   # full-split PKLs
    ├── val_occscannet_full.pkl
    ├── test_occscannet_full.pkl
    ├── train_subscenes.txt                         # needed only if regenerating PKLs
    ├── val_subscenes.txt                           # needed only if regenerating PKLs
    ├── gathered_data/
    │   └── <scene>/
    │       └── <frame>.pkl
    ├── posed_images/
    │   └── <scene>/
    │       ├── <frame>.jpg
    │       └── <frame>.png                         # default raw uint depth
    ├── gts_camvisbits/
    │   └── <scene>/
    │       └── <frame>/
    │           └── labels.npz
    └── depth_splatssc_stage1_ftdav2_vitb_20m_full/  # optional precomputed depth
        └── <scene>/
            └── <frame>.png
```

The three mini PKLs and the three full-split PKLs are indexes. Their paths are relative to `data/OccScanNet`.

## Mini annotation PKLs

Generate the mini PKLs from prepared OccScanNet split files when needed. AdaOcc follows the ISO OccScanNet-mini reference setup: `iso/config/iso_occscannet_mini.yaml` selects `OccScanNet_mini`, and `iso/scripts/train_iso.py` uses `train_scenes_sample=4639` and `val_scenes_sample=2007`.

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite
```

The script reads `train_subscenes.txt` and `val_subscenes.txt`, builds `train/val/test_occscannet_mini.pkl`, and preserves the camera/lidar coordinate convention used by the baseline. The test split mirrors the mini validation split unless you provide a separate output/source policy.

## Full-split annotation PKLs

`configs/occscannet/radio_occscannet_full.py` uses the full OccScanNet split, i.e. every entry in `train_subscenes.txt` / `val_subscenes.txt`. Generate the full-split PKLs with the same index builder and `--train-count 0 --val-count 0`:

```bash
python scripts/generate_occscannet_mini_pkls.py \
  --data-root data/OccScanNet \
  --train-count 0 --val-count 0 \
  --train-output train_occscannet_full.pkl \
  --val-output val_occscannet_full.pkl \
  --test-output test_occscannet_full.pkl \
  --overwrite
```

The full and mini PKLs point at the same `posed_images/`, `gathered_data/`, and `gts_camvisbits/` trees; only the indexed frame list changes. Because the label and depth generators default to the mini PKLs, pass the full-split PKLs explicitly when preparing a fresh full split:

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py \
  --data-root data/OccScanNet \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl \
  --overwrite

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl
```

The generated label and depth trees are shared with the mini split, so running them for the full split also covers every mini frame.

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

## Depth modes

Default public reproduction uses local/raw same-stem depth PNGs under:

```text
posed_images/<scene>/<frame>.png
```

These are the original raw uint depth files next to each RGB frame. The mini configs default to `ADAOCC_ONLINE_DEPTH=0` and `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`, so the loader derives the raw depth path from each `posed_images/<scene>/<frame>.jpg`. `configs/occscannet/radio_occscannet_full.py` is the exception: it defaults to `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` because the released full-split run used the generated precomputed depth stored in the PKL.

For optional online Depth-Anything prediction, set `ADAOCC_ONLINE_DEPTH=1` for both training and evaluation and provide:

```text
pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
```

For optional generated precomputed-depth training/eval, generate or provide the SPlatSSC-style FT-DaV2 tree:

```text
depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png
```

Generate the tree from mini PKLs and posed images with the same public FT-DaV2 checkpoint used by SPlatSSC:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --verify-only
```

The generation script writes to the `depth_path` stored in each PKL camera record. By default this is `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png`. Existing files are skipped unless `--overwrite` is used. Use `--limit N` for a small smoke run.

These PNGs are binary depth containers. Each pixel stores one `float32` metric-depth value in meters. The file is written by casting the depth map to little-endian float32 (`<f4`) and viewing each 4-byte float as RGBA `uint8` channels (`H x W x 4`). On load, AdaOcc reads the raw RGBA bytes and views them back as `<f4`. Do not convert, resize, color-map, or re-save these PNGs as normal images.

Select generated precomputed depth with `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` when online depth is disabled. This is already the default for `configs/occscannet/radio_occscannet_full.py`, so the released full-split checkpoint evaluates with no depth env prefix.

## Asset check

Verify the default raw-depth data tree with:

```bash
python scripts/check_assets.py --raw-depth-from-images --verify-depth-png
```

The checker reports exact missing pkl/lidar/image/label/depth/pretrain paths. Use `--precomputed-depth --verify-depth-png` for the generated-depth tree, add `--online-depth` when checking the online Depth-Anything checkpoint, and add `--efficientnet-b7` when validating the EfficientNet-B7 config asset at `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth`. Add `--splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl` when checking the full-split PKLs. `--verify-depth-png` decodes a small sample by default (`--max-depth-checks 16`) while still checking every required depth path exists. Use `--max-depth-checks 0` only if you want an exhaustive decode pass.
