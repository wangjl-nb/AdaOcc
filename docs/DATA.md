# Data preparation

AdaOcc expects OccScanNet files under an ignored runtime root, normally `data/OccScanNet`. The repository does not include any data, generated labels, depth PNGs, or PKL annotations.

## Mini annotation PKLs

Generate the mini PKLs from prepared OccScanNet split files when needed. The common OccScanNet-mini setup follows the prepared split-file order: train first 4639 entries, val/test first 2007 entries. This matches the reference AdaOcc mini PKLs used for the reported numbers.

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

Default public reproduction uses online DepthAnything (`ADAOCC_ONLINE_DEPTH=1`), so precomputed depth PNGs are not required.

For optional precomputed-depth training/eval, generate or provide:

```text
depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png
```

These PNGs store `float32` depth as little-endian RGBA bytes. Verify with:

```bash
python scripts/check_assets.py --data-root data/OccScanNet --pretrain-root pretrain --precomputed-depth --verify-depth-png
```

## Asset check

```bash
python scripts/check_assets.py --data-root data/OccScanNet --pretrain-root pretrain --online-depth
```

The checker reports exact missing pkl/lidar/image/label/depth/pretrain paths.

## Final expected layout

After generating PKLs/labels and linking weights, the relevant data tree should be:

```text
data/OccScanNet/
├── train_occscannet_mini.pkl
├── val_occscannet_mini.pkl
├── test_occscannet_mini.pkl
├── gathered_data/<scene>/<frame>.pkl
├── posed_images/<scene>/<frame>.jpg
├── gts_camvisbits/<scene>/<frame>/labels.npz
└── depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png  # optional precomputed depth
```

The three PKLs are indexes. Their paths are relative to `ADAOCC_DATA_ROOT` from `configs/local_paths.sh`.

`--verify-depth-png` decodes a small sample by default (`--max-depth-checks 16`) while still checking every pkl-referenced depth path exists. Use `--max-depth-checks 0` only if you want an exhaustive OpenCV decode pass.
