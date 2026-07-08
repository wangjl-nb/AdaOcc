# Reproducibility

## Config choices

AdaOcc public reproduction is config-selected. Choose one full config and its matching smoke config before running commands:

| choice | full config | smoke config | notes |
| --- | --- | --- | --- |
| RADIO released/reference baseline | `configs/occscannet/radio_occscannet_mini.py` | `configs/occscannet/radio_occscannet_mini_smoke.py` | Use this for the released checkpoint/metrics. |
| EfficientNet-B7 additional image encoder option | `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py` | Requires `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth`; no released metrics are claimed here. |

Example shell selection:

```bash
CONFIG=configs/occscannet/radio_occscannet_mini.py
SMOKE_CONFIG=configs/occscannet/radio_occscannet_mini_smoke.py

# Or:
CONFIG=configs/occscannet/efficientnet_b7_occscannet_mini.py
SMOKE_CONFIG=configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py
```

## Shared runtime and depth defaults

- Runtime layout: fixed repo-relative paths under `data/`, `pretrain/`, and `checkpoints/`.
- Distributed wrappers do not export CUDA/NCCL/HF/AdaOcc path variables; pass run naming with `--run-label` and use command-prefix env only for explicit mode switches.
- Default depth mode: local/raw same-stem `posed_images/<scene>/<frame>.png` files (`ADAOCC_ONLINE_DEPTH=0`, `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`).
- Online Depth-Anything mode is opt-in: set `ADAOCC_ONLINE_DEPTH=1` and provide `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth`.
- Generated precomputed depth is opt-in: prepare `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png` with the FT-DaV2 checkpoint used by SPlatSSC, then set `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` when online depth is disabled.
- Reference seed: `0`.
- Epochs: 200.
- Global batch size: 64.
- Query schedule: `100 -> 500`, +100 every 40 epochs.
- Optimizer: AdamW, lr `2e-4`, weight decay `0.01`.

## Reference metrics

Released RADIO online-depth epoch-200 reference:

| mIoU | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furniture | objects |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 58.49 | 65.49 | 47.80 | 57.61 | 56.41 | 48.26 | 59.09 | 75.04 | 75.29 | 57.78 | 43.56 | 64.60 | 57.99 |

A generated precomputed-depth RADIO baseline from the same project was close (`mIoU≈58.20`, `IoU≈65.31`). EfficientNet-B7 is documented as an additional image encoder config option, not as a replacement for the released RADIO baseline or as a new metric claim.

Evaluate the released checkpoint with the RADIO config and online Depth-Anything enabled:

```bash
ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

## Smoke and loss-scale check

Before a full run, run the matching smoke config on the normal mini PKLs:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$SMOKE_CONFIG" \
  --run-label smoke-config-selected
```

Then inspect early logs:

```bash
latest_log=$(find outputs/AdaOcc -name '*.log' | sort | tail -n 1)
python scripts/check_loss_scale.py "$latest_log" --first-n 20
```

Expected first-stage signals:

- `train_runtime_num_query: 100`
- finite total loss, usually around `5-8` early
- finite, nonzero `loss_containment`

## Full commands

Default local/raw-depth training and evaluation with the selected config is for new raw-depth runs and checkpoints. These commands do not evaluate the released online-depth checkpoint; use the fixed RADIO command above for that checkpoint.

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$CONFIG" \
  --run-label config-selected-raw-depth-mini

ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 "$CONFIG" /path/to/epoch_200.pth
```

Optional online Depth-Anything mode:

```bash
ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$CONFIG" \
  --run-label online-depth-mini

ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 "$CONFIG" /path/to/epoch_200.pth
```

Optional generated precomputed-depth mode first requires FT-DaV2/SPlatSSC-aligned depth PNGs:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png
```

Then train/evaluate with the generated depth tree selected:

```bash
ADAOCC_RAW_DEPTH_FROM_IMAGES=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$CONFIG" \
  --run-label precomputed-depth-mini

ADAOCC_RAW_DEPTH_FROM_IMAGES=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 "$CONFIG" /path/to/epoch_200.pth
```

Record full commands, environment versions, config path, checkpoint path, and metrics.
