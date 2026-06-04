# Reproducibility

## Reference config

- Config: `configs/adaocc/radio_occscannet_mini.py`
- Runtime layout: fixed repo-relative paths under `data/` and `pretrain/`
- Distributed wrappers do not export CUDA/NCCL/HF/AdaOcc path variables; pass run naming with `--run-label` and use command-prefix env only for explicit mode switches.
- Default depth mode: online DepthAnything (`ADAOCC_ONLINE_DEPTH=1` by default)
- Reference seed: `301619034`
- Epochs: 200
- Global batch size: 64
- Query schedule: `100 -> 500`, +100 every 40 epochs
- Optimizer: AdamW, lr `2e-4`, weight decay `0.01`

## Reference metrics

Online-depth epoch-200 reference:

| mIoU | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furniture | objects |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 58.49 | 65.49 | 47.80 | 57.61 | 56.41 | 48.26 | 59.09 | 75.04 | 75.29 | 57.78 | 43.56 | 64.60 | 57.99 |

A precomputed-depth baseline from the same project was close (`mIoU≈58.20`, `IoU≈65.31`), but online depth is the default public reproduction path.

## Smoke and loss-scale check

Before a full run, run one epoch with validation on the normal mini PKLs:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini_smoke.py \
  --run-label smoke-1epoch
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

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py \
  --run-label online-depth-mini

ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Optional precomputed-depth mode first requires generated depth PNGs:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only
python scripts/check_assets.py --precomputed-depth --verify-depth-png
```

Then train/evaluate with online depth disabled:

```bash
ADAOCC_ONLINE_DEPTH=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py \
  --run-label precomputed-depth-mini

ADAOCC_ONLINE_DEPTH=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Record full commands, environment versions, checkpoint path, and metrics.
