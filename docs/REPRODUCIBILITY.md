# Reproducibility

## Reference config

- Config: `configs/adaocc/radio_occscannet_mini.py`
- Runtime layout: fixed repo-relative paths under `data/` and `pretrain/`
- Default depth mode: online DepthAnything (`ADAOCC_ONLINE_DEPTH=1` by default)
- Reference seed: `301619034`
- Epochs: 200
- Global batch size: 64
- Query schedule: `100 -> 500`, +100 every 40 epochs
- Optimizer: AdamW, lr `2e-4`, weight decay `0.01`

## Reference metrics

Online-depth epoch-200 reference:

| metric | target | tolerance |
| --- | ---: | ---: |
| `occ/mIoU` | 58.49 | ±0.50 |
| `occ/IoU` | 65.49 | ±0.50 |
| `occ/mIoU_small` | 45.68 | observational |
| `occ/mIoU_head` | 61.34 | observational |

A precomputed-depth baseline from the same project was close (`mIoU≈58.20`, `IoU≈65.31`), but online depth is the default public reproduction path.

## Smoke and loss-scale check

Before a full run, run one epoch with validation on the normal mini PKLs:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
ADAOCC_RUN_LABEL=smoke-1epoch \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini_smoke.py
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
ADAOCC_RUN_LABEL=online-depth-mini \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py

ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Optional precomputed-depth mode:

```bash
ADAOCC_ONLINE_DEPTH=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py
```

Record full commands, environment versions, checkpoint path, and metrics.
