# Reproducibility

## Config choices

Choose the image encoder by passing the config path directly to `dist_train.sh` / `dist_val.sh`.

| choice | full config | smoke config | notes |
| --- | --- | --- | --- |
| RADIO released/reference baseline (OccScanNet-mini) | `configs/occscannet/radio_occscannet_mini.py` | `configs/occscannet/radio_occscannet_mini_smoke.py` | Use this for the released OccScanNet-mini checkpoint/metrics. |
| RADIO released full-split checkpoint (OccScanNet full) | `configs/occscannet/radio_occscannet_full.py` | — | Full-split companion of the mini baseline; same model, but full PKLs, a 100-epoch / 200-500 query schedule, and generated precomputed depth by default. |
| EfficientNet-B7 additional image encoder option | `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py` | Requires `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth`; no released metrics are claimed here. |

RADIO remains the released/reference baseline. EfficientNet-B7 is an additional config-selected image encoder option, not a replacement baseline. `radio_occscannet_full.py` is a data-split/schedule companion of `radio_occscannet_mini.py`, not a different architecture.

## Shared runtime and depth defaults

- Runtime layout: fixed repo-relative paths under `data/`, `pretrain/`, and `checkpoints/`.
- `pretrain/` stores pretrained encoder/depth/fusion assets; `checkpoints/` stores trained AdaOcc model checkpoints.
- Distributed wrappers do not export CUDA/NCCL/HF/AdaOcc path variables; pass run naming with `--run-label` and use command-prefix env only for explicit mode switches.
- Default depth mode: local/prepared same-stem `posed_images/<scene>/<frame>.png` files (`ADAOCC_ONLINE_DEPTH=0`, `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`), not online prediction.
- Online Depth-Anything mode is opt-in: set `ADAOCC_ONLINE_DEPTH=1` and provide `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth`.
- Generated precomputed depth is opt-in: prepare `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png`, then set `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` when online depth is disabled.
- Reference seed: `0`.
- OccScanNet-mini baseline (`radio_occscannet_mini.py`): 200 epochs, global batch size 64, query schedule `100 -> 500` (+100 every 40 epochs).
- OccScanNet full-split release (`radio_occscannet_full.py`): 100 epochs, global batch size 64, query schedule `200 -> 500` (+100 every 25 epochs), generated precomputed depth by default.
- Optimizer: AdamW, lr `2e-4`, weight decay `0.01`.

## Optional MSMV fallback

The MSMV CUDA extension is optional. If it is not built or unavailable, set `ADAOCC_DISABLE_MSMV_CUDA` to `1` on the affected command to use the PyTorch fallback; do not prefix every command by default.

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py \
  --run-label smoke-radio
```

## Reference metrics

Released RADIO AdaOcc checkpoints:

| checkpoint | split | epoch | mIoU | IoU |
| --- | --- | ---: | ---: | ---: |
| `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` | OccScanNet-mini | 200 | 58.49 | 65.49 |
| `checkpoints/adaocc_radio_occscannet_full_epoch100.pth` | OccScanNet full | 100 | 59.67 | 65.29 |

Released RADIO online-depth epoch-200 OccScanNet-mini per-class reference:

| mIoU | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furniture | objects |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 58.49 | 65.49 | 47.80 | 57.61 | 56.41 | 48.26 | 59.09 | 75.04 | 75.29 | 57.78 | 43.56 | 64.60 | 57.99 |

A generated precomputed-depth RADIO mini baseline from the same project was close (`mIoU≈58.20`, `IoU≈65.31`). EfficientNet-B7 has no released metric claim in this repository.

Evaluate the released checkpoint with the RADIO config and online Depth-Anything enabled:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

Evaluate the released full-split checkpoint with the RADIO full config (generated precomputed depth is the config default):

```bash
./dist_val.sh 8 configs/occscannet/radio_occscannet_full.py \
  checkpoints/adaocc_radio_occscannet_full_epoch100.pth
```

Expected reproduction tolerance is about ±0.5 for `mIoU` / `IoU`. Both released checkpoints load into their public configs without key remapping; check a pair before evaluating with `python scripts/check_checkpoint.py --config <config> --checkpoint <checkpoint>`.

## Asset checks

Default local/prepared-depth RADIO assets:

```bash
python scripts/check_assets.py --radio --raw-depth-from-images --verify-depth-png
```

Default local/prepared-depth EfficientNet-B7 assets:

```bash
python scripts/check_assets.py --efficientnet-b7 --raw-depth-from-images --verify-depth-png
```

Online-depth asset check:

```bash
python scripts/check_assets.py --radio --raw-depth-from-images --online-depth --verify-depth-png
```

Generated precomputed-depth asset checks:

```bash
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png
python scripts/check_assets.py --efficientnet-b7 --precomputed-depth --verify-depth-png
```

Full-split checkpoint assets (full PKLs + generated precomputed depth):

```bash
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl
```

Checkpoint/config compatibility checks:

```bash
python scripts/check_checkpoint.py \
  --config configs/occscannet/radio_occscannet_mini.py \
  --checkpoint checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth

python scripts/check_checkpoint.py \
  --config configs/occscannet/radio_occscannet_full.py \
  --checkpoint checkpoints/adaocc_radio_occscannet_full_epoch100.pth
```

## Smoke and loss-scale checks

RADIO smoke:

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py \
  --run-label smoke-radio
```

EfficientNet-B7 smoke:

```bash
./dist_train.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py \
  --run-label smoke-efficientnet-b7
```

Inspect early logs:

```bash
latest_log=$(find outputs/AdaOcc -name '*.log' | sort | tail -n 1)
python scripts/check_loss_scale.py "$latest_log" --first-n 20
```

Expected first-stage signals:

- `train_runtime_num_query: 100`
- finite total loss, usually around `5-8` early
- finite, nonzero `loss_containment`

## Full commands

Default local/prepared-depth RADIO training and evaluation:

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-raw-depth-mini

./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Default local/prepared-depth EfficientNet-B7 training and evaluation:

```bash
./dist_train.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini.py \
  --run-label efficientnet-b7-raw-depth-mini

./dist_val.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini.py /path/to/epoch_200.pth
```

Optional online Depth-Anything mode, shown for RADIO:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-online-depth-mini

ADAOCC_ONLINE_DEPTH=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Optional generated precomputed-depth mode first requires FT-DaV2/SPlatSSC-aligned depth PNGs:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only
```

Then train/evaluate with generated depth selected:

```bash
ADAOCC_RAW_DEPTH_FROM_IMAGES=0 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-precomputed-depth-mini

ADAOCC_RAW_DEPTH_FROM_IMAGES=0 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Released full-split RADIO checkpoint evaluation (the config defaults to generated precomputed depth):

```bash
./dist_val.sh 8 configs/occscannet/radio_occscannet_full.py \
  checkpoints/adaocc_radio_occscannet_full_epoch100.pth
```

Full-split RADIO training (100 epochs, 200 -> 500 queries):

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_full.py \
  --run-label radio-full-split
```

Use the same config path and depth-mode prefix for train/eval of a given checkpoint. Record full commands, environment versions, config path, checkpoint path, and metrics.
