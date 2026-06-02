# Reproducibility

## Reference config

- Config: `configs/adaocc/radio_occscannet_mini.py`
- Default depth mode: `ADAOCC_ONLINE_DEPTH=1` in `configs/local_paths.sh`
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

## Loss-scale check

Before a full run, inspect early logs:

```bash
python scripts/check_loss_scale.py /path/to/train.log --first-n 20
```

Expected first-stage signals:

- `train_runtime_num_query: 100`
- finite total loss, usually around `5-8` early
- finite, nonzero `loss_containment`

## Full commands

```bash
cp configs/local_paths.example.sh configs/local_paths.sh
${EDITOR:-nano} configs/local_paths.sh

./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py \
  --override randomness.seed=301619034

./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Record full commands, environment versions, checkpoint path, and metrics.
