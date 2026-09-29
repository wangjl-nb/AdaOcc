# AI-agent AdaOcc reproduction guide

Use this file as the command-oriented source of truth when an AI agent is asked to reproduce public AdaOcc on OccScanNet-mini and the released OccScanNet full-split checkpoint.

## Role and scope

You are reproducing the public AdaOcc repository state, not redesigning the project.

Allowed work:

- prepare local data and assets under the expected repo-relative paths;
- run environment checks, data-generation scripts, smoke training, training, and evaluation;
- report exact commands, metrics, logs, and blockers.

Non-goals:

- do not modify unrelated model, data-loader, training, evaluation, or config logic;
- do not change dataset contents except by running documented generation scripts;
- do not add new dependencies unless explicitly requested;
- do not commit or push unless explicitly asked.

## Fixed paths

Run commands from the AdaOcc repository root.

```text
AdaOcc/
├── data/
│   └── OccScanNet/
│       ├── train_subscenes.txt
│       ├── val_subscenes.txt
│       ├── gathered_data/
│       │   └── <scene>/
│       │       └── <frame>.pkl
│       ├── posed_images/
│       │   └── <scene>/
│       │       ├── <frame>.jpg
│       │       └── <frame>.png
│       ├── train_occscannet_mini.pkl
│       ├── val_occscannet_mini.pkl
│       ├── test_occscannet_mini.pkl
│       ├── train_occscannet_full.pkl
│       ├── val_occscannet_full.pkl
│       ├── test_occscannet_full.pkl
│       ├── gts_camvisbits/
│       │   └── <scene>/
│       │       └── <frame>/
│       │           └── labels.npz
│       └── depth_splatssc_stage1_ftdav2_vitb_20m_full/  # optional
│           └── <scene>/
│               └── <frame>.png
├── pretrain/                         # external/pretrained weights and initializers
│   ├── fusion_pretrain_model.pth
│   ├── radio/
│   │   └── C-RADIOv3-B/
│   ├── timm/
│   │   └── tf_efficientnet_b7_ns-1dbc32de.pth
│   └── depth_anything/
│       └── finetune_scannet_depthanythingv2.pth
├── checkpoints/                      # trained AdaOcc model checkpoints
│   ├── adaocc_online_depth_occscannet_mini_epoch200.pth
│   └── adaocc_radio_occscannet_full_epoch100.pth
└── outputs/
```

Rules:

- `pretrain/` holds external pretrained weights and training initializers.
- `checkpoints/` holds released or user-provided trained AdaOcc model checkpoints.
- Default depth uses local/prepared same-stem `posed_images/<scene>/<frame>.png` with `ADAOCC_ONLINE_DEPTH=0` and `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`; it is not online prediction.
- Online depth is opt-in with `ADAOCC_ONLINE_DEPTH=1`.
- Generated precomputed depth is opt-in with `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` when online depth is disabled; the generated tree is `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png`.
- `configs/occscannet/radio_occscannet_full.py` is the only public config that defaults to generated precomputed depth, because the released full-split run used it.

## Asset paths

| asset | source | target path | required for |
| --- | --- | --- | --- |
| OccScanNet | <https://huggingface.co/datasets/hongxiaoy/OccScanNet> | `data/OccScanNet/` | all runs |
| RADIO weights/cache | <https://huggingface.co/nvidia/C-RADIOv3-B> | `pretrain/radio/C-RADIOv3-B/` | RADIO config |
| EfficientNet-B7 weight | <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth> | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` | EfficientNet-B7 config |
| Fusion pretrain | <https://huggingface.co/wjldragon/AdaOcc/blob/main/pretrain/fusion_pretrain_model.pth> | `pretrain/fusion_pretrain_model.pth` | training from scratch |
| Depth-Anything FT checkpoint | <https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth> | `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` | online depth or generated precomputed depth |
| Released AdaOcc mini checkpoint | <https://huggingface.co/wjldragon/AdaOcc/blob/main/checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth> | `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` | released mini-checkpoint evaluation |
| Released AdaOcc full-split checkpoint | <https://huggingface.co/wjldragon/AdaOcc/blob/main/checkpoints/adaocc_radio_occscannet_full_epoch100.pth> | `checkpoints/adaocc_radio_occscannet_full_epoch100.pth` | released full-split-checkpoint evaluation |

Direct asset commands:

```bash
hf download wjldragon/AdaOcc \
  pretrain/fusion_pretrain_model.pth \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth \
  checkpoints/adaocc_radio_occscannet_full_epoch100.pth \
  --local-dir .

hf download nvidia/C-RADIOv3-B --local-dir pretrain/radio/C-RADIOv3-B

mkdir -p pretrain/timm
wget -O pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
```

Place `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` manually from the public FT-DaV2 checkpoint URL in the table when needed.

## Environment

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

The MSMV CUDA extension is optional for public reproduction. If it is not built or unavailable, set `ADAOCC_DISABLE_MSMV_CUDA` to `1` only on the affected command to use the PyTorch fallback; do not prefix every command by default.

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py \
  --run-label smoke-radio
```

If building the optional extension, set `TORCH_CUDA_ARCH_LIST` for the actual GPU:

```bash
cd models/csrc
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  python setup.py build_ext --inplace
cd ../..
```

## Data prep order

1. Put or symlink OccScanNet at `data/OccScanNet`.

```bash
mkdir -p data
ln -s /path/to/OccScanNet data/OccScanNet
```

2. Generate mini PKLs, and full-split PKLs when reproducing `configs/occscannet/radio_occscannet_full.py`.

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite

python scripts/generate_occscannet_mini_pkls.py \
  --data-root data/OccScanNet \
  --train-count 0 --val-count 0 \
  --train-output train_occscannet_full.pkl \
  --val-output val_occscannet_full.pkl \
  --test-output test_occscannet_full.pkl \
  --overwrite
```

3. Generate and verify labels. Add the full-split `--splits` for the full-split checkpoint.

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --verify-only

python scripts/generate_occscannet_mini_gts_camvisbits.py \
  --data-root data/OccScanNet \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl \
  --overwrite
```

4. Validate default local/prepared depth data and the selected image-encoder asset.

RADIO:

```bash
python scripts/check_assets.py --radio --raw-depth-from-images --verify-depth-png
```

EfficientNet-B7:

```bash
python scripts/check_assets.py --efficientnet-b7 --raw-depth-from-images --verify-depth-png
```

5. Optional generated precomputed depth. The full-split checkpoint requires it, so also run the full-split `--splits` variant.

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl
```

Validate generated precomputed depth:

```bash
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png
python scripts/check_assets.py --efficientnet-b7 --precomputed-depth --verify-depth-png
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png \
  --splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl
```

## Config choices

Choose the image encoder by passing the config path directly to `dist_train.sh` / `dist_val.sh`.

| choice | full config | smoke config | metric claim |
| --- | --- | --- | --- |
| RADIO released/reference baseline (OccScanNet-mini) | `configs/occscannet/radio_occscannet_mini.py` | `configs/occscannet/radio_occscannet_mini_smoke.py` | released mini checkpoint/metrics |
| RADIO released full-split checkpoint (OccScanNet full) | `configs/occscannet/radio_occscannet_full.py` | — | released full-split checkpoint/metrics |
| EfficientNet-B7 additional option | `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py` | no released metric claim |

Do not use global `CONFIG` or `SMOKE_CONFIG` shell variables in reports. Write the config path directly in each command.

## Validation commands

Syntax check:

```bash
python -m py_compile scripts/check_assets.py scripts/check_checkpoint.py
```

Public script tests:

```bash
pytest -q tests/test_public_scripts.py
```

Markdown/file whitespace check:

```bash
git diff --check -- README.md docs/*.md scripts/check_assets.py scripts/check_checkpoint.py tests/test_public_scripts.py
```

Quick config import check:

```bash
python - <<'PY'
from mmengine.config import Config
for path in [
    'configs/occscannet/radio_occscannet_mini.py',
    'configs/occscannet/radio_occscannet_full.py',
    'configs/occscannet/efficientnet_b7_occscannet_mini.py',
]:
    cfg = Config.fromfile(path)
    print(path, cfg.model.type, cfg.model.img_encoder.image_backbone_cfg.type, cfg.model.online_depth.enabled)
PY
```

Checkpoint/config compatibility check:

```bash
python scripts/check_checkpoint.py \
  --config configs/occscannet/radio_occscannet_mini.py \
  --checkpoint checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth

python scripts/check_checkpoint.py \
  --config configs/occscannet/radio_occscannet_full.py \
  --checkpoint checkpoints/adaocc_radio_occscannet_full_epoch100.pth
```

## Smoke, train, and eval commands

### RADIO default local/prepared-depth run

Use these commands for a default local/prepared-depth RADIO run and checkpoint.

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py \
  --run-label smoke-radio

./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-raw-depth-mini

./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py /path/to/epoch_200.pth
```

### EfficientNet-B7 default local/prepared-depth run

Use these commands for a default local/prepared-depth EfficientNet-B7 run and checkpoint.

```bash
./dist_train.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py \
  --run-label smoke-efficientnet-b7

./dist_train.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini.py \
  --run-label efficientnet-b7-raw-depth-mini

./dist_val.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini.py /path/to/epoch_200.pth
```

### Released checkpoint evaluation

The released checkpoint is a RADIO online-depth checkpoint:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

Expected released RADIO online-depth epoch-200 validation result is about `mIoU=58.49`, `IoU=65.49`, with about ±0.5 tolerance.

### Released full-split checkpoint evaluation

The released full-split checkpoint uses the RADIO full config, whose default depth mode is generated precomputed depth:

```bash
./dist_val.sh 8 configs/occscannet/radio_occscannet_full.py \
  checkpoints/adaocc_radio_occscannet_full_epoch100.pth
```

Expected released RADIO full-split epoch-100 validation result on OccScanNet full validation is about `mIoU=59.67`, `IoU=65.29`, with about ±0.5 tolerance.

### Full-split RADIO run

Use these commands for the full-split RADIO config (100 epochs, 200 -> 500 queries, generated precomputed depth by default).

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_full.py \
  --run-label radio-full-split

./dist_val.sh 8 configs/occscannet/radio_occscannet_full.py /path/to/epoch_100.pth
```

## Depth mode switches

Default local/prepared depth:

```bash
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-raw-depth-mini
```

Online Depth-Anything opt-in:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-online-depth-mini
```

Generated precomputed-depth opt-in:

```bash
ADAOCC_RAW_DEPTH_FROM_IMAGES=0 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-precomputed-depth-mini
```

Use the same depth-mode prefix for train and eval of a given checkpoint.

## Troubleshooting

- Missing RADIO asset: run `hf download nvidia/C-RADIOv3-B --local-dir pretrain/radio/C-RADIOv3-B` or use the EfficientNet-B7 config and check with `--efficientnet-b7`.
- Missing EfficientNet-B7 asset: download `https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth` to `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth`.
- Missing raw depth: confirm `posed_images/<scene>/<frame>.png` exists next to each RGB `posed_images/<scene>/<frame>.jpg`.
- Missing labels: rerun `scripts/generate_occscannet_mini_gts_camvisbits.py` and `--verify-only`.
- Optional MSMV import warning: rerun the affected command with `ADAOCC_DISABLE_MSMV_CUDA` set to `1` to use the PyTorch fallback.
- CUDA extension build mismatch: use the conda CUDA toolkit matching `torch.version.cuda`, or keep the PyTorch fallback.
- Released checkpoint mismatch: confirm the command uses `ADAOCC_ONLINE_DEPTH=1`, the RADIO config, and `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth`.
- Released full-split checkpoint mismatch: confirm the command uses `configs/occscannet/radio_occscannet_full.py`, the full PKLs, `checkpoints/adaocc_radio_occscannet_full_epoch100.pth`, and the default precomputed depth (no `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`).

## Final report checklist

Report:

- git diff summary and changed files, if any;
- exact data-root and asset paths checked;
- exact config path and depth mode used;
- smoke command and result;
- train command, output directory, final checkpoint path, and early loss-scale observation;
- eval command and metrics;
- validation commands and pass/fail status;
- any blockers, skipped steps, or assumptions.
