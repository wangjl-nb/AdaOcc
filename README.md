# AdaOcc

**AdaOcc: Adaptive 3D Occupancy Prediction for Embodied Tasks**

AdaOcc is a point-based adaptive 3D semantic occupancy framework for embodied scene understanding. It represents occupied regions as sparse semantic points and supports flexible inference by adjusting query numbers or decoder depth. To handle heterogeneous embodied platforms, AdaOcc combines RGB observations with geometric cues from estimated depth maps, depth cameras, or LiDAR scans through an adaptive geometry-guided dual-branch encoder. Progressive query learning and containment-guided optimization improve the accuracy-efficiency trade-off and spatial consistency around occupied structures.

This repository provides the public AdaOcc code, data-preparation scripts, and checkpoints for running the OccScanNet-mini setup. The released/reference image encoder is RADIO, and EfficientNet-B7 is available as an additional config-selected image encoder option. The default depth path uses the original raw depth PNGs shipped with OccScanNet next to each RGB frame; optional online Depth-Anything-V2 and generated precomputed-depth paths are also supported.

This repo contains code and docs only. It does not redistribute OccScanNet data, generated labels/depth maps, pretrained weights, checkpoints, logs, or private artifacts.

## Released result and checkpoints

Checkpoints: <https://huggingface.co/wjldragon/AdaOcc>

| file | use |
| --- | --- |
| `pretrain/fusion_pretrain_model.pth` | slim OPUS-derived initializer for the current TPV/sparse encoder path |
| `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` | trained online-depth OccScanNet-mini checkpoint |
| `configs/radio_occscannet_mini_training_snapshot.py` | config snapshot from the released run |
| `logs/online_depth_occscannet_mini_epoch200.log` | training/eval log |

Released epoch-200 OccScanNet-mini validation result:

| mIoU | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furniture | objects |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 58.49 | 65.49 | 47.80 | 57.61 | 56.41 | 48.26 | 59.09 | 75.04 | 75.29 | 57.78 | 43.56 | 64.60 | 57.99 |

Expected reproduction tolerance is about ±0.5 for `mIoU` / `IoU`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the reference config, smoke checks, and metric tolerance.

Evaluate the released checkpoint with the RADIO config and online Depth-Anything enabled:

```bash
ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

## 1. Start from OccScanNet

Download/prepare OccScanNet according to its license. Before running AdaOcc scripts, the OccScanNet root should at least contain the original mini split files, frame metadata, posed images, and same-stem raw depth PNGs:

```text
/path/to/OccScanNet/
├── train_subscenes.txt
├── val_subscenes.txt
├── gathered_data/
│   └── <scene>/
│       └── <frame>.pkl
└── posed_images/
    └── <scene>/
        ├── <frame>.jpg
        └── <frame>.png
```

Then put that root inside this repo as `data/OccScanNet`. Copying or symlinking is fine for local reproduction:

```bash
mkdir -p data
ln -s /path/to/OccScanNet data/OccScanNet
```

At this point, `train_occscannet_mini.pkl` and `gts_camvisbits/` may not exist yet; they are generated in Step 4. The default training path reads the original `posed_images/<scene>/<frame>.png` raw depth files directly. See [`docs/DATA.md`](docs/DATA.md) for label keys, raw-axis convention, and depth PNG details.

## 2. Put weights/checkpoints in fixed paths

AdaOcc configs use repo-relative paths, so place external weights under `pretrain/` and the released AdaOcc model checkpoint under `checkpoints/`.

Assets:

| asset | source | target |
| --- | --- | --- |
| OccScanNet | <https://huggingface.co/datasets/hongxiaoy/OccScanNet> | `data/OccScanNet/` |
| RADIO | <https://huggingface.co/nvidia/C-RADIOv3-B> | `pretrain/radio/C-RADIOv3-B/` |
| EfficientNet-B7 weight file, required only for the EfficientNet-B7 config | <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth> | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` |
| Depth-Anything FT checkpoint, optional for `ADAOCC_ONLINE_DEPTH=1` or generated-depth mode | <https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth>; also used by <https://github.com/Made-Gpt/SplatSSC> as FT-DaV2 | `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` |
| AdaOcc fusion pretrain | <https://huggingface.co/wjldragon/AdaOcc/blob/main/pretrain/fusion_pretrain_model.pth> or full OPUS pretrain from <https://github.com/jbwang1997/OPUS> | `pretrain/fusion_pretrain_model.pth` |
| Released AdaOcc model checkpoint | <https://huggingface.co/wjldragon/AdaOcc/blob/main/checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth> | `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` |

Download AdaOcc-provided checkpoints **from the AdaOcc repo root**. The HF repo already stores these files with repo-relative paths, and `--local-dir .` means "place them under the current AdaOcc directory while preserving those paths":

```bash
# run inside AdaOcc/
hf download wjldragon/AdaOcc \
  pretrain/fusion_pretrain_model.pth \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth \
  --local-dir .
```

After download, the files should be here:

```text
AdaOcc/pretrain/fusion_pretrain_model.pth
AdaOcc/checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

Do not use `--local-dir pretrain` or `--local-dir checkpoints` for this command, because `hf download` preserves the HF repo paths and would create nested paths such as `pretrain/pretrain/...`.

The two files have different roles:

- `pretrain/fusion_pretrain_model.pth` is the training initializer. It is a slim OPUS-derived fusion pretrain that only keeps the sparse encoder weights used by the public config. Download this for training from scratch.
- `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` is the fully trained AdaOcc online-depth model. Download this for evaluating the released result directly; it is not required if you only want to train from scratch.

Download RADIO locally:

```bash
hf download nvidia/C-RADIOv3-B --local-dir pretrain/radio/C-RADIOv3-B
```

If you choose the EfficientNet-B7 config, download its weight file to `pretrain/timm`:

```bash
mkdir -p pretrain/timm; wget -O pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
```

For `ADAOCC_ONLINE_DEPTH=1` or generated precomputed-depth mode, place the Depth-Anything checkpoint manually at:

```text
pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
```

If you prefer the full OPUS-generated fusion pretrain instead of the slim AdaOcc file, place it at the same target path: `pretrain/fusion_pretrain_model.pth`. See [`docs/LICENSE_AND_ASSETS.md`](docs/LICENSE_AND_ASSETS.md) for upstream asset routes, attribution notes, and OPUS extraction details.

## 3. Create the environment

Use the machine/conda defaults for CUDA, NCCL, library paths, and distributed launch. AdaOcc does not require global CUDA/NCCL exports in the README path.

```bash
conda env create -f environment.yml
conda activate AdaOcc
```

After activating the environment, set build variables for this shell and install Python dependencies:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H20; adjust for other GPUs.
export MAX_JOBS=8

python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

The custom MSMV CUDA extension is optional for the public single-level image-encoder configs. If it is unavailable, use the PyTorch fallback by prefixing the train/eval command with `ADAOCC_DISABLE_MSMV_CUDA=1`. If you want to compile it, reuse the same shell variables above:

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
```

Do not set `CUDA_VISIBLE_DEVICES` or NCCL variables in the scripts unless your machine or cluster specifically requires them. See [`docs/INSTALL.md`](docs/INSTALL.md) for the tested stack, MMCV/MSMV build notes, and CUDA mismatch fallback guidance; [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) is a short environment checklist.

## 4. Generate AdaOcc data files

Generate mini PKLs from `train_subscenes.txt` / `val_subscenes.txt`:

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite
```

This creates:

```text
data/OccScanNet/train_occscannet_mini.pkl
data/OccScanNet/val_occscannet_mini.pkl
data/OccScanNet/test_occscannet_mini.pkl
```

Generate and verify `labels.npz` supervision from `gathered_data/*.pkl`:

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --verify-only
```

Default raw-depth training does not need Depth-Anything or generated precomputed depth PNGs; it reads the original same-stem depth files under `posed_images/`. Set `ADAOCC_ONLINE_DEPTH=1` only when you want online Depth-Anything prediction during train/eval.

For the optional generated precomputed-depth mode, first generate the SPlatSSC-style FT-DaV2-aligned depth tree with the same Depth-Anything-V2 checkpoint used by SPlatSSC:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --verify-only
```

The script reads the mini PKLs, loads `CAM_FRONT` images, writes depth to each pkl-referenced `cam["depth_path"]` under `depth_splatssc_stage1_ftdav2_vitb_20m_full/`, and skips existing files unless `--overwrite` is passed. Use `--limit N` for a small generation smoke test, or `--device cuda:0` / `--device cpu` to choose the inference device. Select these generated PNGs with `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` when online depth is disabled.

Precomputed depth PNGs are binary depth containers, not visual images: each pixel is one little-endian `float32` meter value stored as four uint8 PNG channels (`H x W x 4`). Do not convert, resize, color-map, or re-save them with image editors.

Check assets:

```bash
python scripts/check_assets.py --radio --raw-depth-from-images --verify-depth-png
```

Use `--radio` for the RADIO config asset. Use `--efficientnet-b7` instead when checking the EfficientNet-B7 config asset; that requires only `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` and does not require RADIO. Omitting both image-encoder flags keeps the backwards-compatible RADIO check. Use `--online-depth` for the optional online Depth-Anything mode, or `--precomputed-depth --verify-depth-png` for the optional generated-depth mode. More data-generation details are in [`docs/DATA.md`](docs/DATA.md).

## 5. Final expected layout

After Steps 1-4, the repo should look like this:

```text
AdaOcc/
├── data/
│   └── OccScanNet/
│       ├── train_subscenes.txt
│       ├── val_subscenes.txt
│       ├── train_occscannet_mini.pkl
│       ├── val_occscannet_mini.pkl
│       ├── test_occscannet_mini.pkl
│       ├── gathered_data/
│       │   └── <scene>/
│       │       └── <frame>.pkl
│       ├── posed_images/
│       │   └── <scene>/
│       │       ├── <frame>.jpg
│       │       └── <frame>.png
│       ├── gts_camvisbits/
│       │   └── <scene>/
│       │       └── <frame>/
│       │           └── labels.npz
│       └── depth_splatssc_stage1_ftdav2_vitb_20m_full/  # optional
│           └── <scene>/
│               └── <frame>.png
├── pretrain/
│   ├── fusion_pretrain_model.pth
│   ├── depth_anything/
│   │   └── finetune_scannet_depthanythingv2.pth
│   ├── timm/
│   │   └── tf_efficientnet_b7_ns-1dbc32de.pth  # only for EfficientNet-B7 config
│   └── radio/
│       └── C-RADIOv3-B/
├── checkpoints/
│   └── adaocc_online_depth_occscannet_mini_epoch200.pth
└── outputs/
```

## 6. Choose image encoder config

All smoke/train/eval commands below take a config path. Choose one image encoder config and keep using it for that run:

| choice | full config | smoke config | extra asset |
| --- | --- | --- | --- |
| RADIO released/reference baseline | `configs/occscannet/radio_occscannet_mini.py` | `configs/occscannet/radio_occscannet_mini_smoke.py` | `pretrain/radio/C-RADIOv3-B/` |
| EfficientNet-B7 additional option | `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py` | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` |

RADIO remains the released/reference baseline and is the config to use with the released checkpoint/metrics above. EfficientNet-B7 changes the config-selected `img_encoder.image_backbone_cfg`; it is not a new default and this README does not claim EfficientNet reproduction metrics.

Set the config variables once per shell, for example:

```bash
# RADIO baseline
CONFIG=configs/occscannet/radio_occscannet_mini.py
SMOKE_CONFIG=configs/occscannet/radio_occscannet_mini_smoke.py

# Or EfficientNet-B7 option
CONFIG=configs/occscannet/efficientnet_b7_occscannet_mini.py
SMOKE_CONFIG=configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py
```

## 7. Smoke, train, evaluate

Import/config smoke:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 CONFIG="$CONFIG" python - <<'PY'
import os
import torch, mmcv, mmengine, mmdet, mmdet3d, spconv, transformers
from mmengine.config import Config
cfg = Config.fromfile(os.environ['CONFIG'])
backbone = cfg.model.img_encoder.image_backbone_cfg
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())
print('model', cfg.model.type, 'image_backbone', backbone.type, 'online_depth', cfg.model.online_depth.enabled)
PY
```

Smoke training with the matching smoke config:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$SMOKE_CONFIG" \
  --run-label smoke-config-selected
```

Default raw-depth full training for a new raw-depth checkpoint:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$CONFIG" \
  --run-label config-selected-raw-depth-mini
```

Evaluate a checkpoint produced by that new raw-depth run with the same config:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 "$CONFIG" /path/to/epoch_200.pth
```

The released checkpoint is an online-depth RADIO checkpoint, not a default raw-depth checkpoint. To reproduce the released validation result, use the fixed RADIO config path, enable online depth, and pass the released checkpoint path:

```bash
ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

For optional online Depth-Anything train/eval of a new run, keep the same config and add `ADAOCC_ONLINE_DEPTH=1`.

For optional generated precomputed-depth training/eval, make sure `depth_splatssc_stage1_ftdav2_vitb_20m_full/` exists and set `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` for both commands:

```bash
ADAOCC_RAW_DEPTH_FROM_IMAGES=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 "$CONFIG" \
  --run-label precomputed-depth-mini

ADAOCC_RAW_DEPTH_FROM_IMAGES=0 ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 "$CONFIG" /path/to/epoch_200.pth
```

`dist_train.sh` and `dist_val.sh` intentionally do not export CUDA, NCCL, Hugging Face, or AdaOcc path variables internally. They only call `torch.distributed.run` with the requested GPU count, config, and remaining arguments. Use `train.py` arguments such as `--run-label`, `--output-root`, or `--work-dir` for run placement; use command-prefix environment variables only for explicit mode switches such as `ADAOCC_DISABLE_MSMV_CUDA=1`, `ADAOCC_ONLINE_DEPTH=1`, or `ADAOCC_RAW_DEPTH_FROM_IMAGES=0`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the online-depth/RADIO/EfficientNet/TPV data flow and query schedule; [`docs/DEPENDENCY_TRACE.md`](docs/DEPENDENCY_TRACE.md) maps major code modules.

## More details

- [`docs/INSTALL.md`](docs/INSTALL.md) / [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md): environment and optional CUDA extension notes
- [`docs/DATA.md`](docs/DATA.md): PKLs, label fields, raw-axis convention, and depth PNG format
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): RADIO/EfficientNet image encoders, depth paths, TPV, and query schedule
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): smoke/full reproduction checklist and reference metrics
- [`docs/LICENSE_AND_ASSETS.md`](docs/LICENSE_AND_ASSETS.md): upstream assets, licenses, and citations
- [`docs/DEPENDENCY_TRACE.md`](docs/DEPENDENCY_TRACE.md): major code-module map

Please cite/acknowledge OccScanNet, ScanNet, CompleteScanNet/SCFusion if used by your data preparation, OPUS, SPlatSSC, Depth Anything V2, and RADIO according to their licenses.
