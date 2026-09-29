# AdaOcc

**AdaOcc: Adaptive 3D Occupancy Prediction for Embodied Tasks**

<p align="center">
  <a href="https://wangjl-nb.github.io/AdaOcc_web/"><img src="https://img.shields.io/badge/Project_Page-AdaOcc-green" alt="Project Page"></a>
  <a href="https://huggingface.co/wjldragon/AdaOcc"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow" alt="Models and checkpoints"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-blueviolet" alt="NeurIPS 2026">
</p>

🎉 **AdaOcc has been accepted to NeurIPS 2026.** See the [project page](https://wangjl-nb.github.io/AdaOcc_web/) for videos and real-robot demos, and [Hugging Face](https://huggingface.co/wjldragon/AdaOcc) for released checkpoints.

AdaOcc is a point-based adaptive 3D semantic occupancy framework for embodied scene understanding. This public repository contains code, data-preparation scripts, and reproduction docs for the OccScanNet-mini setup.

If you are using an AI agent to reproduce AdaOcc, point it to [`docs/AI_REPRODUCTION.md`](docs/AI_REPRODUCTION.md).

RADIO remains the released/reference baseline. EfficientNet-B7 additional option support is available through a separate config-selected image encoder path; no released EfficientNet-B7 metric claim is made here. The default depth mode is local/prepared same-stem `posed_images/<scene>/<frame>.png` (`ADAOCC_ONLINE_DEPTH=0`, `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`), not online prediction.

This repo does not redistribute OccScanNet data, pretrained weights, trained checkpoints, generated labels, generated depth files, logs, or private artifacts.

## Released result

Checkpoints and release files: <https://huggingface.co/wjldragon/AdaOcc>

Released RADIO online-depth epoch-200 OccScanNet-mini validation result:

| mIoU | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furniture | objects |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 58.49 | 65.49 | 47.80 | 57.61 | 56.41 | 48.26 | 59.09 | 75.04 | 75.29 | 57.78 | 43.56 | 64.60 | 57.99 |

Expected reproduction tolerance is about ±0.5 for `mIoU` / `IoU`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Evaluate the released checkpoint:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

## 1. Prepare OccScanNet

Prepare or symlink OccScanNet at `data/OccScanNet` and see [`docs/DATA.md`](docs/DATA.md).

```bash
mkdir -p data
ln -s /path/to/OccScanNet data/OccScanNet
```

Minimum starting data layout:

```text
data/OccScanNet/
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

## 2. Download weights and checkpoints

Place pretrained/external weights under `pretrain/`. `checkpoints/` means user-provided or released trained AdaOcc model checkpoints.

| asset | source | target |
| --- | --- | --- |
| OccScanNet | <https://huggingface.co/datasets/hongxiaoy/OccScanNet> | `data/OccScanNet/` |
| RADIO | <https://huggingface.co/nvidia/C-RADIOv3-B> | `pretrain/radio/C-RADIOv3-B/` |
| EfficientNet-B7 weight file | <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth> | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` |
| Depth-Anything FT checkpoint, optional for online/generated depth | <https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth> | `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` |
| AdaOcc fusion pretrain | <https://huggingface.co/wjldragon/AdaOcc/blob/main/pretrain/fusion_pretrain_model.pth> | `pretrain/fusion_pretrain_model.pth` |
| Released AdaOcc trained checkpoint | <https://huggingface.co/wjldragon/AdaOcc/blob/main/checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth> | `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` |

```bash
hf download wjldragon/AdaOcc \
  pretrain/fusion_pretrain_model.pth \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth \
  --local-dir .

hf download nvidia/C-RADIOv3-B --local-dir pretrain/radio/C-RADIOv3-B

mkdir -p pretrain/timm
wget -O pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
```

Expected downloaded project layout:

```text
AdaOcc/
├── data/
│   └── OccScanNet/
├── pretrain/                         # external/pretrained weights and initializers
│   ├── fusion_pretrain_model.pth
│   ├── radio/
│   │   └── C-RADIOv3-B/
│   ├── timm/
│   │   └── tf_efficientnet_b7_ns-1dbc32de.pth
│   └── depth_anything/               # optional unless using online/generated depth
│       └── finetune_scannet_depthanythingv2.pth
└── checkpoints/                      # trained AdaOcc model checkpoints
    └── adaocc_online_depth_occscannet_mini_epoch200.pth
```

See [`docs/LICENSE_AND_ASSETS.md`](docs/LICENSE_AND_ASSETS.md) for asset routes, license notes, and OPUS-derived fusion-pretrain details.

## 3. Install

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

The custom MSMV CUDA extension is optional for these public configs. If it is not built or unavailable, set `ADAOCC_DISABLE_MSMV_CUDA` to `1` on the affected command to use the PyTorch fallback. If you build the extension, set `TORCH_CUDA_ARCH_LIST` for your GPU, for example:

```bash
cd models/csrc
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  python setup.py build_ext --inplace
cd ../..
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for tested versions and CUDA-extension troubleshooting.

## 4. Generate data files

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --verify-only
python scripts/check_assets.py --radio --raw-depth-from-images --verify-depth-png
```

Optional generated precomputed-depth mode:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only
python scripts/check_assets.py --radio --precomputed-depth --verify-depth-png
```

Use `--efficientnet-b7` instead of `--radio` when checking the EfficientNet-B7 asset. Use `--online-depth` when checking the optional online Depth-Anything checkpoint.

## 5. Choose config, depth mode, and run

### 5.1 Choose image encoder by config path

Pass the config path directly to `dist_train.sh` / `dist_val.sh`.

| choice | full config | smoke config | extra asset |
| --- | --- | --- | --- |
| RADIO released/reference baseline | `configs/occscannet/radio_occscannet_mini.py` | `configs/occscannet/radio_occscannet_mini_smoke.py` | `pretrain/radio/C-RADIOv3-B/` |
| EfficientNet-B7 additional option | `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py` | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` |

RADIO is the released/reference baseline. EfficientNet-B7 is an additional config-selected image encoder option and has no released metric claim here.

### 5.2 Choose depth mode

- Default local/prepared depth: no depth env prefix; uses local same-stem `posed_images/<scene>/<frame>.png` files, not online prediction.
- Online predicted depth: prefix the command with `ADAOCC_ONLINE_DEPTH=1` and provide `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth`.
- Generated precomputed depth: after generating `depth_splatssc_stage1_ftdav2_vitb_20m_full/`, prefix the command with `ADAOCC_RAW_DEPTH_FROM_IMAGES=0`.
- Use the same config path and depth-mode switch for train/eval of a given checkpoint.

### 5.3 Optional MSMV fallback

If the optional MSMV CUDA extension is not built or unavailable, set `ADAOCC_DISABLE_MSMV_CUDA` to `1` on that command to use the PyTorch fallback. Do not add it to every command by default.

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py \
  --run-label smoke-radio
```

### 5.4 Final expected layout

```text
data/OccScanNet/
├── train_occscannet_mini.pkl        # generated PKLs
├── val_occscannet_mini.pkl
├── test_occscannet_mini.pkl
├── gts_camvisbits/<scene>/<frame>/labels.npz
├── posed_images/<scene>/<frame>.{jpg,png}
└── depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png   # optional generated precomputed depth
pretrain/                                         # external/pretrained weights/initializers
├── fusion_pretrain_model.pth
├── radio/C-RADIOv3-B/
├── timm/tf_efficientnet_b7_ns-1dbc32de.pth
└── depth_anything/finetune_scannet_depthanythingv2.pth   # optional online/generated-depth initializer
checkpoints/                                      # trained AdaOcc model checkpoints
└── adaocc_online_depth_occscannet_mini_epoch200.pth
outputs/
```

### 5.5 Smoke, train, and eval examples

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

Online predicted-depth RADIO evaluation:

```bash
ADAOCC_ONLINE_DEPTH=1 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

Generated precomputed-depth RADIO training and evaluation after generating `depth_splatssc_stage1_ftdav2_vitb_20m_full/`:

```bash
ADAOCC_RAW_DEPTH_FROM_IMAGES=0 \
./dist_train.sh 8 configs/occscannet/radio_occscannet_mini.py \
  --run-label radio-precomputed-depth-mini

ADAOCC_RAW_DEPTH_FROM_IMAGES=0 \
./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py /path/to/epoch_200.pth
```

## More docs

- [`docs/AI_REPRODUCTION.md`](docs/AI_REPRODUCTION.md): command-oriented AI-agent reproduction guide
- [`docs/INSTALL.md`](docs/INSTALL.md): environment and optional CUDA extension notes
- [`docs/DATA.md`](docs/DATA.md): PKLs, labels, raw-depth defaults, and generated-depth format
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): smoke/full reproduction checklist and reference metrics
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): image encoders, depth paths, TPV, and query schedule
- [`docs/LICENSE_AND_ASSETS.md`](docs/LICENSE_AND_ASSETS.md): upstream assets, licenses, and citations
- [`docs/DEPENDENCY_TRACE.md`](docs/DEPENDENCY_TRACE.md): major code-module map

Please cite/acknowledge OccScanNet, ScanNet, CompleteScanNet/SCFusion, OPUS, SPlatSSC, Depth Anything V2, and RADIO according to their licenses.
