# AdaOcc

AdaOcc is an OccScanNet-mini occupancy reproduction with **RADIO** image features and **online Depth-Anything-V2** depth-to-points. The default public path predicts depth inside the model; precomputed depth PNGs are still supported as an optional data path.

This repo contains code and docs only. It does not redistribute OccScanNet data, generated labels/depth maps, pretrained weights, checkpoints, logs, or private artifacts.

## Released result and checkpoints

Checkpoints: <https://huggingface.co/wjldragon/AdaOcc>

| file | use |
| --- | --- |
| `pretrain/fusion_pretrain_model.pth` | slim OPUS-derived initializer for the current TPV/sparse encoder path |
| `checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth` | trained online-depth OccScanNet-mini checkpoint |
| `configs/radio_occscannet_mini_training_snapshot.py` | config snapshot from the released run |
| `logs/online_depth_occscannet_mini_epoch200.log` | training/eval log |

Released epoch-200 validation result:

| metric | value |
| --- | ---: |
| `occ/mIoU` | 58.49 |
| `occ/IoU` | 65.49 |
| `occ/mIoU_small` | 45.68 |
| `occ/mIoU_head` | 61.34 |

Expected reproduction tolerance is about ±0.5 for `mIoU` / `IoU`.

## 1. Prepare the fixed layout

AdaOcc configs use repo-relative paths. Arrange local assets under this ignored tree:

```text
AdaOcc/
├── data/OccScanNet/
│   ├── train_occscannet_mini.pkl
│   ├── val_occscannet_mini.pkl
│   ├── test_occscannet_mini.pkl
│   ├── train_subscenes.txt                         # only needed to regenerate PKLs
│   ├── val_subscenes.txt                           # only needed to regenerate PKLs
│   ├── gathered_data/<scene>/<frame>.pkl
│   ├── posed_images/<scene>/<frame>.jpg
│   ├── gts_camvisbits/<scene>/<frame>/labels.npz
│   └── depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png  # optional
├── pretrain/
│   ├── fusion_pretrain_model.pth
│   ├── depth_anything/finetune_scannet_depthanythingv2.pth
│   └── radio/C-RADIOv3-B/
└── outputs/
```

Required assets:

| asset | source | target |
| --- | --- | --- |
| OccScanNet | <https://huggingface.co/datasets/hongxiaoy/OccScanNet> | `data/OccScanNet/` |
| RADIO | <https://huggingface.co/nvidia/C-RADIOv3-B> | `pretrain/radio/C-RADIOv3-B/` |
| Depth-Anything FT checkpoint | <https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth>; also used by <https://github.com/Made-Gpt/SplatSSC> as FT-DaV2 | `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` |
| AdaOcc/OPUS fusion pretrain | <https://huggingface.co/wjldragon/AdaOcc/blob/main/pretrain/fusion_pretrain_model.pth> or full OPUS pretrain from <https://github.com/jbwang1997/OPUS> | `pretrain/fusion_pretrain_model.pth` |

Download released AdaOcc files:

```bash
hf download wjldragon/AdaOcc pretrain/fusion_pretrain_model.pth --local-dir .
hf download wjldragon/AdaOcc checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth --local-dir .
```

Download RADIO locally:

```bash
hf download nvidia/C-RADIOv3-B --local-dir pretrain/radio/C-RADIOv3-B
```

`fusion_pretrain_model.pth` can be the slim AdaOcc HF file or the full OPUS-generated file. The slim file keeps only `pts_middle_encoder.*`, which is what the public config uses because `enable_pts_feature_branch=False`.

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H20; change for other GPUs
export MAX_JOBS=8
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

The custom MSMV CUDA extension is optional for this single-level RADIO baseline. If it is unavailable, keep:

```bash
export ADAOCC_DISABLE_MSMV_CUDA=1
```

See `docs/INSTALL.md` for MMCV/MSMV build notes.

## 3. Generate data files

Generate mini PKLs from `train_subscenes.txt` / `val_subscenes.txt`:

```bash
python scripts/generate_occscannet_mini_pkls.py --data-root data/OccScanNet --overwrite
```

Generate and verify `labels.npz` supervision from `gathered_data/*.pkl`:

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --overwrite
python scripts/generate_occscannet_mini_gts_camvisbits.py --data-root data/OccScanNet --verify-only
```

Default online-depth training does not need precomputed depth PNGs. To generate the optional precomputed-depth tree:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth
python scripts/generate_occscannet_mini_depth_da_v2.py --data-root data/OccScanNet --verify-only
```

Precomputed depth PNGs are binary depth containers, not visual images: each pixel is one little-endian `float32` meter value stored as four uint8 PNG channels (`H x W x 4`). Do not convert, resize, color-map, or re-save them with image editors.

Check assets:

```bash
python scripts/check_assets.py --online-depth
```

Use `--precomputed-depth --verify-depth-png` for the optional precomputed-depth mode.

## 4. Smoke, train, evaluate

Import/config smoke:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 python - <<'PY'
import torch, mmcv, mmengine, mmdet, mmdet3d, spconv, transformers
from mmengine.config import Config
cfg = Config.fromfile('configs/adaocc/radio_occscannet_mini.py')
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())
print('model', cfg.model.type, 'online_depth', cfg.model.online_depth.enabled)
PY
```

One-epoch train+val smoke:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
ADAOCC_RUN_LABEL=smoke-1epoch \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini_smoke.py
```

Full training:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
ADAOCC_RUN_LABEL=online-depth-mini \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py
```

Evaluate the released checkpoint or your final checkpoint:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 \
./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py \
  checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth
```

For optional precomputed-depth training, add `ADAOCC_ONLINE_DEPTH=0` and make sure `depth_splatssc_stage1_ftdav2_vitb_20m_full/` exists.

## More details

- `docs/DATA.md`: label fields, PKLs, and depth PNG format
- `docs/REPRODUCIBILITY.md`: smoke/full reproduction checklist
- `docs/INSTALL.md`: environment and CUDA extension notes
- `docs/LICENSE_AND_ASSETS.md`: upstream assets and citations

Please cite/acknowledge OccScanNet, ScanNet, CompleteScanNet/SCFusion if used by your data preparation, OPUS, SPlatSSC, Depth Anything V2, and RADIO according to their licenses.
