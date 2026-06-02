# AdaOcc

AdaOcc is a compact reproduction of the **OccScanNet-mini + RADIO** occupancy baseline. The default public path trains with **online Depth-Anything-V2** inside the model and keeps the older **precomputed-depth PNG** path as an option.

This repository contains code, configuration, documentation, and reproducibility scripts only. It does **not** redistribute OccScanNet data, generated labels/depth maps, pretrained weights, checkpoints, logs, or private experiment outputs.

## Baseline

- Dataset: OccScanNet-mini, `CAM_FRONT` only
- Image encoder: Hugging Face RADIO `nvidia/C-RADIOv3-B`
- Image features: one stride-16 level, `768 -> 512` projection
- Geometry branch: TPV enabled, point-feature branch disabled
- Queries: progressive `100 -> 500`, +100 every 40 epochs, 200 epochs total
- Default depth: online frozen Depth-Anything-V2 with the OccScanNet FT-DaV2 fine-tuned checkpoint used by SPlatSSC/EmbodiedOcc
- Optional depth: precomputed SPlatSSC-stage1-ftDAV2 float32-RGBA PNGs

Reference online-depth epoch-200 metrics on OccScanNet-mini:

| metric | value |
| --- | ---: |
| `occ/mIoU` | 58.49 |
| `occ/IoU` | 65.49 |
| `occ/mIoU_small` | 45.68 |
| `occ/mIoU_head` | 61.34 |

Primary reproduction tolerance: about ±0.5 for `mIoU` and `IoU`.

## Repository map

- Configs: `configs/adaocc/radio_occscannet_mini.py`, `configs/adaocc/radio_occscannet_mini_smoke.py`
- Training/eval: `train.py`, `val.py`, `dist_train.sh`, `dist_val.sh`
- Online depth: `models/adaocc/online_depth.py`, `loaders/pipelines/online_depth_inputs.py`
- Data scripts: `scripts/check_assets.py`, `scripts/generate_occscannet_mini_pkls.py`, `scripts/generate_occscannet_mini_gts_camvisbits.py`, `scripts/generate_occscannet_mini_depth_da_v2.py`
- Utility scripts: `scripts/link_local_assets.sh`, `scripts/check_loss_scale.py`
- Docs: `docs/INSTALL.md`, `docs/DATA.md`, `docs/REPRODUCIBILITY.md`, `docs/LICENSE_AND_ASSETS.md`, `docs/ARCHITECTURE.md`

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H20; change for other GPUs.
export MAX_JOBS=8
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

See `docs/INSTALL.md` for notes on MMCV builds and the optional MSMV CUDA extension. If MSMV is not compiled you may see `No module named models.csrc._msmv_sampling_cuda`; the single-level RADIO baseline can run with `ADAOCC_DISABLE_MSMV_CUDA=1` and will use the PyTorch fallback.

Optional fused MSMV build:

```bash
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H20; adjust for other GPUs.
cd models/csrc && python setup.py build_ext --inplace && cd ../..
python - <<'PY'
from models.csrc.wrapper import MSMV_CUDA
print('MSMV_CUDA =', MSMV_CUDA)
PY
```

If it fails, keep `ADAOCC_DISABLE_MSMV_CUDA=1` in `configs/local_paths.sh` and continue with the fallback. If it succeeds and you want to use fused MSMV in `dist_train.sh` / `dist_val.sh`, set `ADAOCC_DISABLE_MSMV_CUDA=0`.

## 2. Prepare data and weights

Prepare/download OccScanNet data according to its terms. Put or symlink the processed mini data under `data/OccScanNet`.

Required user-prepared assets:

| asset | original link | how to obtain | repo-local target |
| --- | --- | --- | --- |
| OccScanNet data | [hongxiaoy/OccScanNet](https://huggingface.co/datasets/hongxiaoy/OccScanNet) | Download/prepare OccScanNet according to its terms; see `docs/DATA.md`. | `data/OccScanNet/` |
| RADIO | [nvidia/C-RADIOv3-B](https://huggingface.co/nvidia/C-RADIOv3-B) | Download through Hugging Face cache or snapshot locally. | `pretrain/radio/C-RADIOv3-B/` or `pretrain/huggingface/hub/models--nvidia--C-RADIOv3-B/` |
| SPlatSSC FT-DaV2 depth checkpoint | [SPlatSSC](https://github.com/Made-Gpt/SplatSSC), [HF mirror](https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth) | Download `finetune_scannet_depthanythingv2.pth` from the HF mirror (same FT-DaV2/Depth-Anything-V2 checkpoint used by the OccScanNet setup) and place/symlink it here. | `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth` |
| OPUS fusion pretrain | [jbwang1997/OPUS](https://github.com/jbwang1997/OPUS), [OPUS HF weights](https://huggingface.co/jbwang1997/OPUS), [generation script](https://github.com/jbwang1997/OPUS/blob/main/scripts/gen_fusion_pretrain_model.py) | Prepare `fusion_pretrain_model.pth` in OPUS from DAL-tiny + NuImages Cascade Mask R-CNN; see steps below. | `pretrain/fusion_pretrain_model.pth` |

### OPUS `fusion_pretrain_model.pth`

AdaOcc only uses the OPUS fusion pretrain as an initialization checkpoint; this repository does not redistribute it. OPUS documents the generation path:

1. Clone OPUS: `https://github.com/jbwang1997/OPUS`.
2. Download OPUS DAL-tiny pretrained weight from [OPUS HF weights](https://huggingface.co/jbwang1997/OPUS), usually named `dal-tiny-map66.9-nds71.1.pth`.
3. Download the [NuImages Cascade Mask R-CNN checkpoint](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) referenced by OPUS, usually named `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`.
4. Put both files under `OPUS/pretrain/`.
5. In the OPUS repo, run:

   ```bash
   python scripts/gen_fusion_pretrain_model.py
   ```

6. Copy or symlink the generated `OPUS/pretrain/fusion_pretrain_model.pth` to:

   ```text
   AdaOcc/pretrain/fusion_pretrain_model.pth
   ```

AdaOcc loads only matching modules from this checkpoint; old OPUS image-backbone keys that do not match RADIO are expected to be skipped.

Create a private local config file and edit paths once:

```bash
cp configs/local_paths.example.sh configs/local_paths.sh
${EDITOR:-nano} configs/local_paths.sh
```

`configs/local_paths.sh` is ignored by git. `dist_train.sh`, `dist_val.sh`, and `scripts/link_local_assets.sh` source it automatically. If you want to keep the config elsewhere, set `ADAOCC_LOCAL_CONFIG=/path/to/local_paths.sh` for that command.

After editing `ADAOCC_LINK_*_SRC` in `configs/local_paths.sh`, create repo-local symlinks:

```bash
scripts/link_local_assets.sh
```

If any required asset is missing, do not guess a replacement; report the exact missing file/path.

## 3. Generate/verify PKLs, labels, and optional depth

If your OccScanNet root has `train_subscenes.txt`, `val_subscenes.txt`, and `gathered_data/`, generate the mini annotation PKLs first. The OccScanNet-mini sample counts follow the ISO reference setup: [`iso_occscannet_mini.yaml`](https://github.com/hongxiaoy/ISO/blob/main/iso/config/iso_occscannet_mini.yaml) selects `OccScanNet_mini`, and [`train_iso.py`](https://github.com/hongxiaoy/ISO/blob/main/iso/scripts/train_iso.py) uses `train_scenes_sample=4639` and `val_scenes_sample=2007`.

```bash
python scripts/generate_occscannet_mini_pkls.py \
  --data-root data/OccScanNet \
  --overwrite
```

PKLs are indexes, not full data. `labels.npz` is generated from `gathered_data/*.pkl`:

```bash
python scripts/generate_occscannet_mini_gts_camvisbits.py \
  --data-root data/OccScanNet \
  --overwrite

python scripts/generate_occscannet_mini_gts_camvisbits.py \
  --data-root data/OccScanNet \
  --verify-only
```

Default online-depth training does not require precomputed depth PNGs. To create/verify the optional precomputed-depth tree:

```bash
python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --weights pretrain/depth_anything/finetune_scannet_depthanythingv2.pth

python scripts/generate_occscannet_mini_depth_da_v2.py \
  --data-root data/OccScanNet \
  --verify-only
```

Depth PNGs are not visual PNGs. They store `float32` depth meters as little-endian bytes in an `H x W x 4 uint8` PNG. Do not open/re-save them with image editors. The asset checker validates depth PNG structure with Pillow; training still reads them with the AdaOcc/OpenCV loader.

Run the asset checker:

```bash
python scripts/check_assets.py \
  --data-root data/OccScanNet \
  --pretrain-root pretrain \
  --online-depth
```

Use `--precomputed-depth --verify-depth-png` when validating the optional precomputed-depth mode.

## 4. Check the final local layout

After generating PKLs/labels and linking weights, the ignored runtime tree should look like this:

```text
AdaOcc/
├── configs/local_paths.sh                         # private, git-ignored
├── data/OccScanNet/
│   ├── train_occscannet_mini.pkl
│   ├── val_occscannet_mini.pkl
│   ├── test_occscannet_mini.pkl
│   ├── gathered_data/<scene>/<frame>.pkl
│   ├── posed_images/<scene>/<frame>.jpg
│   ├── gts_camvisbits/<scene>/<frame>/labels.npz
│   └── depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png  # optional
├── pretrain/
│   ├── fusion_pretrain_model.pth
│   ├── depth_anything/
│   │   └── finetune_scannet_depthanythingv2.pth
│   ├── radio/
│   │   └── C-RADIOv3-B/                         # RADIO local snapshot option
│   └── huggingface/
│       └── hub/models--nvidia--C-RADIOv3-B/     # RADIO HF cache option
└── outputs/
```

`pretrain/radio/C-RADIOv3-B/` and `pretrain/huggingface/hub/models--nvidia--C-RADIOv3-B/` are alternative RADIO layouts. `scripts/link_local_assets.sh` creates the parent directories for both layouts, so empty `radio/` or `huggingface/` directories are normal during setup.

For precomputed-depth training, set `ADAOCC_ONLINE_DEPTH=0` in `configs/local_paths.sh` and ensure the depth PNG tree exists.

## 5. Smoke checks

Import/config check:

```bash
python - <<'PY'
import torch, mmcv, mmengine, mmdet, mmdet3d, spconv, transformers
from mmengine.config import Config
cfg = Config.fromfile('configs/adaocc/radio_occscannet_mini.py')
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())
print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__)
print('mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__)
print('model', cfg.model.type, 'dataset', cfg.dataset_type, 'online_depth', cfg.model.online_depth.enabled)
PY
```

One-epoch train+val smoke. This uses the same mini PKLs/data as the full run and only shortens the epoch count:

```bash
ADAOCC_RUN_LABEL=smoke-1epoch \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini_smoke.py
```

Then check the training log:

```bash
latest_log=$(find outputs/AdaOcc -name '*.log' | sort | tail -n 1)
python scripts/check_loss_scale.py "$latest_log" --first-n 20
```

A healthy first stage should show `train_runtime_num_query: 100`, finite `loss_containment`, and early total loss roughly in the `5-8` range.

## 6. Full training and eval

```bash
ADAOCC_RUN_LABEL=online-depth-mini \
./dist_train.sh 8 configs/adaocc/radio_occscannet_mini.py
```

Evaluate the final checkpoint:

```bash
./dist_val.sh 8 configs/adaocc/radio_occscannet_mini.py /path/to/epoch_200.pth
```

Record the command, environment, checkpoint path, and final metrics. The target online-depth result is `mIoU≈58.49`, `IoU≈65.49`.

## Citations / related projects

Please cite/acknowledge the relevant upstream projects and datasets according to their licenses: OccScanNet, ScanNet, CompleteScanNet/SCFusion if used by your data preparation, OPUS, SPlatSSC, Depth Anything V2, and RADIO.
