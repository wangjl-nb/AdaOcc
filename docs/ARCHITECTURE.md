# Architecture notes

## Image path

AdaOcc uses a config-selected image encoder under `model.img_encoder.image_backbone_cfg`:

1. The loader reads the current front-camera image (`CAM_FRONT`).
2. `RandomTransformImage` resizes/crops the image and updates image metadata.
3. `ModularOccEncoder` wraps the selected image backbone and returns one stride-16 feature level.
4. A trainable projection maps the selected image features to the 512-channel occupancy feature space.

Public OccScanNet image-encoder configs:

| config | image backbone | checkpoint/cache | backbone training policy |
| --- | --- | --- | --- |
| `configs/occscannet/radio_occscannet_mini.py` | `RADIOHFBackbone` for `nvidia/C-RADIOv3-B` | `pretrain/radio/C-RADIOv3-B/` | RADIO is mostly frozen, with `unfreeze_last_n_blocks=4` so the final RADIO blocks are trainable. |
| `configs/occscannet/radio_occscannet_full.py` | `RADIOHFBackbone` for `nvidia/C-RADIOv3-B` | `pretrain/radio/C-RADIOv3-B/` | Same architecture as the mini baseline; only the data split, query schedule, and default depth mode differ, so the released full-split checkpoint evaluates directly. |
| `configs/occscannet/efficientnet_b7_occscannet_mini.py` | `TimmFeatureBackbone` for `tf_efficientnet_b7_ns` | `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` | EfficientNet-B7 backbone weights are frozen. |

RADIO remains the released/reference baseline. EfficientNet-B7 is an additional config option that swaps the image backbone path; it does not rewrite the decoder, TPV branch, or query schedule. `radio_occscannet_full.py` is the full-split companion of the RADIO mini baseline and keeps the same model as `radio_occscannet_mini.py`.

## Depth paths

Default local/raw depth uses same-stem raw uint depth PNGs next to RGB frames:

```text
posed_images/<scene>/<frame>.jpg
posed_images/<scene>/<frame>.png
```

The public configs default to `ADAOCC_ONLINE_DEPTH=0` and `ADAOCC_RAW_DEPTH_FROM_IMAGES=1`, so the pipeline reads local raw depth and backprojects it before image augmentation. The exception is `configs/occscannet/radio_occscannet_full.py`, which defaults to `ADAOCC_RAW_DEPTH_FROM_IMAGES=0` because the released full-split run used the generated precomputed depth stored in the PKL.

When `ADAOCC_ONLINE_DEPTH=1`, AdaOcc builds a frozen `OnlineDepthAnythingPoints` module. `PackOnlineDepthInputs` saves the original pre-augmentation image and original camera intrinsics/extrinsics; Depth-Anything predicts depth from that original image, backprojects with the original calibration, converts points into occ space, and feeds the TPV branch. Depth-Anything runs under `torch.inference_mode()` and does not receive gradients.

When online depth is disabled and `ADAOCC_RAW_DEPTH_FROM_IMAGES=0`, the pipeline reads generated precomputed depth PNGs from the pkl `depth_path`, normally the SPlatSSC-style FT-DaV2 tree `depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png`. These PNGs are AdaOcc float32-RGBA depth containers: each metric-depth float32 is stored as little-endian bytes in the four PNG channels, not as a visualized depth image. The generator uses `pretrain/depth_anything/finetune_scannet_depthanythingv2.pth`, the same public FT-DaV2 checkpoint noted for SPlatSSC alignment.

The model can overlap online depth with the image encoder on CUDA when `online_depth.mode="parallel"`.

## Decoder / query schedule

The decoder samples one image feature level plus TPV planes. The mini baseline (`radio_occscannet_mini.py`) grows the training query budget over 200 epochs:

- epochs 1-40: 100 queries
- epochs 41-80: 200 queries
- epochs 81-120: 300 queries
- epochs 121-160: 400 queries
- epochs 161-200: 500 queries

The full-split companion (`radio_occscannet_full.py`) grows it over 100 epochs:

- epochs 1-25: 200 queries
- epochs 26-50: 300 queries
- epochs 51-75: 400 queries
- epochs 76-100: 500 queries

Evaluation uses 500 queries for both configs.
