# Architecture notes

## Image path

1. The loader reads the current front-camera image (`CAM_FRONT`).
2. `PackOnlineDepthInputs` saves the original pre-augmentation image and original camera intrinsics/extrinsics for online depth.
3. `RandomTransformImage` later resizes/crops the image for RADIO and updates image metadata.
4. `ModularOccEncoder` wraps RADIO `nvidia/C-RADIOv3-B` and returns one stride-16 feature level.
5. A trainable projection maps RADIO features from `768` to `512` channels.

## Online depth path

When `ADAOCC_ONLINE_DEPTH=1`, AdaOcc builds a frozen `OnlineDepthAnythingPoints` module. It predicts depth from the original image, backprojects with the original camera calibration, converts points into occ space, and feeds the TPV branch. DepthAnything runs under `torch.inference_mode()` and does not receive gradients.

The model can overlap online depth with the RADIO image encoder on CUDA when `online_depth.mode="parallel"`.

## Optional precomputed-depth path

When `ADAOCC_ONLINE_DEPTH=0`, the pipeline reads precomputed depth PNGs and backprojects them before image augmentation. The PNGs are AdaOcc float32-RGBA depth files, not visualization PNGs.

## Decoder / query schedule

The decoder samples one image feature level plus TPV planes. Training query budget grows:

- epochs 1-40: 100 queries
- epochs 41-80: 200 queries
- epochs 81-120: 300 queries
- epochs 121-160: 400 queries
- epochs 161-200: 500 queries

Evaluation uses 500 queries.
