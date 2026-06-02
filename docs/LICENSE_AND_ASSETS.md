# License and external assets

The AdaOcc code in this repository is released under the top-level `LICENSE`.

The repository does not redistribute these assets:

- OccScanNet data, generated labels, generated depth maps, mini PKLs
- OPUS fusion pretrain checkpoint
- RADIO weights/cache
- DepthAnything/SPlatSSC FT-DaV2 fine-tuned checkpoint
- AdaOcc training checkpoints, logs, TensorBoard events, predictions

Users are responsible for obtaining every external asset under its own license and terms.


## OPUS fusion pretrain

`pretrain/fusion_pretrain_model.pth` is generated outside AdaOcc using OPUS. In OPUS, download:

- DAL-tiny pretrained weight from `https://huggingface.co/jbwang1997/OPUS`
- the NuImages Cascade Mask R-CNN checkpoint linked in the OPUS README

Put both under `OPUS/pretrain/`, run `python scripts/gen_fusion_pretrain_model.py` in OPUS, then place the generated `fusion_pretrain_model.pth` under `AdaOcc/pretrain/`.

Useful upstream routes:

- OccScanNet dataset: `https://huggingface.co/datasets/hongxiaoy/OccScanNet`
- OPUS: `https://github.com/jbwang1997/OPUS`
- SPlatSSC: `https://github.com/Made-Gpt/SplatSSC`
- Depth Anything V2: `https://github.com/DepthAnything/Depth-Anything-V2`
- RADIO: `https://huggingface.co/nvidia/C-RADIOv3-B`
- SPlatSSC public FT-DaV2 depth-branch weights: `https://github.com/Made-Gpt/SplatSSC`
