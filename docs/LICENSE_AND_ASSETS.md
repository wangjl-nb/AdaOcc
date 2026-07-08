# License and external assets

The AdaOcc code in this repository is released under the top-level `LICENSE`.

The repository does not redistribute these assets:

- OccScanNet data, generated labels, generated depth maps, mini PKLs
- OPUS fusion pretrain checkpoint
- RADIO weights/cache
- EfficientNet-B7/timm checkpoint (`checkpoints/tf_efficientnet_b7_ns-1dbc32de.pth`)
- DepthAnything/SPlatSSC FT-DaV2 fine-tuned checkpoint
- AdaOcc training checkpoints, logs, TensorBoard events, predictions

Users are responsible for obtaining every external asset under its own license and terms.


## OPUS fusion pretrain

`pretrain/fusion_pretrain_model.pth` may be the AdaOcc HF slim middle-encoder subset (`https://huggingface.co/wjldragon/AdaOcc`) or the full OPUS-generated checkpoint. The slim subset keeps only `pts_middle_encoder.*` for the current public baseline and keeps OPUS/upstream checkpoint attribution.

To generate the full checkpoint with OPUS, download:

- DAL-tiny pretrained weight from `https://huggingface.co/jbwang1997/OPUS`
- the NuImages Cascade Mask R-CNN checkpoint linked in the OPUS README

Put both under `OPUS/pretrain/`, run `python scripts/gen_fusion_pretrain_model.py` in OPUS, then optionally run AdaOcc's `scripts/extract_adaocc_fusion_pretrain.py` to produce the slim subset.

Useful upstream routes:

- OccScanNet dataset: `https://huggingface.co/datasets/hongxiaoy/OccScanNet`
- OPUS: `https://github.com/jbwang1997/OPUS`
- SPlatSSC: `https://github.com/Made-Gpt/SplatSSC`
- Depth Anything V2: `https://github.com/DepthAnything/Depth-Anything-V2`
- RADIO: `https://huggingface.co/nvidia/C-RADIOv3-B`
- timm / EfficientNet-B7 Noisy Student weights for `tf_efficientnet_b7_ns`: `https://github.com/huggingface/pytorch-image-models`
- EmbodiedOcc public FT-DaV2 checkpoint (`finetune_scannet_depthanythingv2.pth`, fine-tuned Depth Anything V2 on OccScanNet): `https://huggingface.co/YkiWu/EmbodiedOcc/blob/main/finetune_scannet_depthanythingv2.pth`
- SPlatSSC uses the same FT-DaV2 checkpoint path in its public configuration: `https://github.com/Made-Gpt/SplatSSC`
