# Dependency trace

Public OccScanNet configs:

- RADIO released/reference baseline: `configs/occscannet/radio_occscannet_mini.py`
- RADIO released full-split companion: `configs/occscannet/radio_occscannet_full.py`
- EfficientNet-B7 additional image encoder option: `configs/occscannet/efficientnet_b7_occscannet_mini.py`

All three configs keep the same AdaOcc detector, TPV branch, decoder, and data pipeline. The two RADIO configs differ only in the data split, the progressive-query schedule, and the default depth mode; the EfficientNet-B7 config differs in the config-selected `model.img_encoder.image_backbone_cfg` image backbone.

Key local modules:

| Component | Files |
| --- | --- |
| Detector | `models/adaocc/adaocc.py` |
| Head/decoder | `models/adaocc/adaocc_head.py`, `models/adaocc/adaocc_transformer.py`, `models/adaocc/adaocc_sampling.py` |
| Depth paths | `models/adaocc/online_depth.py`, `loaders/pipelines/online_depth_inputs.py`, `loaders/pipelines/loading.py` |
| RADIO image encoder | `models/backbones/radiov4_hf_backbone.py`, `models/backbones/modular_occ_encoder.py`, `models/radio/*` |
| EfficientNet-B7 image encoder | `models/backbones/timm_feature_backbone.py`, `models/backbones/modular_occ_encoder.py` |
| TPV branch | `models/lidar_encoder/sparse_encoder_tpv.py`, `models/lidar_encoder/tpv_lite_encoder.py` |
| Dataset | `loaders/occscannet_dataset.py`, `loaders/pipelines/*`, `loaders/metrics/occ3d_metric.py` |
| Asset/docs checks | `scripts/check_assets.py` (`--raw-depth-from-images`, `--online-depth`, `--precomputed-depth`, `--efficientnet-b7`, `--splits`) |
| Checkpoint check | `scripts/check_checkpoint.py` (checkpoint/config parameter parity) |
| Runtime | `train.py`, `val.py`, `dist_train.sh`, `dist_val.sh`, `models/runtime.py` |
| Optional CUDA sampling | `models/csrc/*` |

External packages are pinned in `environment.yml` and `requirements.txt`. The EfficientNet-B7 config additionally relies on the already listed `timm` dependency and the local weight file `pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth` from <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth>; it is not the default config.
