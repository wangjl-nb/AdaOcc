# Dependency trace

Active public config: `configs/occscannet/radio_occscannet_mini.py`.

Key local modules:

| Component | Files |
| --- | --- |
| Detector | `models/adaocc/adaocc.py` |
| Head/decoder | `models/adaocc/adaocc_head.py`, `models/adaocc/adaocc_transformer.py`, `models/adaocc/adaocc_sampling.py` |
| Online depth | `models/adaocc/online_depth.py`, `loaders/pipelines/online_depth_inputs.py` |
| RADIO wrapper | `models/backbones/radiov4_hf_backbone.py`, `models/backbones/modular_occ_encoder.py`, `models/radio/*` |
| TPV branch | `models/lidar_encoder/sparse_encoder_tpv.py`, `models/lidar_encoder/tpv_lite_encoder.py` |
| Dataset | `loaders/occscannet_dataset.py`, `loaders/pipelines/*`, `loaders/metrics/occ3d_metric.py` |
| Runtime | `train.py`, `val.py`, `dist_train.sh`, `dist_val.sh`, `models/runtime.py` |
| Optional CUDA sampling | `models/csrc/*` |

External packages are pinned in `environment.yml` and `requirements.txt`.
