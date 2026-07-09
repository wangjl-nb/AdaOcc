# Installation

Tested stack:

- Linux x86_64
- Python 3.10
- PyTorch 2.2.0 + CUDA 12.1
- MMCV 2.1.0 / MMEngine 0.10.7 / MMDetection 3.3.0 / MMDetection3D 1.4.0

## Conda environment

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

`mmcv==2.1.0` may build from source. Keep `setuptools<81` because older OpenMMLab build code imports `pkg_resources`.

Quick version check:

```bash
python - <<'PY'
import torch, mmcv, mmengine, mmdet, mmdet3d, spconv, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__)
print('mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__)
PY
```

## Optional MSMV CUDA extension

The custom MSMV sampling extension is optional for the public single-level image-encoder configs. If it is not compiled, imports may print a warning like:

```text
No module named 'models.csrc._msmv_sampling_cuda'
```

This is expected when using the PyTorch fallback. Keep fallback enabled by prefixing commands with:

```bash
ADAOCC_DISABLE_MSMV_CUDA=1 ./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py --run-label smoke-radio
```

If you want the fused CUDA extension, build it after installing requirements. Use the CUDA toolkit that matches PyTorch (`torch.version.cuda`); for the provided conda environment this usually means `CUDA_HOME=$CONDA_PREFIX`, not a newer system CUDA install. Set `TORCH_CUDA_ARCH_LIST` for your GPU, for example `9.0` on H100/H20:

```bash
cd models/csrc
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  python setup.py build_ext --inplace
cd ../..
ADAOCC_DISABLE_MSMV_CUDA=0 python - <<'PY'
from models.csrc.wrapper import MSMV_CUDA
print('MSMV_CUDA =', MSMV_CUDA)
assert MSMV_CUDA
PY
```

If compilation fails with a CUDA mismatch such as `detected CUDA version (13.x) mismatches ... PyTorch (12.1)`, use the conda CUDA toolkit matching PyTorch or keep `ADAOCC_DISABLE_MSMV_CUDA=1`. Smoke, train, and eval should all use the same MSMV policy.
