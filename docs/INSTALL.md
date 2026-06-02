# Installation

Tested stack:

- Linux x86_64
- Python 3.10
- PyTorch 2.2.0 + CUDA 12.1
- MMCV 2.1.0 / MMEngine 0.10.7 / MMDetection 3.3.0 / MMDetection3D 1.4.0

```bash
conda env create -f environment.yml
conda activate AdaOcc
python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

`mmcv==2.1.0` may build from source. Keep `setuptools<81` because older OpenMMLab build code imports `pkg_resources`.

## MSMV CUDA extension

Build if you need the custom sampling extension:

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
python - <<'PY'
from models.csrc.wrapper import MSMV_CUDA
print('MSMV_CUDA =', MSMV_CUDA)
PY
```

The released single-level RADIO baseline can run with the PyTorch fallback:

```bash
export ADAOCC_DISABLE_MSMV_CUDA=1
```

Use the same MSMV policy for smoke, train, and eval.
