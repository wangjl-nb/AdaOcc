# Installation

Tested stack:

- Linux x86_64
- Python 3.10
- PyTorch 2.2.0 + CUDA 12.1
- MMCV 2.1.0 / MMEngine 0.10.7 / MMDetection 3.3.0 / MMDetection3D 1.4.0

```bash
conda env create -f environment.yml
conda activate AdaOcc
```

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8

python -m pip install "setuptools<81" wheel "numpy==1.26.4" "packaging==24.2" "PyYAML>=6.0" ninja
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

`mmcv==2.1.0` may build from source. Keep `setuptools<81` because older OpenMMLab build code imports `pkg_resources`.

## MSMV CUDA extension

The custom MSMV sampling extension is optional for the released single-level RADIO baseline. If it is not compiled, imports print a warning like:

```text
No module named 'models.csrc._msmv_sampling_cuda'
```

This is expected when using the PyTorch fallback. Keep fallback enabled with:

```bash
export ADAOCC_DISABLE_MSMV_CUDA=1
```

If you want the fused CUDA extension, build it after installing requirements. Use the CUDA toolkit that matches PyTorch (`torch.version.cuda`); for the provided conda environment this means `CUDA_HOME=$CONDA_PREFIX`, not a system CUDA 13.x install.

```bash
conda activate AdaOcc
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0"   # H100/H20; adjust for other GPUs.
export MAX_JOBS=8

cd models/csrc
python setup.py build_ext --inplace
cd ../..
python - <<'PY'
from models.csrc.wrapper import MSMV_CUDA
print('MSMV_CUDA =', MSMV_CUDA)
assert MSMV_CUDA
PY
```

If compilation fails with a CUDA mismatch such as `detected CUDA version (13.x) mismatches ... PyTorch (12.1)`, reset `CUDA_HOME/CUDA_PATH/PATH` to the conda environment as above and rebuild. If it still fails, leave `ADAOCC_DISABLE_MSMV_CUDA=1`. If compilation succeeds and you want `dist_train.sh` / `dist_val.sh` to use fused MSMV, set `ADAOCC_DISABLE_MSMV_CUDA=0`; smoke, train, and eval should all use the same MSMV policy.
