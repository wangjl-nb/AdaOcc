# Environment setup

Use `docs/INSTALL.md` for the installation contract and `README.md` for the end-to-end reproduction sequence.

Short form:

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

Then follow README sections 2-6 for assets, checks, training, and eval.
