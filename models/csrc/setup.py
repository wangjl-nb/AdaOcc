import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension


DEFAULT_CUDA_ARCH_LIST = ['8.0', '8.6', '9.0+PTX']


def _arch_sort_key(arch):
    arch = arch.replace('+PTX', '')
    major, minor = arch.split('.')
    return int(major), int(minor)


def _normalize_arch_token(token):
    token = token.strip()
    if not token:
        return None

    suffix = ''
    if token.endswith('+PTX'):
        token = token[:-4]
        suffix = '+PTX'

    token = token.lower().replace('sm_', '').replace('compute_', '')
    if '.' not in token:
        if len(token) < 2:
            raise ValueError(f'Invalid CUDA arch token: {token!r}')
        token = f'{token[:-1]}.{token[-1]}'
    return f'{token}{suffix}'


def _split_arch_list(spec):
    arches = []
    for token in spec.replace(',', ';').replace(' ', ';').split(';'):
        normalized = _normalize_arch_token(token)
        if normalized:
            arches.append(normalized)
    return arches


def _detect_cuda_arch_list():
    env_arch_list = (
        os.getenv('MSMV_CUDA_ARCH_LIST')
        or os.getenv('TORCH_CUDA_ARCH_LIST')
    )
    if env_arch_list:
        return _split_arch_list(env_arch_list)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        arches = sorted({
            f'{major}.{minor}'
            for major, minor in (
                torch.cuda.get_device_capability(device_idx)
                for device_idx in range(torch.cuda.device_count())
            )
        }, key=_arch_sort_key)
        arches[-1] = f'{arches[-1]}+PTX'
        return arches

    return list(DEFAULT_CUDA_ARCH_LIST)


def get_nvcc_flags():
    arches = _detect_cuda_arch_list()
    flags = []
    for arch in arches:
        ptx = arch.endswith('+PTX')
        arch_number = arch.replace('.', '').replace('+PTX', '')
        flags.append(f'-gencode=arch=compute_{arch_number},code=sm_{arch_number}')
        if ptx:
            flags.append(f'-gencode=arch=compute_{arch_number},code=compute_{arch_number}')
    print(f'Building MSMV CUDA extension for arch list: {arches}')
    return flags


def _dedupe_keep_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def get_cuda_library_dirs():
    candidate_roots = _dedupe_keep_order([
        os.getenv('CUDA_HOME'),
        os.getenv('CUDA_PATH'),
        CUDA_HOME,
        os.getenv('CONDA_PREFIX'),
        *glob.glob('/usr/local/cuda*'),
    ])
    library_dirs = []
    for root in candidate_roots:
        for rel_dir in ('lib64', 'lib', 'targets/x86_64-linux/lib'):
            lib_dir = os.path.join(root, rel_dir)
            if os.path.exists(os.path.join(lib_dir, 'libcudart.so')):
                library_dirs.append(lib_dir)
    library_dirs = _dedupe_keep_order(library_dirs)
    print(f'Using CUDA library dirs: {library_dirs}')
    return library_dirs


def get_ext_modules():
    return [
        CUDAExtension(
            name='_msmv_sampling_cuda',
            sources=[
                'msmv_sampling/msmv_sampling.cpp',
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu'
            ],
            include_dirs=['msmv_sampling'],
            library_dirs=get_cuda_library_dirs(),
            extra_compile_args=dict(
                nvcc=get_nvcc_flags()
            )
        )
    ]


if __name__ == '__main__':
    setup(
        name='csrc',
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': BuildExtension},
    )
