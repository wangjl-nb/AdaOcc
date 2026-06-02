import importlib
import os


_LIDAR_ENCODER_MODULES = {
    'SparseEncoderTPVOnly': 'sparse_encoder_tpv',
    'TPVLiteEncoder': 'tpv_lite_encoder',
}

_LIGHTWEIGHT_IMPORT = str(
    os.environ.get('ADAOCC_IMPORT_LIGHTWEIGHT', '')).lower() in ('1', 'true', 'yes', 'on')
_EAGER_IMPORTS = [
    'SparseEncoderTPVOnly',
    'TPVLiteEncoder',
]
if not _LIGHTWEIGHT_IMPORT:
    _EAGER_IMPORTS = list(_LIDAR_ENCODER_MODULES.keys())


def _load_lidar_encoder(name):
    module_name = _LIDAR_ENCODER_MODULES[name]
    module = importlib.import_module(f'.{module_name}', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


for _name in _EAGER_IMPORTS:
    _load_lidar_encoder(_name)


def __getattr__(name):
    if name in _LIDAR_ENCODER_MODULES:
        return _load_lidar_encoder(name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    'SparseEncoderTPVOnly',
    'TPVLiteEncoder',
]
