import importlib
import os


_BACKBONE_MODULES = {
    'ModularOccEncoder': 'modular_occ_encoder',
    'RADIOHFBackbone': 'radiov4_hf_backbone',
    'RADIOv4HFBackbone': 'radiov4_hf_backbone',
    'TimmFeatureBackbone': 'timm_feature_backbone',
}
_LAZY_ONLY_BACKBONES = {'TimmFeatureBackbone'}

_LIGHTWEIGHT_IMPORT = str(
    os.environ.get('ADAOCC_IMPORT_LIGHTWEIGHT', '')).lower() in ('1', 'true', 'yes', 'on')
_EAGER_IMPORTS = ['ModularOccEncoder', 'RADIOHFBackbone', 'RADIOv4HFBackbone']
if not _LIGHTWEIGHT_IMPORT:
    _EAGER_IMPORTS = [
        name for name in _BACKBONE_MODULES
        if name not in _LAZY_ONLY_BACKBONES
    ]


def _load_backbone(name):
    module_name = _BACKBONE_MODULES[name]
    module = importlib.import_module(f'.{module_name}', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


for _name in _EAGER_IMPORTS:
    _load_backbone(_name)


def __getattr__(name):
    if name in _BACKBONE_MODULES:
        return _load_backbone(name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    'ModularOccEncoder',
    'RADIOHFBackbone',
    'RADIOv4HFBackbone',
    'TimmFeatureBackbone',
]
