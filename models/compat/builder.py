from mmdet3d.registry import MODELS as MMDET3D_MODELS

try:
    from mmdet.registry import MODELS as MMDET_MODELS
except Exception:  # pragma: no cover
    MMDET_MODELS = None


def _build_with_fallback(cfg):
    if cfg is None:
        return None
    try:
        return MMDET3D_MODELS.build(cfg)
    except Exception:
        if MMDET_MODELS is None:
            raise
        return MMDET_MODELS.build(cfg)


def build_transformer(cfg):
    return _build_with_fallback(cfg)


def build_loss(cfg):
    return _build_with_fallback(cfg)
