from pathlib import Path

import torch
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


_ALLOWED_UNEXPECTED_PREFIXES = (
    "classifier.",
    "fc.",
    "head.",
    "global_pool.",
    "conv_head.",
    "bn2.",
)
_STRICT_TRUNK_PREFIXES = (
    "conv_stem.",
    "bn1.",
    "blocks.",
)
_PREFIXES_TO_STRIP = ("module.", "model.")


@MODELS.register_module()
class TimmFeatureBackbone(BaseModule):
    """timm feature-map backbone wrapper for AdaOcc image encoders.

    The wrapper intentionally imports timm only during construction so normal
    RADIO imports do not require timm unless an EfficientNet/timm config is used.
    """

    def __init__(
        self,
        model_name,
        checkpoint_path=None,
        pretrained=False,
        features_only=True,
        out_indices=(3,),
        freeze=True,
        strict_checkpoint=True,
        expected_output_stride=None,
        in_channels=None,
        out_channels=None,
        create_model_kwargs=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if not features_only:
            raise ValueError("TimmFeatureBackbone requires features_only=True")
        self.model_name = str(model_name)
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.pretrained = bool(pretrained)
        self.out_indices = tuple(out_indices)
        if len(self.out_indices) == 0:
            raise ValueError("out_indices must be non-empty")
        self.freeze = bool(freeze)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.expected_output_stride = (
            None if expected_output_stride is None else int(expected_output_stride)
        )

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "TimmFeatureBackbone requires the optional dependency 'timm'. "
                "Install the project requirements or use the RADIO config."
            ) from exc

        kwargs = dict(create_model_kwargs or {})
        kwargs.update(
            dict(
                pretrained=self.pretrained,
                features_only=True,
                out_indices=self.out_indices,
            )
        )
        self.model = timm.create_model(self.model_name, **kwargs)
        self._validate_feature_info()
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        selected_channels = self._selected_channels()
        if in_channels is None:
            if len(selected_channels) != 1 and out_channels is not None:
                raise ValueError(
                    "in_channels is required when projecting multiple timm feature levels"
                )
            resolved_in_channels = selected_channels[0]
        else:
            resolved_in_channels = int(in_channels)

        self.output_proj = None
        if out_channels is not None:
            self.output_proj = torch.nn.Conv2d(
                resolved_in_channels,
                int(out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        self._apply_freeze_policy()

    @staticmethod
    def allowed_unexpected_prefixes():
        return _ALLOWED_UNEXPECTED_PREFIXES

    @staticmethod
    def strict_trunk_prefixes():
        return _STRICT_TRUNK_PREFIXES

    def _feature_info(self):
        feature_info = getattr(self.model, "feature_info", None)
        if feature_info is None:
            raise ValueError("timm features_only model does not expose feature_info")
        return feature_info

    def _feature_info_values(self, name):
        feature_info = self._feature_info()
        value = getattr(feature_info, name, None)
        if callable(value):
            return list(value())
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError(f"timm feature_info does not expose {name}()")

    def _selected_channels(self):
        channels = self._feature_info_values("channels")
        if len(channels) != len(self.out_indices):
            raise ValueError(
                f"feature_info channels length {len(channels)} does not match "
                f"out_indices length {len(self.out_indices)}"
            )
        return [int(v) for v in channels]

    def _validate_feature_info(self):
        if self.expected_output_stride is None:
            return
        reductions = [int(v) for v in self._feature_info_values("reduction")]
        if len(reductions) != len(self.out_indices):
            raise ValueError(
                f"feature_info reductions length {len(reductions)} does not match "
                f"out_indices length {len(self.out_indices)}"
            )
        bad = [v for v in reductions if v != self.expected_output_stride]
        if bad:
            raise ValueError(
                f"Selected timm feature reductions {reductions} do not match "
                f"expected_output_stride={self.expected_output_stride}"
            )

    @staticmethod
    def _unwrap_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    @staticmethod
    def _strip_prefixes(state_dict):
        stripped = {}
        for key, value in state_dict.items():
            out_key = str(key)
            changed = True
            while changed:
                changed = False
                for prefix in _PREFIXES_TO_STRIP:
                    if out_key.startswith(prefix):
                        out_key = out_key[len(prefix):]
                        changed = True
            stripped[out_key] = value
        return stripped

    @staticmethod
    def _matches_prefix(name, prefixes):
        return any(str(name).startswith(prefix) for prefix in prefixes)

    def _load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        checkpoint = torch.load(str(path), map_location="cpu")
        state_dict = self._unwrap_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint {path} does not contain a state dict")
        state_dict = self._strip_prefixes(state_dict)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if not self.strict_checkpoint:
            return

        bad_missing = [
            key for key in missing
            if self._matches_prefix(key, _STRICT_TRUNK_PREFIXES)
            or not self._matches_prefix(key, _ALLOWED_UNEXPECTED_PREFIXES)
        ]
        bad_unexpected = [
            key for key in unexpected
            if not self._matches_prefix(key, _ALLOWED_UNEXPECTED_PREFIXES)
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Failed to strictly load timm feature checkpoint "
                f"{path}: missing={bad_missing}, unexpected={bad_unexpected}. "
                f"Allowed unexpected prefixes={_ALLOWED_UNEXPECTED_PREFIXES}"
            )

    def _apply_freeze_policy(self):
        if not self.freeze:
            return
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _project_single_feature(self, features):
        if not isinstance(features, torch.Tensor) or features.dim() != 4:
            raise ValueError(
                "timm feature must be Tensor[N, C, H, W], "
                f'got type={type(features)} shape={getattr(features, "shape", None)}'
            )
        if self.output_proj is None:
            return features
        if not features.is_contiguous():
            features = features.contiguous()
        in_dtype = features.dtype
        proj_dtype = self.output_proj.weight.dtype
        if in_dtype != proj_dtype:
            features = features.to(dtype=proj_dtype)
        features = self.output_proj(features)
        if features.dtype != in_dtype:
            features = features.to(dtype=in_dtype)
        return features

    def forward(self, image_tensor):
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.dim() != 4:
            raise ValueError(
                "image_tensor must be Tensor[N, C, H, W], "
                f'got type={type(image_tensor)} shape={getattr(image_tensor, "shape", None)}'
            )
        if self.freeze:
            with torch.no_grad():
                features = self.model(image_tensor)
        else:
            features = self.model(image_tensor)
        if isinstance(features, torch.Tensor):
            return self._project_single_feature(features)
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError(f"timm model returned unsupported features type={type(features)}")
        projected = [self._project_single_feature(feature) for feature in features]
        return projected[0] if len(projected) == 1 else projected
