import shutil
from pathlib import Path

import torch
from mmengine.model import BaseModule
from transformers import AutoModel

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RADIOHFBackbone(BaseModule):
    """Thin Hugging Face RADIO backbone wrapper returning NCHW spatial features."""

    def __init__(
        self,
        model_id_or_path="nvidia/C-RADIOv4-SO400M",
        from_pretrained_kwargs=None,
        freeze=True,
        unfreeze_last_n_blocks=0,
        in_channels=None,
        out_channels=None,
        intermediate_indices=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_id_or_path = model_id_or_path
        self.from_pretrained_kwargs = dict(from_pretrained_kwargs or {})
        self.from_pretrained_kwargs.setdefault("trust_remote_code", True)
        self.from_pretrained_kwargs.setdefault("local_files_only", True)
        self.freeze = bool(freeze)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        if self.unfreeze_last_n_blocks < 0:
            raise ValueError(
                f'unfreeze_last_n_blocks must be non-negative, got {self.unfreeze_last_n_blocks}')
        self.in_channels = None if in_channels is None else int(in_channels)
        self.intermediate_indices = (
            None if intermediate_indices is None else list(intermediate_indices)
        )

        self.model = self._load_radio_model()
        self.model._register_load_state_dict_pre_hook(self._inject_current_radio_state_dict)
        self.output_proj = None
        if out_channels is not None:
            if self.in_channels is None:
                self.output_proj = torch.nn.LazyConv2d(int(out_channels), kernel_size=1, bias=True)
            else:
                self.output_proj = torch.nn.Conv2d(
                    self.in_channels,
                    int(out_channels),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )

        self._apply_freeze_policy()

    def _load_radio_model(self):
        """Load local HF RADIO code, repairing incomplete dynamic-module cache.

        Transformers copies ``trust_remote_code`` modules into
        ``HF_HOME/modules/transformers_modules`` before importing them.  Some
        RADIO snapshots use ``from . import dual_hybrid_vit`` style imports that
        may be missed by older/newer dynamic-module scanners, leaving a stale
        cache without that sibling file.  When loading from a local RADIO
        directory, copy the missing sibling from the snapshot and retry.
        """

        src_dir = Path(str(self.model_id_or_path)).expanduser()
        max_repairs = 16 if src_dir.is_dir() else 1
        repaired = set()
        last_error = None
        for _ in range(max_repairs):
            try:
                return AutoModel.from_pretrained(
                    self.model_id_or_path,
                    **self.from_pretrained_kwargs)
            except FileNotFoundError as exc:
                last_error = exc
                missing_path = Path(str(exc.filename or ""))
                src_path = src_dir / missing_path.name
                if (
                    not src_dir.is_dir()
                    or not src_path.is_file()
                    or missing_path in repaired
                ):
                    raise
                missing_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, missing_path)
                repaired.add(missing_path)
        raise last_error

    def _get_radio_blocks(self):
        radio_model = getattr(self.model, 'radio_model', None)
        trunk = getattr(radio_model, 'model', None)
        blocks = getattr(trunk, 'blocks', None)
        if isinstance(blocks, (torch.nn.Sequential, torch.nn.ModuleList, list, tuple)):
            return list(blocks)
        return []

    def _partial_unfreeze_enabled(self):
        return bool(self.freeze and self.unfreeze_last_n_blocks > 0)

    def _apply_freeze_policy(self):
        if not self.freeze:
            return

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if not self._partial_unfreeze_enabled():
            self.model.eval()
            return

        blocks = self._get_radio_blocks()
        if not blocks:
            raise ValueError(
                'unfreeze_last_n_blocks requires radio_model.model.blocks to exist')
        if self.unfreeze_last_n_blocks > len(blocks):
            raise ValueError(
                f'unfreeze_last_n_blocks={self.unfreeze_last_n_blocks} exceeds '
                f'available RADIO blocks={len(blocks)}')

        for block in blocks[-self.unfreeze_last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze and not self._partial_unfreeze_enabled():
            self.model.eval()
        elif self._partial_unfreeze_enabled():
            self.model.eval()
            for block in self._get_radio_blocks()[-self.unfreeze_last_n_blocks:]:
                block.train(mode)
        return self

    def _ensure_model_device(self, target_device):
        for parameter in self.model.parameters():
            if parameter.device != target_device:
                self.model.to(target_device)
            return
        for buffer in self.model.buffers():
            if buffer.device != target_device:
                self.model.to(target_device)
            return

    def _inject_current_radio_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        # AdaOcc load_from checkpoints do not contain Hugging Face RADIO trunk weights.
        # Populate missing RADIO keys with the currently loaded frozen backbone state
        # so custom remote-code _load_from_state_dict hooks do not fail on empty input.
        current_state = self.model.state_dict()
        for name, value in current_state.items():
            full_key = f"{prefix}{name}"
            if full_key not in state_dict:
                state_dict[full_key] = value.detach().clone()

    def _project_single_feature(self, features):
        if not isinstance(features, torch.Tensor) or features.dim() != 4:
            raise ValueError(
                "RADIO backbone feature must be Tensor[N, C, H, W], "
                f'got type={type(features)} shape={getattr(features, "shape", None)}'
            )
        if self.output_proj is None:
            return features

        # Make the layout explicit before the trainable projection.
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

        self._ensure_model_device(image_tensor.device)
        radio_model = self.model.radio_model
        if self.freeze and not self._partial_unfreeze_enabled():
            with torch.no_grad():
                if self.intermediate_indices is None:
                    _, features = radio_model(image_tensor, feature_fmt="NCHW")
                else:
                    features = radio_model.forward_intermediates(
                        image_tensor,
                        indices=self.intermediate_indices,
                        output_fmt="NCHW",
                        intermediates_only=True,
                    )
        else:
            if self.intermediate_indices is None:
                _, features = radio_model(image_tensor, feature_fmt="NCHW")
            else:
                features = radio_model.forward_intermediates(
                    image_tensor,
                    indices=self.intermediate_indices,
                    output_fmt="NCHW",
                    intermediates_only=True,
                )

        if self.intermediate_indices is not None:
            if not isinstance(features, (list, tuple)) or len(features) == 0:
                raise ValueError(
                    "RADIO intermediate path must return a non-empty list of Tensor[N, C, H, W], "
                    f'got type={type(features)}'
                )
            return [self._project_single_feature(level_feat) for level_feat in features]

        return self._project_single_feature(features)


@MODELS.register_module()
class RADIOv4HFBackbone(RADIOHFBackbone):
    """Backward-compatible alias for existing RADIOv4 configs."""

    pass
