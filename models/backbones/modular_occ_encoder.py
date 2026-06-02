import copy
import warnings

import numpy as np
import torch
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from ..radio.pyramid_adapter import ImagePyramidAdapter
from ..radio.view_preprocessor import RadioViewPreprocessor


@MODELS.register_module()
class ModularOccEncoder(BaseModule):
    """Composable external-image encoder for AdaOcc."""

    def __init__(
        self,
        preprocess_cfg,
        image_backbone_cfg,
        pyramid_adapter_cfg=None,
        num_views=6,
        num_frames=None,
        chunk_by_frame=True,
        strict_shapes=True,
        expect_contiguous=True,
        tn_align_mode="strict",
        lidar_injection="shared",
        preserve_meta_keys=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for ModularOccEncoder: {sorted(kwargs.keys())}"
            )

        self.num_views = int(num_views)
        self.num_frames = None if num_frames is None else int(num_frames)
        self.chunk_by_frame = bool(chunk_by_frame)
        self.strict_shapes = bool(strict_shapes)
        self.expect_contiguous = bool(expect_contiguous)
        self.tn_align_mode = tn_align_mode

        valid_align_modes = ("strict", "pad_last", "truncate_tail")
        if self.tn_align_mode not in valid_align_modes:
            raise ValueError(
                f"tn_align_mode must be one of {valid_align_modes}, got {self.tn_align_mode}"
            )

        self.view_preprocessor = RadioViewPreprocessor(
            preprocess_cfg=preprocess_cfg,
            lidar_injection=lidar_injection,
            tn_align_mode=tn_align_mode,
            preserve_meta_keys=preserve_meta_keys,
        )
        self.image_backbone = MODELS.build(image_backbone_cfg)
        self.pyramid_adapter = None
        if pyramid_adapter_cfg is not None and pyramid_adapter_cfg.get("enabled", True):
            adapter_cfg = dict(pyramid_adapter_cfg)
            adapter_cfg.pop("enabled", None)
            self.pyramid_adapter = ImagePyramidAdapter(**adapter_cfg)

    def build_runtime_img_metas(self, img, img_metas, runtime_num_views=None):
        del runtime_num_views
        return self.view_preprocessor.build_runtime_img_metas(img=img, img_metas=img_metas)

    def _align_tn_img(self, img, target_tn):
        current_tn = img.shape[1]
        if current_tn == target_tn:
            return img

        if self.tn_align_mode == "strict":
            raise ValueError(f"Input TN mismatch: expected {target_tn}, got {current_tn}")

        if self.tn_align_mode == "truncate_tail":
            if current_tn < target_tn:
                raise ValueError(
                    f"truncate_tail mode requires TN >= {target_tn}, got {current_tn}"
                )
            warnings.warn(f"Truncating image TN from {current_tn} to {target_tn}.", stacklevel=2)
            return img[:, :target_tn]

        if current_tn > target_tn:
            warnings.warn(f"Truncating image TN from {current_tn} to {target_tn}.", stacklevel=2)
            return img[:, :target_tn]

        pad_count = target_tn - current_tn
        pad_feat = img[:, -1:].expand(img.shape[0], pad_count, *img.shape[2:])
        warnings.warn(
            f"Padding image TN from {current_tn} to {target_tn} using pad_last.",
            stacklevel=2,
        )
        return torch.cat([img, pad_feat], dim=1)

    def _align_meta_value(self, value, target_tn):
        if isinstance(value, list):
            current_tn = len(value)
            if current_tn == target_tn or self.tn_align_mode == "strict":
                return value
            if self.tn_align_mode == "truncate_tail":
                return value[:target_tn] if current_tn >= target_tn else value
            if current_tn >= target_tn:
                return value[:target_tn]
            if current_tn == 0:
                return [None for _ in range(target_tn)]
            return value + [copy.deepcopy(value[-1]) for _ in range(target_tn - current_tn)]

        if isinstance(value, tuple):
            return tuple(self._align_meta_value(list(value), target_tn))

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            current_tn = int(value.shape[0])
            if current_tn == target_tn or self.tn_align_mode == "strict":
                return value
            if self.tn_align_mode == "truncate_tail":
                return value[:target_tn] if current_tn >= target_tn else value
            if current_tn >= target_tn:
                return value[:target_tn]
            if current_tn == 0:
                return value
            last = value[-1:]
            pad = last.repeat(target_tn - current_tn, *([1] * (last.dim() - 1)))
            return torch.cat([value, pad], dim=0)

        if isinstance(value, np.ndarray) and value.ndim > 0:
            current_tn = int(value.shape[0])
            if current_tn == target_tn or self.tn_align_mode == "strict":
                return value
            if self.tn_align_mode == "truncate_tail":
                return value[:target_tn] if current_tn >= target_tn else value
            if current_tn >= target_tn:
                return value[:target_tn]
            if current_tn == 0:
                return value
            last = value[-1:]
            pad = np.repeat(last, repeats=target_tn - current_tn, axis=0)
            return np.concatenate([value, pad], axis=0)
        return value

    def _align_img_metas(self, img_metas, target_tn):
        if img_metas is None:
            return None
        aligned = []
        for meta in img_metas:
            if not isinstance(meta, dict):
                aligned.append(meta)
                continue
            aligned.append({key: self._align_meta_value(value, target_tn) for key, value in meta.items()})
        return aligned

    def _align_view_extra(self, view_extra, target_tn):
        if view_extra is None:
            return None
        if isinstance(view_extra, dict):
            view_extra = [copy.deepcopy(view_extra)]
        aligned = []
        for sample_extra in view_extra:
            if sample_extra is None:
                aligned.append(None)
                continue
            sample_out = copy.deepcopy(sample_extra)
            views = sample_out.get("views", None)
            if isinstance(views, tuple):
                views = list(views)
            if isinstance(views, list):
                current_tn = len(views)
                if self.tn_align_mode == "strict" and current_tn != target_tn:
                    raise ValueError(
                        f"view_extra views TN mismatch: expected {target_tn}, got {current_tn}"
                    )
                elif self.tn_align_mode == "truncate_tail":
                    if current_tn < target_tn:
                        raise ValueError(
                            f"truncate_tail mode requires extra views >= {target_tn}, got {current_tn}"
                        )
                    views = views[:target_tn]
                else:
                    if current_tn >= target_tn:
                        views = views[:target_tn]
                    elif current_tn == 0:
                        views = [dict() for _ in range(target_tn)]
                    else:
                        views = views + [copy.deepcopy(views[-1]) for _ in range(target_tn - current_tn)]
                sample_out["views"] = views
            aligned.append(sample_out)
        return aligned

    def _slice_img_metas(self, img_metas, start, end, total_views):
        if img_metas is None:
            return None
        sliced = []
        for meta in img_metas:
            if not isinstance(meta, dict):
                sliced.append(meta)
                continue
            chunk_meta = {}
            for key, value in meta.items():
                if isinstance(value, list) and len(value) == total_views:
                    chunk_meta[key] = value[start:end]
                elif isinstance(value, tuple) and len(value) == total_views:
                    chunk_meta[key] = value[start:end]
                elif hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == total_views:
                    chunk_meta[key] = value[start:end]
                else:
                    chunk_meta[key] = value
            sliced.append(chunk_meta)
        return sliced

    def _slice_view_extra(self, view_extra, start, end):
        if view_extra is None:
            return None
        sliced = []
        for item in view_extra:
            if item is None:
                sliced.append(None)
                continue
            out = copy.deepcopy(item)
            if "views" in out and isinstance(out["views"], (list, tuple)):
                out["views"] = list(out["views"][start:end])
            sliced.append(out)
        return sliced

    @staticmethod
    def _stack_processed_view_images(processed_views_batch):
        stacked = []
        for sample_views in processed_views_batch:
            for view in sample_views:
                stacked.append(view["img"])
        return torch.cat(stacked, dim=0)

    @staticmethod
    def _reshape_backbone_feature_levels(features, batch_size, total_views):
        if isinstance(features, torch.Tensor):
            if features.dim() != 4:
                raise ValueError(
                    "image_backbone must return Tensor[N, C, H, W] or list of such tensors, "
                    f'got shape={getattr(features, "shape", None)}'
                )
            if int(features.shape[0]) != batch_size * total_views:
                raise ValueError(
                    f"Backbone batch mismatch: expected {batch_size * total_views}, got {features.shape[0]}"
                )
            return features.reshape(batch_size, total_views, *features.shape[1:])

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError(
                "image_backbone must return Tensor[N, C, H, W] or a non-empty list of such tensors"
            )

        reshaped = []
        for level_idx, level_feat in enumerate(features):
            if not isinstance(level_feat, torch.Tensor) or level_feat.dim() != 4:
                raise ValueError(
                    f"image_backbone level {level_idx} must be Tensor[N, C, H, W], "
                    f'got type={type(level_feat)} shape={getattr(level_feat, "shape", None)}'
                )
            if int(level_feat.shape[0]) != batch_size * total_views:
                raise ValueError(
                    f"Backbone batch mismatch at level {level_idx}: "
                    f"expected {batch_size * total_views}, got {level_feat.shape[0]}"
                )
            reshaped.append(level_feat.reshape(batch_size, total_views, *level_feat.shape[1:]))
        return reshaped

    def _encode_chunk(self, img, points, img_metas, view_extra):
        processed_views_batch = self.view_preprocessor(
            img=img,
            points=points,
            img_metas=img_metas,
            view_extra=view_extra,
        )
        image_tensor = self._stack_processed_view_images(processed_views_batch)
        features = self.image_backbone(image_tensor)
        batch_size = len(processed_views_batch)
        total_views = len(processed_views_batch[0])
        features = self._reshape_backbone_feature_levels(
            features,
            batch_size=batch_size,
            total_views=total_views,
        )
        if self.pyramid_adapter is not None:
            return self.pyramid_adapter(features, processed_views_batch)
        return features

    def forward(self, img, points=None, img_metas=None, view_extra=None, runtime_num_views=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
                f"img must be Tensor[B, TN, C, H, W], got type={type(img)} "
                f'shape={getattr(img, "shape", None)}'
            )

        effective_num_views = self.num_views if runtime_num_views is None else int(runtime_num_views)
        expected_tn = None if self.num_frames is None else effective_num_views * self.num_frames
        if expected_tn is not None:
            valid_tn = {expected_tn}
            if self.chunk_by_frame:
                valid_tn.add(effective_num_views)
            if img.shape[1] not in valid_tn:
                img = self._align_tn_img(img, expected_tn)
                img_metas = self._align_img_metas(img_metas, expected_tn)
                view_extra = self._align_view_extra(view_extra, expected_tn)

        batch_size, total_views = img.shape[:2]
        if isinstance(view_extra, dict):
            view_extra = [copy.deepcopy(view_extra) for _ in range(batch_size)]
        elif isinstance(view_extra, list) and len(view_extra) == 1 and batch_size > 1:
            view_extra = [copy.deepcopy(view_extra[0]) for _ in range(batch_size)]
        if view_extra is not None:
            if not isinstance(view_extra, list) or len(view_extra) != batch_size:
                raise ValueError(
                    f"view_extra must be None/dict or list with length B={batch_size}, "
                    f'got type={type(view_extra)} '
                    f'len={len(view_extra) if isinstance(view_extra, list) else "N/A"}'
                )
        if img_metas is None:
            img_metas = [{} for _ in range(batch_size)]
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f"img_metas must be list of length B={batch_size}, got {type(img_metas)}"
            )

        if self.chunk_by_frame:
            if total_views % effective_num_views != 0:
                raise ValueError(
                    f"total views {total_views} is not divisible by num_views {effective_num_views}"
                )
            num_frames = total_views // effective_num_views
            multi_level_feats = None
            feats = []
            for frame_idx in range(num_frames):
                start = frame_idx * effective_num_views
                end = start + effective_num_views
                chunk_img = img[:, start:end]
                chunk_metas = self._slice_img_metas(img_metas, start, end, total_views)
                chunk_extra = self._slice_view_extra(view_extra, start, end)
                chunk_feat = self._encode_chunk(
                    chunk_img,
                    points=points,
                    img_metas=chunk_metas,
                    view_extra=chunk_extra,
                )
                if isinstance(chunk_feat, list):
                    if multi_level_feats is None:
                        multi_level_feats = [[] for _ in range(len(chunk_feat))]
                    for level_idx, level_feat in enumerate(chunk_feat):
                        multi_level_feats[level_idx].append(level_feat)
                else:
                    feats.append(chunk_feat)

            output = (
                [torch.cat(level_chunks, dim=1) for level_chunks in multi_level_feats]
                if multi_level_feats is not None
                else torch.cat(feats, dim=1)
            )
        else:
            output = self._encode_chunk(
                img,
                points=points,
                img_metas=img_metas,
                view_extra=view_extra,
            )

        if isinstance(output, list):
            normalized = []
            for level_feat in output:
                normalized.append(level_feat.contiguous() if self.expect_contiguous else level_feat)
            return normalized
        return output.contiguous() if self.expect_contiguous else output
