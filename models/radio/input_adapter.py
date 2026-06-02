import copy
import warnings

import numpy as np
import torch


class AdaOccViewInputAdapter:
    """Adapt AdaOcc batch tensors to RADIO-style list[view-dict] inputs."""

    _DEFAULT_META_KEYS = (
        'filename',
        'img_shape',
        'ori_shape',
        'pad_shape',
        'img_timestamp',
        'ego2img',
        'ego2occ',
        'ego2lidar',
        'sample_token',
        'scene_name',
        'sample_idx',
    )

    def __init__(self,
                 lidar_injection='shared',
                 tn_align_mode='strict',
                 preserve_meta_keys=None):
        valid_lidar_modes = ('none', 'shared', 'per_view', 'both')
        if lidar_injection not in valid_lidar_modes:
            raise ValueError(
                f'lidar_injection must be one of {valid_lidar_modes}, got {lidar_injection}')
        valid_align_modes = ('strict', 'pad_last', 'truncate_tail')
        if tn_align_mode not in valid_align_modes:
            raise ValueError(
                f'tn_align_mode must be one of {valid_align_modes}, got {tn_align_mode}')

        self.lidar_injection = lidar_injection
        self.tn_align_mode = tn_align_mode
        self.preserve_meta_keys = tuple(preserve_meta_keys or self._DEFAULT_META_KEYS)

    def _normalize_points(self, points, batch_size):
        if points is None:
            return None

        if isinstance(points, torch.Tensor):
            if points.dim() == 3 and points.shape[0] == batch_size:
                return [points[i] for i in range(batch_size)]
            if points.dim() == 2 and batch_size == 1:
                return [points]
            raise TypeError(
                'points tensor input must have shape [B, N, C] when passed as Tensor, '
                f'but got {tuple(points.shape)}')

        if not isinstance(points, (list, tuple)):
            raise TypeError(
                'points must be None, Tensor[B,N,C], or list/tuple of per-sample tensors, '
                f'but got {type(points)}')

        points_list = list(points)
        if len(points_list) != batch_size:
            raise ValueError(
                f'points batch size mismatch: got {len(points_list)} samples but expected {batch_size}')

        normalized = []
        for pts in points_list:
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            normalized.append(pts)
        return normalized

    def _normalize_view_extra(self, view_extra, batch_size):
        if view_extra is None:
            return [dict() for _ in range(batch_size)]

        if isinstance(view_extra, dict):
            return [copy.deepcopy(view_extra) for _ in range(batch_size)]

        if not isinstance(view_extra, (list, tuple)):
            raise TypeError(
                'view_extra must be None, dict, or list/tuple of dict, '
                f'but got {type(view_extra)}')

        view_extra = list(view_extra)
        if len(view_extra) != batch_size:
            raise ValueError(
                f'view_extra batch size mismatch: got {len(view_extra)} samples, '
                f'expected {batch_size}')

        normalized = []
        for item in view_extra:
            if item is None:
                normalized.append(dict())
            elif isinstance(item, dict):
                normalized.append(copy.deepcopy(item))
            else:
                raise TypeError(
                    'Each view_extra item must be dict/None, '
                    f'but got {type(item)}')
        return normalized

    def _align_extra_views(self, views_extra, target_tn):
        if views_extra is None:
            return [dict() for _ in range(target_tn)]
        if not isinstance(views_extra, (list, tuple)):
            raise TypeError(
                f'view_extra["views"] must be list/tuple, got {type(views_extra)}')

        views_extra = list(views_extra)
        cur_tn = len(views_extra)
        if cur_tn == target_tn:
            return views_extra

        if self.tn_align_mode == 'strict':
            raise ValueError(
                f'view_extra views length mismatch: expected TN={target_tn}, got {cur_tn}')

        if self.tn_align_mode == 'truncate_tail':
            if cur_tn < target_tn:
                raise ValueError(
                    f'truncate_tail mode requires extra views >= TN={target_tn}, got {cur_tn}')
            warnings.warn(
                f'Truncating view_extra views from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return views_extra[:target_tn]

        # pad_last mode
        if cur_tn > target_tn:
            warnings.warn(
                f'Truncating view_extra views from {cur_tn} to {target_tn}.',
                stacklevel=2)
            return views_extra[:target_tn]

        if cur_tn == 0:
            pad_item = dict()
        else:
            pad_item = views_extra[-1]
        out = list(views_extra)
        while len(out) < target_tn:
            out.append(copy.deepcopy(pad_item))
        warnings.warn(
            f'Padding view_extra views from {cur_tn} to {target_tn} using pad_last.',
            stacklevel=2)
        return out

    def _slice_meta_value(self, value, idx, total_views):
        if isinstance(value, list) and len(value) == total_views:
            return value[idx]
        if isinstance(value, tuple) and len(value) == total_views:
            return value[idx]
        if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == total_views:
            return value[idx]
        return value

    @staticmethod
    def _is_batched_sequence(value, batch_size):
        return isinstance(value, (list, tuple)) and len(value) == batch_size and batch_size > 0

    @staticmethod
    def _value_shape(value):
        if isinstance(value, torch.Tensor):
            return tuple(int(v) for v in value.shape)
        if isinstance(value, np.ndarray):
            return tuple(int(v) for v in value.shape)
        try:
            arr = np.asarray(value)
            return tuple(int(v) for v in arr.shape)
        except Exception:
            return None

    @staticmethod
    def _is_batched_geom_sequence(value, batch_size, expected_shapes):
        if not AdaOccViewInputAdapter._is_batched_sequence(value, batch_size):
            return False
        if len(value) == 0:
            return False
        first_shape = AdaOccViewInputAdapter._value_shape(value[0])
        return first_shape in expected_shapes

    @staticmethod
    def _select_batched_value(key, value, batch_idx, batch_size):
        # Geometry entries can appear either per-view (single matrix/vector)
        # or pseudo-collated as per-batch sequences/tensors; handle both.
        if key in {'intrinsics', 'camera_poses', 'camera_pose_quats', 'camera_pose_trans'}:
            expected_shapes = {
                'intrinsics': {(3, 3)},
                'camera_poses': {(4, 4)},
                'camera_pose_quats': {(4,)},
                'camera_pose_trans': {(3,)},
            }[key]
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch_size:
                tail_shape = tuple(int(v) for v in value.shape[1:])
                if tail_shape in expected_shapes:
                    return value[batch_idx]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == batch_size:
                tail_shape = tuple(int(v) for v in value.shape[1:])
                if tail_shape in expected_shapes:
                    return np.array(value[batch_idx], copy=True)
            if AdaOccViewInputAdapter._is_batched_geom_sequence(
                    value, batch_size=batch_size, expected_shapes=expected_shapes):
                return copy.deepcopy(value[batch_idx])
            return copy.deepcopy(value)

        if key == 'is_metric_scale':
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                return value[batch_idx]
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
                return np.array(value[batch_idx], copy=True)

        # For generic tensors/arrays, only treat ndim>=3 as batched payload.
        # This avoids accidental slicing of per-view 2D matrices like 3x3 / 4x4.
        if isinstance(value, torch.Tensor) and value.dim() >= 3 and value.shape[0] == batch_size:
            return value[batch_idx]
        if isinstance(value, np.ndarray) and value.ndim >= 3 and value.shape[0] == batch_size:
            return np.array(value[batch_idx], copy=True)
        if AdaOccViewInputAdapter._is_batched_sequence(value, batch_size):
            first = value[0]
            if isinstance(first, (dict, list, tuple, np.ndarray, torch.Tensor, bool, np.bool_)):
                return copy.deepcopy(value[batch_idx])
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return copy.deepcopy(value[0])
        return value

    def _merge_view_extra(self, base_view, view_extra, batch_idx, batch_size):
        if view_extra is None:
            return base_view
        if not isinstance(view_extra, dict):
            raise TypeError(
                f'Each view-level extra must be dict/None, got {type(view_extra)}')
        if 'img' in view_extra:
            raise KeyError('view_extra.views[*] must not override the "img" field')
        merged = dict(base_view)
        normalized_extra = {}
        for key, value in view_extra.items():
            if key == 'is_metric_scale' and self._is_batched_sequence(value, batch_size):
                normalized_extra[key] = copy.deepcopy(value[batch_idx])
                continue
            normalized_extra[key] = self._select_batched_value(
                key, value, batch_idx=batch_idx, batch_size=batch_size)
        merged.update(normalized_extra)
        return merged

    def __call__(self, img, points, img_metas, view_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
                f'img must be Tensor[B, TN, C, H, W], got type={type(img)} '
                f'shape={getattr(img, "shape", None)}')

        batch_size, total_views, channels, _, _ = img.shape
        if channels != 3:
            raise ValueError(f'Expected 3-channel image tensor, got C={channels}')
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f'img_metas must be list of length B={batch_size}, got {type(img_metas)}')

        points_list = self._normalize_points(points, batch_size)
        extra_list = self._normalize_view_extra(view_extra, batch_size)

        batch_views = []
        for batch_idx in range(batch_size):
            meta = img_metas[batch_idx] if isinstance(img_metas[batch_idx], dict) else {}
            extra = extra_list[batch_idx]

            shared_extra = dict(extra.get('shared', {}))
            if self.lidar_injection in ('shared', 'both') and points_list is not None:
                shared_extra.setdefault('lidar_points', points_list[batch_idx])

            views_extra = self._align_extra_views(extra.get('views', None), total_views)

            sample_views = []
            for view_idx in range(total_views):
                view = dict(img=img[batch_idx, view_idx].permute(1, 2, 0).contiguous())
                for key in self.preserve_meta_keys:
                    if key in meta:
                        view[key] = self._slice_meta_value(meta[key], view_idx, total_views)

                if self.lidar_injection in ('per_view', 'both') and points_list is not None:
                    view['lidar_points'] = points_list[batch_idx]
                if self.lidar_injection in ('shared', 'both'):
                    view['shared'] = shared_extra

                view = self._merge_view_extra(
                    view,
                    views_extra[view_idx],
                    batch_idx=batch_idx,
                    batch_size=batch_size)
                sample_views.append(view)

            batch_views.append(sample_views)
        return batch_views
