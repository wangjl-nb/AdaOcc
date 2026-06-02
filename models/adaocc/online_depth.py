import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(value, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _ensure_bhw_depth(depth):
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f'Depth model must return Tensor, got {type(depth)}')
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    elif depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.dim() != 3:
        raise ValueError(f'Depth output must be [B,H,W], got {tuple(depth.shape)}')
    return depth


def project_depth_to_lidar_points(depth,
                                  cam_intrinsics,
                                  sensor2lidar_rotations,
                                  sensor2lidar_translations,
                                  depth_min=0.1,
                                  depth_max=7.5,
                                  sample_stride=1,
                                  intensity_value=0.0,
                                  base_timestamps=None,
                                  image_timestamps=None,
                                  view_indices=None,
                                  max_points_total=0):
    """Project dense depth maps to AdaOcc pseudo-LiDAR point tensors.

    Mirrors ``LoadPointsFromMultiViewDepth`` math: pixel centers use +0.5,
    OpenCV camera convention, row-vector sensor-to-lidar transform
    ``pts_cam @ R.T + t``.
    """
    depth = _ensure_bhw_depth(depth)
    device = depth.device
    depth = depth.to(dtype=torch.float32)
    num_views, height, width = depth.shape
    stride = max(int(sample_stride), 1)

    cam_intrinsics = cam_intrinsics.to(device=device, dtype=torch.float32)
    sensor2lidar_rotations = sensor2lidar_rotations.to(device=device, dtype=torch.float32)
    sensor2lidar_translations = sensor2lidar_translations.to(device=device, dtype=torch.float32)
    if cam_intrinsics.shape != (num_views, 3, 3):
        raise ValueError(
            f'cam_intrinsics shape must be {(num_views, 3, 3)}, got {tuple(cam_intrinsics.shape)}')
    if sensor2lidar_rotations.shape != (num_views, 3, 3):
        raise ValueError(
            'sensor2lidar_rotations shape must be '
            f'{(num_views, 3, 3)}, got {tuple(sensor2lidar_rotations.shape)}')
    if sensor2lidar_translations.shape != (num_views, 3):
        raise ValueError(
            'sensor2lidar_translations shape must be '
            f'{(num_views, 3)}, got {tuple(sensor2lidar_translations.shape)}')

    v_idx = torch.arange(0, height, stride, device=device, dtype=torch.long)
    u_idx = torch.arange(0, width, stride, device=device, dtype=torch.long)
    if v_idx.numel() == 0 or u_idx.numel() == 0:
        return (torch.zeros((0, 5), device=device, dtype=torch.float32),
                torch.zeros((0,), device=device, dtype=torch.long))

    # indexing='ij' yields arrays matching depth[v, u].
    v_grid, u_grid = torch.meshgrid(v_idx, u_idx, indexing='ij')
    d_sub = depth[:, v_idx][:, :, u_idx]
    valid = torch.isfinite(d_sub) & (d_sub > float(depth_min)) & (d_sub < float(depth_max))

    if base_timestamps is None:
        base_timestamps = torch.zeros((num_views,), device=device, dtype=torch.float32)
    else:
        base_timestamps = _as_tensor(base_timestamps, device=device, dtype=torch.float32).reshape(-1)
        if base_timestamps.numel() == 1:
            base_timestamps = base_timestamps.expand(num_views)
    if image_timestamps is None:
        image_timestamps = base_timestamps
    else:
        image_timestamps = _as_tensor(image_timestamps, device=device, dtype=torch.float32).reshape(-1)
        if image_timestamps.numel() == 1:
            image_timestamps = image_timestamps.expand(num_views)
    if view_indices is None:
        view_indices = torch.arange(num_views, device=device, dtype=torch.long)
    else:
        view_indices = torch.as_tensor(view_indices, device=device, dtype=torch.long).reshape(-1)
        if view_indices.numel() != num_views:
            raise ValueError(
                f'view_indices length mismatch: expected {num_views}, got {view_indices.numel()}')

    point_chunks = []
    view_id_chunks = []
    u_float_full = u_grid.to(dtype=torch.float32) + 0.5
    v_float_full = v_grid.to(dtype=torch.float32) + 0.5
    for view_idx in range(num_views):
        mask = valid[view_idx]
        if not torch.any(mask):
            continue
        d = d_sub[view_idx][mask]
        u = u_float_full[mask]
        v = v_float_full[mask]
        intrinsic = cam_intrinsics[view_idx]
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        pts_cam = torch.stack([x, y, d], dim=1)
        pts_lidar = pts_cam @ sensor2lidar_rotations[view_idx].T + sensor2lidar_translations[view_idx].view(1, 3)
        intensity = torch.full((pts_lidar.shape[0], 1), float(intensity_value), device=device, dtype=torch.float32)
        time_delta = torch.clamp(base_timestamps[view_idx] - image_timestamps[view_idx], min=0.0)
        if torch.abs(time_delta).item() < 1e-6:
            time_delta = torch.zeros((), device=device, dtype=torch.float32)
        time_col = torch.full((pts_lidar.shape[0], 1), float(time_delta.item()), device=device, dtype=torch.float32)
        point_chunks.append(torch.cat([pts_lidar.to(torch.float32), intensity, time_col], dim=1))
        view_id_chunks.append(torch.full(
            (pts_lidar.shape[0],), int(view_indices[view_idx].item()), device=device, dtype=torch.long))

    if point_chunks:
        points = torch.cat(point_chunks, dim=0)
        view_ids = torch.cat(view_id_chunks, dim=0)
    else:
        points = torch.zeros((0, 5), device=device, dtype=torch.float32)
        view_ids = torch.zeros((0,), device=device, dtype=torch.long)

    max_points_total = int(max_points_total or 0)
    if max_points_total > 0 and points.shape[0] > max_points_total:
        choice = torch.randperm(points.shape[0], device=device)[:max_points_total]
        points = points.index_select(0, choice)
        view_ids = view_ids.index_select(0, choice)
    return points, view_ids


def transform_points_xyz(points, transform):
    if points.numel() == 0:
        return points
    transform = transform.to(device=points.device, dtype=torch.float32)
    if transform.shape != (4, 4):
        raise ValueError(f'Expected 4x4 point transform, got {tuple(transform.shape)}')
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=torch.float32)
    homo = torch.cat([points[:, :3].to(torch.float32), ones], dim=1)
    xyz = (transform @ homo.T).T[:, :3]
    return torch.cat([xyz, points[:, 3:]], dim=1)


class OnlineDepthAnythingPoints(nn.Module):
    """Frozen DepthAnythingV2 depth inference plus GPU backprojection."""

    def __init__(self,
                 enabled=False,
                 model_path=None,
                 repo_root=None,
                 depth_anything_root=None,
                 encoder='vitb',
                 features=128,
                 out_channels=(96, 192, 384, 768),
                 max_depth=20.0,
                 input_size=518,
                 depth_min=0.1,
                 depth_max=7.5,
                 sample_stride=1,
                 max_points_total=0,
                 point_cloud_range=None,
                 intensity_value=0.0,
                 image_is_bgr=False,
                 depth_model=None):
        super().__init__()
        self.enabled = bool(enabled)
        self.model_path = model_path
        self.repo_root = repo_root
        self.depth_anything_root = depth_anything_root
        self.encoder = str(encoder)
        self.features = int(features)
        self.out_channels = list(out_channels)
        self.max_depth = float(max_depth)
        self.input_size = None if input_size is None else int(input_size)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.sample_stride = max(int(sample_stride), 1)
        self.max_points_total = int(max_points_total or 0)
        self.point_cloud_range = None if point_cloud_range is None \
            else [float(v) for v in point_cloud_range]
        if self.point_cloud_range is not None and len(self.point_cloud_range) != 6:
            raise ValueError(
                f'point_cloud_range must have 6 values, got {self.point_cloud_range}')
        self.intensity_value = float(intensity_value)
        self.image_is_bgr = bool(image_is_bgr)

        self.register_buffer(
            'image_mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.depth_model = depth_model if depth_model is not None else None
        if self.enabled and self.depth_model is None:
            self.depth_model = self._build_depth_model()
        if self.depth_model is not None:
            self._freeze_depth_model()

    def _resolve_repo_root(self):
        if self.repo_root:
            return Path(self.repo_root).expanduser().resolve()
        return Path(__file__).resolve().parents[2]

    def _build_depth_model(self):
        repo_root = self._resolve_repo_root()
        da_root = Path(self.depth_anything_root).expanduser().resolve() \
            if self.depth_anything_root else repo_root / 'Depth_Anything_V2' / 'metric_depth'
        if str(da_root) not in sys.path:
            sys.path.insert(0, str(da_root))
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(
            encoder=self.encoder,
            features=self.features,
            out_channels=self.out_channels,
            max_depth=self.max_depth,
        )
        if self.model_path:
            model_path = Path(self.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = repo_root / model_path
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            checkpoint = torch.load(str(model_path), map_location='cpu')
            state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state, dict):
                raise TypeError(f'Unsupported DepthAnything checkpoint type: {type(checkpoint)}')
            state = {str(k).removeprefix('module.'): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        return model

    def _freeze_depth_model(self):
        for parameter in self.depth_model.parameters():
            parameter.requires_grad_(False)
        self.depth_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.depth_model is not None:
            self.depth_model.eval()
        return self

    @staticmethod
    def _normalize_payload(online_depth):
        if online_depth is None:
            return None
        if isinstance(online_depth, tuple):
            online_depth = list(online_depth)
        if isinstance(online_depth, list):
            return online_depth
        if isinstance(online_depth, dict):
            return [online_depth]
        raise TypeError(f'online_depth must be dict/list/None, got {type(online_depth)}')

    @staticmethod
    def _is_batched_view_field(value, batch_size):
        if isinstance(value, torch.Tensor):
            return value.dim() >= 1 and int(value.shape[0]) == batch_size
        if isinstance(value, np.ndarray):
            return value.ndim >= 1 and int(value.shape[0]) == batch_size
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return True
        return False

    @classmethod
    def _select_field(cls, value, sample_idx, batch_size):
        if cls._is_batched_view_field(value, batch_size):
            if isinstance(value, torch.Tensor):
                return value[sample_idx]
            if isinstance(value, np.ndarray):
                return np.array(value[sample_idx], copy=True)
            return copy.deepcopy(value[sample_idx])
        return copy.deepcopy(value)

    @staticmethod
    def _infer_payload_batch_size(payload):
        if not isinstance(payload, dict):
            return None
        views = payload.get('views', None)
        if not isinstance(views, (list, tuple)) or not views or not isinstance(views[0], dict):
            return None
        for view in views:
            for key, value in view.items():
                if key == 'image':
                    if isinstance(value, torch.Tensor) and value.dim() >= 4:
                        return int(value.shape[0])
                    if isinstance(value, np.ndarray) and value.ndim >= 4:
                        return int(value.shape[0])
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        first = value[0]
                        if isinstance(first, (torch.Tensor, np.ndarray, list, tuple)):
                            return int(len(value))
                if key in {'cam_intrinsic', 'sensor2lidar_rotation'}:
                    if isinstance(value, torch.Tensor) and value.dim() >= 3:
                        return int(value.shape[0])
                    if isinstance(value, np.ndarray) and value.ndim >= 3:
                        return int(value.shape[0])
                if key == 'sensor2lidar_translation':
                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                        return int(value.shape[0])
                    if isinstance(value, np.ndarray) and value.ndim >= 2:
                        return int(value.shape[0])
        return None

    @classmethod
    def _decollate_pseudo_collated_payload(cls, payload, batch_size=None):
        if not isinstance(payload, dict):
            return payload
        views = payload.get('views', None)
        if not isinstance(views, (list, tuple)) or not views or not isinstance(views[0], dict):
            return payload

        inferred = cls._infer_payload_batch_size(payload)
        if batch_size is None:
            batch_size = inferred
        elif inferred is not None and int(inferred) != int(batch_size):
            raise ValueError(
                f'online_depth inferred batch size mismatch: inferred {inferred}, expected {batch_size}')
        if batch_size is None:
            # Common pseudo_collate may already keep online_depth as list[dict].
            return payload
        batch_size = int(batch_size)

        samples = []
        for sample_idx in range(batch_size):
            sample = {}
            for key, value in payload.items():
                if key == 'views':
                    sample_views = []
                    for view in views:
                        sample_views.append({
                            k: cls._select_field(v, sample_idx=sample_idx, batch_size=batch_size)
                            for k, v in view.items()
                        })
                    sample['views'] = sample_views
                else:
                    sample[key] = cls._select_field(value, sample_idx=sample_idx, batch_size=batch_size)
            samples.append(sample)
        return samples

    @classmethod
    def normalize_batch_payload(cls, online_depth, batch_size=None):
        payload = cls._normalize_payload(online_depth)
        if payload is None:
            return None
        if len(payload) == 1 and isinstance(payload[0], dict):
            decollated = cls._decollate_pseudo_collated_payload(payload[0], batch_size=batch_size)
            if isinstance(decollated, list):
                payload = decollated
        if batch_size is not None and len(payload) != int(batch_size):
            if len(payload) == 1 and int(batch_size) > 1:
                payload = [copy.deepcopy(payload[0]) for _ in range(int(batch_size))]
            else:
                raise ValueError(
                    f'online_depth batch size mismatch: got {len(payload)}, expected {batch_size}')
        return payload

    def _view_image_to_tensor(self, image, device):
        if isinstance(image, torch.Tensor):
            tensor = image.to(device=device, non_blocking=True)
        else:
            tensor = torch.as_tensor(image, device=device)
        if tensor.dim() != 3:
            raise ValueError(f'online depth image must be 3D HWC/CHW, got {tuple(tensor.shape)}')
        if int(tensor.shape[0]) == 3 and int(tensor.shape[-1]) != 3:
            chw = tensor.contiguous()
        elif int(tensor.shape[-1]) == 3:
            chw = tensor.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f'Expected 3-channel image, got shape {tuple(tensor.shape)}')
        if chw.dtype == torch.uint8:
            chw = chw.to(dtype=torch.float32).div_(255.0)
        else:
            chw = chw.to(dtype=torch.float32)
            max_value = float(chw.detach().max().item()) if chw.numel() else 0.0
            if max_value > 1.0:
                chw = chw.clamp(0.0, 255.0).div(255.0)
        if self.image_is_bgr:
            chw = chw[[2, 1, 0], :, :]
        return chw

    @staticmethod
    def _resize_lower_bound_multiple(images, input_size, multiple_of=14):
        if input_size is None:
            return images
        _, _, height, width = images.shape
        scale_height = float(input_size) / float(height)
        scale_width = float(input_size) / float(width)
        if scale_width > scale_height:
            scale_height = scale_width
        else:
            scale_width = scale_height

        def _constrain(x, min_val):
            y = int(round(x / multiple_of) * multiple_of)
            if y < min_val:
                y = int(np.ceil(x / multiple_of) * multiple_of)
            return max(y, multiple_of)

        new_h = _constrain(scale_height * height, int(input_size))
        new_w = _constrain(scale_width * width, int(input_size))
        if new_h == height and new_w == width:
            return images
        return F.interpolate(images, size=(new_h, new_w), mode='bicubic', align_corners=False)

    @staticmethod
    def _scalar_float(value, default=0.0):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            return float(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float(default)
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return OnlineDepthAnythingPoints._scalar_float(value[0], default=default)
        if value is None:
            return float(default)
        return float(value)

    def _filter_points_by_range(self, points, view_ids):
        if self.point_cloud_range is None or points.numel() == 0:
            return points, view_ids
        pcr = torch.as_tensor(
            self.point_cloud_range, device=points.device, dtype=torch.float32)
        xyz = points[:, :3]
        mask = (
            (xyz[:, 0] >= pcr[0]) & (xyz[:, 0] < pcr[3]) &
            (xyz[:, 1] >= pcr[1]) & (xyz[:, 1] < pcr[4]) &
            (xyz[:, 2] >= pcr[2]) & (xyz[:, 2] < pcr[5])
        )
        return points[mask], view_ids[mask]

    def _prepare_sample_views(self, sample, device):
        views = sample.get('views', None) if isinstance(sample, dict) else None
        if not isinstance(views, (list, tuple)) or len(views) == 0:
            raise ValueError('online_depth sample must contain non-empty views list')
        image_tensors = []
        intrinsics = []
        rotations = []
        translations = []
        timestamps = []
        view_indices = []
        base_timestamp = sample.get('timestamp', 0.0) if isinstance(sample, dict) else 0.0
        base_timestamp = self._scalar_float(base_timestamp)
        for local_idx, view in enumerate(views):
            if not isinstance(view, dict):
                raise TypeError(f'online depth view must be dict, got {type(view)}')
            image_tensors.append(self._view_image_to_tensor(view['image'], device=device))
            intrinsics.append(_as_tensor(view['cam_intrinsic'], device=device, dtype=torch.float32))
            rotations.append(_as_tensor(view['sensor2lidar_rotation'], device=device, dtype=torch.float32))
            translations.append(_as_tensor(view['sensor2lidar_translation'], device=device, dtype=torch.float32).reshape(3))
            ts = view.get('timestamp', base_timestamp)
            timestamps.append(self._scalar_float(ts, default=base_timestamp))
            view_indices.append(int(view.get('view_idx', local_idx)))

        shapes = {tuple(t.shape[-2:]) for t in image_tensors}
        if len(shapes) != 1:
            raise ValueError(f'online depth views must share H,W in this prototype, got {sorted(shapes)}')
        images = torch.stack(image_tensors, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        rotations = torch.stack(rotations, dim=0)
        translations = torch.stack(translations, dim=0)
        base_timestamps = torch.full((len(views),), base_timestamp, device=device, dtype=torch.float32)
        image_timestamps = torch.as_tensor(timestamps, device=device, dtype=torch.float32)
        view_indices = torch.as_tensor(view_indices, device=device, dtype=torch.long)
        return images, intrinsics, rotations, translations, base_timestamps, image_timestamps, view_indices

    def forward(self, online_depth, device=None, dtype=torch.float32, batch_size=None):
        if not self.enabled:
            return None, None
        if self.depth_model is None:
            self.depth_model = self._build_depth_model()
            self._freeze_depth_model()
        device = torch.device(device) if device is not None else next(self.parameters()).device
        batch_payload = self.normalize_batch_payload(online_depth, batch_size=batch_size)
        if batch_payload is None:
            raise ValueError('online_depth inputs are required when online depth is enabled')

        point_batches = []
        view_id_batches = []
        with torch.inference_mode():
            for sample in batch_payload:
                images, intrinsics, rotations, translations, base_ts, image_ts, view_indices = \
                    self._prepare_sample_views(sample, device=device)
                orig_hw = tuple(int(v) for v in images.shape[-2:])
                model_input = self._resize_lower_bound_multiple(images, self.input_size, multiple_of=14)
                model_input = (model_input - self.image_mean.to(device=device)) / self.image_std.to(device=device)
                depth = self.depth_model(model_input, output_feature=False)
                depth = _ensure_bhw_depth(depth)
                if tuple(depth.shape[-2:]) != orig_hw:
                    depth = F.interpolate(
                        depth[:, None], size=orig_hw, mode='bilinear', align_corners=True)[:, 0]
                points, view_ids = project_depth_to_lidar_points(
                    depth,
                    intrinsics,
                    rotations,
                    translations,
                    depth_min=self.depth_min,
                    depth_max=self.depth_max,
                    sample_stride=self.sample_stride,
                    max_points_total=self.max_points_total,
                    intensity_value=self.intensity_value,
                    base_timestamps=base_ts,
                    image_timestamps=image_ts,
                    view_indices=view_indices,
                )
                lidar2occ = sample.get('lidar2occ', None) if isinstance(sample, dict) else None
                if lidar2occ is not None:
                    points = transform_points_xyz(
                        points, _as_tensor(lidar2occ, device=device, dtype=torch.float32))
                points, view_ids = self._filter_points_by_range(points, view_ids)
                point_batches.append(points.to(dtype=torch.float32))
                view_id_batches.append(view_ids)
        return point_batches, view_id_batches
