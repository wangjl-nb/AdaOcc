import os.path as osp

import numpy as np
from mmdet3d.registry import TRANSFORMS

try:
    from mmdet3d.structures.points import get_points_type
except Exception:  # pragma: no cover
    from mmdet3d.core.points import get_points_type

from ._shared import (
    _build_cam_lookup,
    _depth_candidates_from_image,
    _load_rgba_depth,
    _path_keys,
    _resolve_depth_path,
    _resolve_num_views,
)


@TRANSFORMS.register_module()
class LoadPointsFromMultiViewDepth:
    """Build pseudo-LiDAR points from selected multi-view depth maps."""

    def __init__(self,
                 cam_types=None,
                 depth_key='depth_path',
                 coord_type='LIDAR',
                 load_dim=5,
                 use_dim=[0, 1, 2, 3, 4],
                 time_dim=4,
                 sample_stride=7,
                 sample_stride_current=None,
                 sample_stride_history=None,
                 max_points_total=560000,
                 depth_min=0.1,
                 depth_max=80.0,
                 coord_convention='opencv',
                 intensity_value=0.0,
                 strict_depth_exist=False,
                 fallback_depth_from_image_path=True,
                 history_dynamic_extrinsics=True,
                 dynamic_extrinsics_fallback='static',
                 output_view_ids_key='depth_point_view_ids'):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['LIDAR', 'DEPTH', 'CAMERA']
        assert coord_convention in ['opencv']
        assert dynamic_extrinsics_fallback in ['static', 'skip', 'raise']

        self.cam_types = list(cam_types) if cam_types is not None else None
        self.depth_key = depth_key
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.time_dim = time_dim
        self.sample_stride = max(int(sample_stride), 1)
        self.sample_stride_current = max(int(sample_stride_current), 1) \
            if sample_stride_current is not None else None
        self.sample_stride_history = max(int(sample_stride_history), 1) \
            if sample_stride_history is not None else None
        self.max_points_total = int(max_points_total) if max_points_total is not None else 0
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.coord_convention = coord_convention
        self.intensity_value = float(intensity_value)
        self.strict_depth_exist = bool(strict_depth_exist)
        self.fallback_depth_from_image_path = bool(fallback_depth_from_image_path)
        self.history_dynamic_extrinsics = bool(history_dynamic_extrinsics)
        self.dynamic_extrinsics_fallback = dynamic_extrinsics_fallback
        self.output_view_ids_key = output_view_ids_key

    def _depth_candidates_from_image(self, image_path):
        return _depth_candidates_from_image(image_path)

    def _load_depth(self, depth_path):
        return _load_rgba_depth(depth_path)

    def _resolve_sample_stride(self, is_history=False):
        if is_history and self.sample_stride_history is not None:
            return self.sample_stride_history
        if (not is_history) and self.sample_stride_current is not None:
            return self.sample_stride_current
        return self.sample_stride

    def _depth_to_points_camera(self,
                                depth,
                                cam_intrinsic,
                                sample_stride=None,
                                return_pixel_coords=False):
        if depth is None or depth.ndim != 2:
            empty_points = np.zeros((0, 3), dtype=np.float32)
            empty_pixels = np.zeros((0, 2), dtype=np.int32)
            return (empty_points, empty_pixels) if return_pixel_coords else empty_points

        stride = self.sample_stride if sample_stride is None else max(int(sample_stride), 1)
        h, w = depth.shape
        v_idx = np.arange(0, h, stride, dtype=np.int32)
        u_idx = np.arange(0, w, stride, dtype=np.int32)
        if v_idx.size == 0 or u_idx.size == 0:
            empty_points = np.zeros((0, 3), dtype=np.float32)
            empty_pixels = np.zeros((0, 2), dtype=np.int32)
            return (empty_points, empty_pixels) if return_pixel_coords else empty_points

        depth_sub = depth[np.ix_(v_idx, u_idx)]
        u_pix, v_pix = np.meshgrid(u_idx, v_idx)
        u = u_pix.astype(np.float32) + 0.5
        v = v_pix.astype(np.float32) + 0.5

        valid = np.isfinite(depth_sub)
        valid &= (depth_sub > self.depth_min) & (depth_sub < self.depth_max)
        if not np.any(valid):
            empty_points = np.zeros((0, 3), dtype=np.float32)
            empty_pixels = np.zeros((0, 2), dtype=np.int32)
            return (empty_points, empty_pixels) if return_pixel_coords else empty_points

        d = depth_sub[valid]
        u = u[valid]
        v = v[valid]
        pixel_coords = np.stack([v_pix[valid], u_pix[valid]], axis=1).astype(np.int32)

        intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy

        pts_cam = np.stack([x, y, d], axis=1)
        pts_cam = pts_cam.astype(np.float32)
        if return_pixel_coords:
            return pts_cam, pixel_coords
        return pts_cam

    def _as_rotation_matrix(self, rotation):
        rot = np.asarray(rotation)
        if rot.shape == (3, 3):
            return rot.astype(np.float64)

        quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            raise ValueError(f'Unsupported rotation shape: {rot.shape}')
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError('Quaternion rotation has zero norm')
        w, x, y, z = quat / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _as_translation(self, translation):
        trans = np.asarray(translation, dtype=np.float64).reshape(-1)
        if trans.size != 3:
            raise ValueError(f'Unsupported translation shape: {np.asarray(translation).shape}')
        return trans

    def _compose_row_transform(self, rot_ab, trans_ab, rot_bc, trans_bc):
        rot_ac = rot_bc @ rot_ab
        trans_ac = trans_ab @ rot_bc.T + trans_bc
        return rot_ac, trans_ac

    def _invert_row_transform(self, rot_ab, trans_ab):
        rot_ba = rot_ab.T
        trans_ba = -trans_ab @ rot_ab
        return rot_ba, trans_ba

    def _get_static_sensor2lidar(self, cam_info):
        rot = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32)
        trans = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(-1)
        if rot.shape != (3, 3):
            raise ValueError(f'Invalid sensor2lidar_rotation shape: {rot.shape}')
        if trans.size != 3:
            raise ValueError(f'Invalid sensor2lidar_translation shape: {trans.shape}')
        return rot, trans

    def _get_dynamic_sensor2lidar(self, cam_info, results):
        rot_eg = self._as_rotation_matrix(results['ego2global_rotation'])
        trans_eg = self._as_translation(results['ego2global_translation'])
        rot_le = self._as_rotation_matrix(results['lidar2ego_rotation'])
        trans_le = self._as_translation(results['lidar2ego_translation'])

        rot_lg, trans_lg = self._compose_row_transform(rot_le, trans_le, rot_eg, trans_eg)
        rot_gl, trans_gl = self._invert_row_transform(rot_lg, trans_lg)

        rot_sg = self._as_rotation_matrix(cam_info['sensor2global_rotation'])
        trans_sg = self._as_translation(cam_info['sensor2global_translation'])
        rot_sl, trans_sl = self._compose_row_transform(rot_sg, trans_sg, rot_gl, trans_gl)

        return rot_sl.astype(np.float32), trans_sl.astype(np.float32)

    def _resolve_sensor2lidar(self, cam_info, results, use_dynamic):
        if not use_dynamic:
            return self._get_static_sensor2lidar(cam_info)

        try:
            return self._get_dynamic_sensor2lidar(cam_info, results)
        except Exception:
            if self.dynamic_extrinsics_fallback == 'raise':
                raise
            if self.dynamic_extrinsics_fallback == 'skip':
                return None, None
            return self._get_static_sensor2lidar(cam_info)

    def _camera_to_lidar(self, pts_cam, cam_info, results=None, use_dynamic=False):
        if pts_cam.shape[0] == 0:
            return pts_cam
        rot, trans = self._resolve_sensor2lidar(cam_info, results, use_dynamic)
        if rot is None or trans is None:
            return None
        return pts_cam @ rot.T + trans[None, :]

    def __call__(self, results):
        filenames = results.get('filename', [])
        img_timestamps = results.get('img_timestamp', [])
        if not isinstance(filenames, list):
            filenames = list(filenames)
        if not isinstance(img_timestamps, list):
            img_timestamps = list(img_timestamps)

        cam_lookup, cam_types = _build_cam_lookup(results, self.cam_types)
        num_views = _resolve_num_views(results, cam_types)
        base_timestamp = float(results.get('timestamp', 0.0))

        point_chunks = []
        point_view_id_chunks = []
        for idx, image_path in enumerate(filenames):
            match = None
            for key in _path_keys(image_path):
                if key in cam_lookup:
                    match = cam_lookup[key]
                    break
            if match is None:
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'No camera metadata matched for image: {image_path}')
                continue

            _, cam_info = match
            depth_path = _resolve_depth_path(
                cam_info,
                image_path,
                depth_key=self.depth_key,
                fallback_depth_from_image_path=self.fallback_depth_from_image_path,
            )
            if depth_path is None or (not osp.exists(depth_path)):
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'No depth file for image: {image_path}')
                continue

            depth = _load_rgba_depth(depth_path)
            if depth is None:
                if self.strict_depth_exist:
                    raise FileNotFoundError(f'Failed to load depth file: {depth_path}')
                continue

            is_history = idx >= num_views
            sample_stride = self._resolve_sample_stride(is_history=is_history)
            pts_cam = self._depth_to_points_camera(
                depth,
                cam_info['cam_intrinsic'],
                sample_stride=sample_stride,
                return_pixel_coords=False)
            if pts_cam.shape[0] == 0:
                continue
            use_dynamic = self.history_dynamic_extrinsics and is_history
            pts_lidar = self._camera_to_lidar(
                pts_cam,
                cam_info,
                results=results,
                use_dynamic=use_dynamic)
            if pts_lidar is None:
                continue

            if idx < num_views:
                time_delta = 0.0
            else:
                img_ts = float(img_timestamps[idx]) if idx < len(img_timestamps) else base_timestamp
                time_delta = max(base_timestamp - img_ts, 0.0)
                if abs(time_delta) < 1e-6:
                    time_delta = 0.0

            intensity = np.full((pts_lidar.shape[0], 1), self.intensity_value, dtype=np.float32)
            time_col = np.full((pts_lidar.shape[0], 1), time_delta, dtype=np.float32)
            pts = np.concatenate([pts_lidar.astype(np.float32), intensity, time_col], axis=1)
            point_chunks.append(pts)
            point_view_id_chunks.append(
                np.full((pts_lidar.shape[0],), idx, dtype=np.int32))

        if point_chunks:
            points = np.concatenate(point_chunks, axis=0)
            point_view_ids = np.concatenate(point_view_id_chunks, axis=0)
        else:
            points = np.zeros((0, self.load_dim), dtype=np.float32)
            point_view_ids = np.zeros((0,), dtype=np.int32)

        if self.max_points_total > 0 and points.shape[0] > self.max_points_total:
            choice = np.random.choice(points.shape[0], self.max_points_total, replace=False)
            points = points[choice]
            point_view_ids = point_view_ids[choice]

        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        if self.output_view_ids_key:
            results[self.output_view_ids_key] = np.ascontiguousarray(point_view_ids)
        return results
