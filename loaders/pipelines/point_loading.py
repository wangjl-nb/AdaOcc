import numpy as np
import torch
from numpy.linalg import inv
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointsRangeFilterWithViewIds:

    def __init__(self,
                 point_cloud_range,
                 point_view_ids_key='depth_point_view_ids'):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.point_view_ids_key = point_view_ids_key

    def __call__(self, results):
        if 'points' not in results:
            return results
        points = results['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        results['points'] = clean_points
        points_mask_np = points_mask.cpu().numpy()

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[points_mask_np]
        if pts_semantic_mask is not None:
            results['pts_semantic_mask'] = pts_semantic_mask[points_mask_np]

        if self.point_view_ids_key and self.point_view_ids_key in results:
            point_view_ids = np.asarray(results[self.point_view_ids_key])
            if point_view_ids.shape[0] != points_mask_np.shape[0]:
                raise ValueError(
                    f'{self.point_view_ids_key} length mismatch before range filter: '
                    f'got {point_view_ids.shape[0]}, expected {points_mask_np.shape[0]}')
            results[self.point_view_ids_key] = np.ascontiguousarray(point_view_ids[points_mask_np])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()}, '
        repr_str += f'point_view_ids_key={self.point_view_ids_key})'
        return repr_str


@TRANSFORMS.register_module()
class PointsVoxelDedupWithViewIds:

    def __init__(self,
                 voxel_size,
                 point_cloud_range=None,
                 point_view_ids_key='depth_point_view_ids',
                 time_dim=4,
                 max_points_total=0,
                 dedup_across_views=True):
        voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
        if voxel_size.size != 3:
            raise ValueError(f'voxel_size must have 3 elements, got shape {voxel_size.shape}')
        self.voxel_size = voxel_size
        self.point_view_ids_key = point_view_ids_key
        self.time_dim = int(time_dim) if time_dim is not None else None
        self.max_points_total = int(max_points_total) if max_points_total is not None else 0
        self.dedup_across_views = bool(dedup_across_views)
        if point_cloud_range is None:
            self.range_origin = None
        else:
            point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32).reshape(-1)
            if point_cloud_range.size != 6:
                raise ValueError(
                    f'point_cloud_range must have 6 elements, got shape {point_cloud_range.shape}')
            self.range_origin = point_cloud_range[:3]

    def _get_priority(self, points_np):
        if self.time_dim is None:
            return np.zeros((points_np.shape[0],), dtype=np.float32)
        if self.time_dim < 0 or self.time_dim >= points_np.shape[1]:
            return np.zeros((points_np.shape[0],), dtype=np.float32)
        return np.abs(points_np[:, self.time_dim]).astype(np.float32)

    def _apply_indices(self, results, keep_idx):
        points = results['points']
        device = points.tensor.device
        keep_idx_t = torch.from_numpy(np.ascontiguousarray(keep_idx)).to(
            device=device, dtype=torch.long)
        results['points'] = points[keep_idx_t]

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[keep_idx]
        if pts_semantic_mask is not None:
            results['pts_semantic_mask'] = pts_semantic_mask[keep_idx]

        if self.point_view_ids_key and self.point_view_ids_key in results:
            point_view_ids = np.asarray(results[self.point_view_ids_key])
            if point_view_ids.shape[0] != points.tensor.shape[0]:
                raise ValueError(
                    f'{self.point_view_ids_key} length mismatch before voxel dedup: '
                    f'got {point_view_ids.shape[0]}, expected {points.tensor.shape[0]}')
            results[self.point_view_ids_key] = np.ascontiguousarray(point_view_ids[keep_idx])
        return results

    def __call__(self, results):
        points = results['points']
        points_np = points.tensor.cpu().numpy()
        if points_np.shape[0] <= 1:
            return results

        xyz = points_np[:, :3].astype(np.float32)
        if self.range_origin is not None:
            xyz = xyz - self.range_origin[None, :]
        voxel_coords = np.floor(xyz / self.voxel_size[None, :]).astype(np.int64)
        dedup_coords = voxel_coords
        if (not self.dedup_across_views) and self.point_view_ids_key and self.point_view_ids_key in results:
            point_view_ids = np.asarray(results[self.point_view_ids_key]).reshape(-1)
            if point_view_ids.shape[0] != points_np.shape[0]:
                raise ValueError(
                    f'{self.point_view_ids_key} length mismatch before voxel dedup: '
                    f'got {point_view_ids.shape[0]}, expected {points_np.shape[0]}')
            dedup_coords = np.concatenate(
                [voxel_coords, point_view_ids[:, None].astype(np.int64)],
                axis=1)

        priority = self._get_priority(points_np)
        original_idx = np.arange(points_np.shape[0], dtype=np.int64)
        lexsort_keys = [original_idx, priority]
        for axis in reversed(range(dedup_coords.shape[1])):
            lexsort_keys.append(dedup_coords[:, axis])
        order = np.lexsort(tuple(lexsort_keys))
        voxel_sorted = dedup_coords[order]
        keep_mask = np.ones((order.shape[0],), dtype=np.bool_)
        if order.shape[0] > 1:
            keep_mask[1:] = np.any(voxel_sorted[1:] != voxel_sorted[:-1], axis=1)
        keep_idx = order[keep_mask]

        if self.max_points_total > 0 and keep_idx.shape[0] > self.max_points_total:
            kept_priority = priority[keep_idx]
            kept_order = np.lexsort((
                np.arange(keep_idx.shape[0], dtype=np.int64),
                kept_priority,
            ))
            keep_idx = keep_idx[kept_order[:self.max_points_total]]

        keep_idx = np.sort(keep_idx)
        return self._apply_indices(results, keep_idx)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(voxel_size={self.voxel_size.tolist()}, '
        repr_str += f'point_cloud_range_origin={None if self.range_origin is None else self.range_origin.tolist()}, '
        repr_str += f'point_view_ids_key={self.point_view_ids_key}, '
        repr_str += f'time_dim={self.time_dim}, '
        repr_str += f'max_points_total={self.max_points_total}, '
        repr_str += f'dedup_across_views={self.dedup_across_views})'
        return repr_str


@TRANSFORMS.register_module()
class LiDARToOccSpace:

    def __call__(self, results):
        points = results['points']
        ego2lidar, ego2occ = results['ego2lidar'], results['ego2occ']

        lidar2occ_np = (ego2occ @ inv(ego2lidar)).astype(np.float32)
        lidar2occ = torch.from_numpy(lidar2occ_np).float()
        ones = torch.ones_like(points.tensor[..., :1])
        pts = torch.cat([points.tensor[..., :3], ones], dim=1).transpose(0, 1)
        pts = torch.matmul(lidar2occ, pts).transpose(0, 1)[..., :3]

        points.tensor = torch.cat([pts, points.tensor[..., 3:]], dim=1)
        results['points'] = points
        results['ego2lidar'] = ego2occ.copy()
        return results
