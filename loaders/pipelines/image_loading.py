import copy
import os
import os.path as osp

import mmcv
import numpy as np
import torch
import mmdet3d.datasets.transforms.loading  # noqa: F401
from mmengine.dist import get_dist_info
from mmengine.fileio import get
from mmdet3d.registry import TRANSFORMS

from ..utils import compose_ego2img
from ._shared import (
    _resolve_cam_types,
    _resolve_num_views,
    _timestamp_us_to_seconds,
)


@TRANSFORMS.register_module(force=True)
class LoadMultiViewImageFromFiles:
    """Load multi-view images and support configurable decoding backend."""

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 imdecode_backend=None,
                 backend_args=None,
                 num_views=5,
                 num_ref_frames=-1,
                 test_mode=False,
                 set_default_scale=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.num_views = num_views
        self.num_ref_frames = num_ref_frames
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def _select_ref_frames(self, results):
        if self.num_ref_frames <= 0:
            return results
        init_choice = np.array([0], dtype=np.int64)
        num_views = int(results.get('num_views', self.num_views) or self.num_views)
        num_frames = len(results['img_filename']) // num_views - 1
        if num_frames == 0:
            choices = np.random.choice(1, self.num_ref_frames, replace=True)
        elif num_frames >= self.num_ref_frames:
            if self.test_mode:
                choices = np.arange(num_frames - self.num_ref_frames, num_frames) + 1
            else:
                choices = np.random.choice(num_frames, self.num_ref_frames, replace=False) + 1
        elif num_frames > 0 and num_frames < self.num_ref_frames:
            if self.test_mode:
                base_choices = np.arange(num_frames) + 1
                random_choices = np.random.choice(
                    num_frames, self.num_ref_frames - num_frames, replace=True) + 1
                choices = np.concatenate([base_choices, random_choices])
            else:
                choices = np.random.choice(num_frames, self.num_ref_frames, replace=True) + 1
        else:
            raise NotImplementedError
        choices = np.concatenate([init_choice, choices])
        select_filename = []
        for choice in choices:
            select_filename += results['img_filename'][choice * num_views:
                                                       (choice + 1) * num_views]
        results['img_filename'] = select_filename
        for key in ['cam2img', 'lidar2cam']:
            if key in results:
                select_results = []
                for choice in choices:
                    select_results += results[key][choice * num_views:(choice + 1) * num_views]
                results[key] = select_results
        for key in ['ego2global']:
            if key in results:
                select_results = []
                for choice in choices:
                    select_results += [results[key][choice]]
                results[key] = select_results
        for key in ['lidar2cam']:
            if key in results:
                for choice_idx in range(1, len(choices)):
                    pad_prev_ego2global = np.eye(4)
                    prev_ego2global = results['ego2global'][choice_idx]
                    pad_prev_ego2global[:prev_ego2global.shape[0], :prev_ego2global.shape[1]] = prev_ego2global
                    pad_cur_ego2global = np.eye(4)
                    cur_ego2global = results['ego2global'][0]
                    pad_cur_ego2global[:cur_ego2global.shape[0], :cur_ego2global.shape[1]] = cur_ego2global
                    cur2prev = np.linalg.inv(pad_prev_ego2global).dot(pad_cur_ego2global)
                    for result_idx in range(choice_idx * num_views,
                                            (choice_idx + 1) * num_views):
                        results[key][result_idx] = results[key][result_idx].dot(cur2prev)
        return results

    def __call__(self, results):
        if 'img_filename' in results:
            results = self._select_ref_frames(results)
            filename = results['img_filename']
        elif 'images' in results:
            filename, cam2img, lidar2cam = [], [], []
            for _, cam_item in results['images'].items():
                filename.append(cam_item['img_path'])
                if 'cam2img' in cam_item:
                    cam2img.append(cam_item['cam2img'])
                if 'lidar2cam' in cam_item:
                    lidar2cam.append(cam_item['lidar2cam'])
            results['filename'] = filename
            if cam2img:
                results['cam2img'] = cam2img
                results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
            if lidar2cam:
                results['lidar2cam'] = lidar2cam
            results['img_filename'] = filename
        else:
            raise KeyError('Results must contain "img_filename" or "images".')

        img_bytes = [get(name, backend_args=self.backend_args) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type, backend=self.imdecode_backend)
            for img_byte in img_bytes
        ]
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
            imgs = [mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        num_views = results.get('num_views', None)
        if num_views is None or int(num_views) <= 0:
            num_views = len(results['img']) if self.num_ref_frames <= 0 else self.num_views
        results['num_views'] = int(num_views)
        results['num_ref_frames'] = self.num_ref_frames
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'imdecode_backend={self.imdecode_backend}, '
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module(name='AdaOccLoadMultiViewImageFromFiles', force=True)
class AdaOccLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """AdaOcc registry name for multi-view image loading."""

    pass


@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6,
                 force_offline=False,
                 cam_types=None,
                 imdecode_backend='turbojpeg'):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline
        self.cam_types = list(cam_types) if cam_types is not None else None

        self.train_interval = train_interval
        self.test_interval = test_interval

        mmcv.use_backend(imdecode_backend)

    def _get_cam_types(self, results):
        return _resolve_cam_types(results, self.cam_types)

    def _pick_sweep(self, sweeps, idx, cam_types):
        sweep_idx = min(idx, len(sweeps) - 1)
        sweep = sweeps[sweep_idx]
        if len(sweep.keys()) < len(cam_types) and sweep_idx > 0:
            sweep = sweeps[sweep_idx - 1]
        sensors = [sensor for sensor in cam_types if sensor in sweep]
        if not sensors:
            sensors = list(sweep.keys())
        return sweep, sensors

    def load_offline(self, results):
        cam_types = self._get_cam_types(results)
        num_views = _resolve_num_views(results, cam_types)

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(num_views):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + [
                    len(results['cam_sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep, sensors = self._pick_sweep(results['cam_sweeps']['prev'], idx, cam_types)
                for sensor in sensors:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(
                        _timestamp_us_to_seconds(sweep[sensor].get('timestamp')))
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        np.asarray(sweep[sensor]['sensor2global_rotation'], dtype=np.float32),
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        assert self.test_mode

        cam_types = self._get_cam_types(results)
        num_views = _resolve_num_views(results, cam_types)

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(num_views):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['ego2img'].append(np.copy(results['ego2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep, sensors = self._pick_sweep(results['cam_sweeps']['prev'], idx, cam_types)
                for sensor in sensors:
                    results['img_timestamp'].append(
                        _timestamp_us_to_seconds(sweep[sensor].get('timestamp')))
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['ego2img'].append(compose_ego2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        np.asarray(sweep[sensor]['sensor2global_rotation'], dtype=np.float32),
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        return self.load_offline(results)


@TRANSFORMS.register_module()
class SelectTemporalFrames:
    """Keep specific zero-based frame blocks from a TN multiview sequence."""

    def __init__(self,
                 frame_indices,
                 keys=('img', 'filename', 'img_timestamp', 'ego2img',
                       'cam2img', 'lidar2cam', 'lidar2img')):
        if not isinstance(frame_indices, (list, tuple)) or len(frame_indices) == 0:
            raise ValueError('frame_indices must be a non-empty list/tuple')
        self.frame_indices = [int(idx) for idx in frame_indices]
        self.keys = tuple(keys)

    @staticmethod
    def _get_sequence_length(value):
        if isinstance(value, (list, tuple)):
            return len(value)
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
        return None

    @staticmethod
    def _slice_value(value, keep_idx, total_views):
        if isinstance(value, list) and len(value) == total_views:
            return [copy.deepcopy(value[idx]) for idx in keep_idx]
        if isinstance(value, tuple) and len(value) == total_views:
            return tuple(copy.deepcopy(value[idx]) for idx in keep_idx)
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_views:
            return np.ascontiguousarray(value[keep_idx])
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == total_views:
            keep_idx_tensor = torch.as_tensor(keep_idx, device=value.device, dtype=torch.long)
            return value.index_select(0, keep_idx_tensor)
        return value

    def __call__(self, results):
        cam_types = _resolve_cam_types(results)
        num_views = _resolve_num_views(results, cam_types)

        total_views = None
        for key in self.keys:
            if key not in results:
                continue
            total_views = self._get_sequence_length(results[key])
            if total_views is not None:
                break
        if total_views is None:
            raise KeyError(
                f'SelectTemporalFrames could not infer total_views from keys={self.keys}')
        if total_views % num_views != 0:
            raise ValueError(
                f'SelectTemporalFrames expects total_views divisible by num_views={num_views}, '
                f'got total_views={total_views}')

        num_frames = total_views // num_views
        normalized_frame_indices = []
        for frame_idx in self.frame_indices:
            if frame_idx < 0 or frame_idx >= num_frames:
                raise IndexError(
                    f'frame index {frame_idx} out of range for num_frames={num_frames}')
            normalized_frame_indices.append(frame_idx)

        keep_idx = []
        for frame_idx in normalized_frame_indices:
            base = frame_idx * num_views
            keep_idx.extend(range(base, base + num_views))

        for key in self.keys:
            if key in results:
                results[key] = self._slice_value(results[key], keep_idx, total_views)

        results['selected_frame_indices'] = list(normalized_frame_indices)
        results['selected_num_frames'] = len(normalized_frame_indices)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(frame_indices={self.frame_indices}, '
        repr_str += f'keys={self.keys})'
        return repr_str
