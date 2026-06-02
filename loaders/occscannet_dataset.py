import os
import os.path as osp
import warnings

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import mkdir_or_exist
from mmdet3d.registry import DATASETS
from tqdm import tqdm

from .geometry import as_rotation_matrix, transform_matrix
from .utils import compose_ego2img, install_numpy_pickle_compat_aliases


@DATASETS.register_module()
class OccScanNetDataset(BaseDataset):
    def __init__(self,
                 ann_file,
                 data_root,
                 pipeline,
                 modality,
                 classes=None,
                 occ_root=None,
                 dataset_cfg=None,
                 test_mode=False,
                 **kwargs):
        self.modality = modality
        self.occ_root = occ_root
        self.dataset_cfg = dataset_cfg or {}
        kwargs.setdefault('serialize_data', False)
        metainfo = dict(classes=classes) if classes is not None else None
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            metainfo=metainfo,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        install_numpy_pickle_compat_aliases()
        data = load(self.ann_file)
        if isinstance(data, dict) and 'infos' in data:
            self.data_infos = data['infos']
        elif isinstance(data, dict) and 'data_list' in data:
            self.data_infos = data['data_list']
        elif isinstance(data, list):
            self.data_infos = data
        else:
            raise TypeError(f'Unsupported annotation format: {type(data)}')
        self.data_infos = [self._resolve_info_paths(info) for info in self.data_infos]
        return self.data_infos

    def _resolve_path(self, path):
        if not isinstance(path, str) or not path:
            return path
        path = osp.expanduser(path)
        if osp.isabs(path):
            return osp.normpath(path)
        if self.data_root:
            return osp.normpath(osp.join(self.data_root, path))
        return osp.normpath(path)

    def _resolve_cam_group_paths(self, cam_group):
        if not isinstance(cam_group, dict):
            return cam_group
        resolved = {}
        for cam_name, cam_info in cam_group.items():
            if not isinstance(cam_info, dict):
                resolved[cam_name] = cam_info
                continue
            cam_info = dict(cam_info)
            cam_info['data_path'] = self._resolve_path(cam_info.get('data_path'))
            cam_info['depth_path'] = self._resolve_path(cam_info.get('depth_path'))
            resolved[cam_name] = cam_info
        return resolved

    def _resolve_lidar_info_paths(self, lidar_info):
        if not isinstance(lidar_info, dict):
            return lidar_info
        lidar_info = dict(lidar_info)
        lidar_info['data_path'] = self._resolve_path(lidar_info.get('data_path'))
        lidar_info['lidar_path'] = self._resolve_path(lidar_info.get('lidar_path'))
        return lidar_info

    def _resolve_info_paths(self, info):
        if not isinstance(info, dict):
            return info
        info = dict(info)
        info['lidar_path'] = self._resolve_path(info.get('lidar_path'))
        info['lidar_points'] = self._resolve_lidar_info_paths(info.get('lidar_points'))
        info['cams'] = self._resolve_cam_group_paths(info.get('cams'))

        cam_sweeps = info.get('cam_sweeps')
        if isinstance(cam_sweeps, list):
            info['cam_sweeps'] = [
                self._resolve_cam_group_paths(sweep) for sweep in cam_sweeps
            ]
        elif isinstance(cam_sweeps, dict):
            info['cam_sweeps'] = {
                side: [self._resolve_cam_group_paths(sweep) for sweep in sweeps]
                if isinstance(sweeps, list) else sweeps
                for side, sweeps in cam_sweeps.items()
            }

        lidar_sweeps = info.get('lidar_sweeps')
        if isinstance(lidar_sweeps, list):
            info['lidar_sweeps'] = [
                self._resolve_lidar_info_paths(sweep) for sweep in lidar_sweeps
            ]
        elif isinstance(lidar_sweeps, dict):
            info['lidar_sweeps'] = {
                side: [self._resolve_lidar_info_paths(sweep) for sweep in sweeps]
                if isinstance(sweeps, list) else sweeps
                for side, sweeps in lidar_sweeps.items()
            }

        return info

    @staticmethod
    def _cam_group_timestamp(cam_group):
        if not isinstance(cam_group, dict):
            return float('-inf')
        timestamps = []
        for cam_info in cam_group.values():
            if isinstance(cam_info, dict) and cam_info.get('timestamp', None) is not None:
                timestamps.append(float(cam_info['timestamp']))
        return max(timestamps) if timestamps else float('-inf')

    @staticmethod
    def _scene_id(info):
        if not isinstance(info, dict):
            return None
        return info.get('scene_token', info.get('scene_name', None))

    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        base_scene = self._scene_id(self.data_infos[index])
        while len(all_sweeps_prev) < into_past and curr_index > 0:
            if self._scene_id(self.data_infos[curr_index]) != base_scene:
                break
            curr_sweeps = self.data_infos[curr_index].get('cam_sweeps', [])
            if len(curr_sweeps) != 0:
                curr_sweeps = sorted(
                    curr_sweeps,
                    key=self._cam_group_timestamp,
                    reverse=True)
                remaining = into_past - len(all_sweeps_prev)
                all_sweeps_prev.extend(curr_sweeps[:remaining])
            if len(all_sweeps_prev) >= into_past:
                break
            prev_info = self.data_infos[curr_index - 1]
            if self._scene_id(prev_info) != base_scene:
                break
            prev_cams = prev_info.get('cams', {})
            if isinstance(prev_cams, dict) and prev_cams:
                all_sweeps_prev.append(prev_cams)
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            if self._scene_id(self.data_infos[curr_index]) != base_scene:
                break
            curr_sweeps = self.data_infos[curr_index].get('cam_sweeps', [])
            if len(curr_sweeps) != 0:
                curr_sweeps = sorted(
                    curr_sweeps,
                    key=self._cam_group_timestamp)
                remaining = into_future - len(all_sweeps_next)
                all_sweeps_next.extend(curr_sweeps[:remaining])
            if len(all_sweeps_next) >= into_future:
                break
            next_cams = self.data_infos[curr_index].get('cams', {})
            if isinstance(next_cams, dict) and next_cams:
                all_sweeps_next.append(next_cams)
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def _target_cam_types(self, info):
        configured = self.dataset_cfg.get('cam_types', None)
        if configured:
            target = [cam_name for cam_name in configured]
            missing = [cam_name for cam_name in target if cam_name not in info.get('cams', {})]
            if missing:
                raise KeyError(
                    f'Missing configured cameras {missing} in sample token={info.get("token")}')
            return target
        return list(info['cams'].keys())

    @staticmethod
    def _filter_cam_group(cam_group, cam_types):
        if not isinstance(cam_group, dict):
            return cam_group
        filtered = {}
        for cam_name in cam_types:
            if cam_name in cam_group:
                filtered[cam_name] = cam_group[cam_name]
        return filtered

    def _filter_cam_sweeps(self, sweeps, cam_types):
        filtered = []
        for sweep in sweeps:
            if not isinstance(sweep, dict):
                continue
            filtered_sweep = self._filter_cam_group(sweep, cam_types)
            if filtered_sweep:
                filtered.append(filtered_sweep)
        return filtered

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        base_scene = self._scene_id(self.data_infos[index])
        while len(all_sweeps_prev) < into_past:
            if curr_index < 0:
                break
            if self._scene_id(self.data_infos[curr_index]) != base_scene:
                break
            curr_sweeps = self.data_infos[curr_index].get('lidar_sweeps', [])
            if len(curr_sweeps) == 0:
                break
            remaining = into_past - len(all_sweeps_prev)
            all_sweeps_prev.extend(curr_sweeps[:remaining])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            if self._scene_id(self.data_infos[curr_index]) != base_scene:
                break
            curr_sweeps = self.data_infos[curr_index].get('lidar_sweeps', [])[::-1]
            if curr_sweeps and curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            if not curr_sweeps:
                curr_index = curr_index + 1
                continue
            remaining = into_future - len(all_sweeps_next)
            all_sweeps_next.extend(curr_sweeps[:remaining])
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    @staticmethod
    def _timestamp_to_seconds(timestamp):
        if timestamp is None:
            return 0.0
        return float(timestamp) / 1e6

    @staticmethod
    def _normalize_cam_intrinsic(cam_intrinsic):
        intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)
        if intrinsic.shape == (3, 3):
            return intrinsic
        if intrinsic.shape == (4, 4):
            return intrinsic[:3, :3].copy()
        raise ValueError(f'Unsupported cam_intrinsic shape: {intrinsic.shape}')

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']

        ego2global_rotation_mat = as_rotation_matrix(ego2global_rotation)
        lidar2ego_rotation_mat = as_rotation_matrix(lidar2ego_rotation)
        ego2lidar = transform_matrix(
            lidar2ego_translation, lidar2ego_rotation, inverse=True)

        input_dict = dict(
            sample_token=info['token'],
            scene_name=info['scene_name'],
            timestamp=self._timestamp_to_seconds(info['timestamp']),
            ego2lidar=ego2lidar,
            ego2obj=ego2lidar,
            ego2occ=np.eye(4),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                pts_filename=info['lidar_path'],
                lidar_points=info.get('lidar_points', {'lidar_path': info['lidar_path']}),
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            ego2img = []
            cam_types = self._target_cam_types(info)
            filtered_cams = self._filter_cam_group(info['cams'], cam_types)

            for cam_name in cam_types:
                cam_info = dict(filtered_cams[cam_name])
                cam_info['timestamp'] = self._timestamp_to_seconds(cam_info.get('timestamp', None))
                cam_info['cam_intrinsic'] = self._normalize_cam_intrinsic(cam_info['cam_intrinsic'])
                filtered_cams[cam_name] = cam_info
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'])
                ego2img.append(
                    compose_ego2img(
                        ego2global_translation,
                        ego2global_rotation_mat,
                        cam_info['sensor2global_translation'],
                        as_rotation_matrix(cam_info['sensor2global_rotation']),
                        cam_info['cam_intrinsic']
                    )
                )

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)
            cam_sweeps_prev = self._filter_cam_sweeps(cam_sweeps_prev, cam_types)
            cam_sweeps_next = self._filter_cam_sweeps(cam_sweeps_next, cam_types)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                cams=filtered_cams,
                ego2img=ego2img,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
                cam_types=cam_types,
                num_views=len(cam_types),
            ))

        if not self.test_mode:
            annos = dict(
                gt_bboxes_3d=np.array([[[]]]),
                gt_labels_3d=np.array([[[]]]),
                gt_names=np.array([[[]]]))
            input_dict['ann_info'] = annos

        input_dict['sample_idx'] = index
        return input_dict

    def _build_metric(self, eval_kwargs):
        from .metrics.occ3d_metric import Occ3DMetric

        occ_io_cfg = self.dataset_cfg.get('occ_io', {})
        metric_cfg = self.dataset_cfg.get('metric', {})

        return Occ3DMetric(
            ann_file=self.ann_file,
            occ_root=eval_kwargs.get('occ_root', self.occ_root or osp.join(self.data_root, 'gts')),
            empty_label=eval_kwargs.get(
                'empty_label', self.dataset_cfg.get('empty_label', metric_cfg.get('empty_label', 79))),
            use_camera_mask=eval_kwargs.get(
                'use_camera_mask', metric_cfg.get('use_camera_mask', True)),
            pc_range=eval_kwargs.get('pc_range', self.dataset_cfg.get('pc_range', None)),
            voxel_size=eval_kwargs.get('voxel_size', self.dataset_cfg.get('voxel_size', None)),
            class_names=eval_kwargs.get('class_names', self.dataset_cfg.get('class_names', None)),
            miou_num_workers=eval_kwargs.get(
                'miou_num_workers', metric_cfg.get('miou_num_workers', 0)),
            occ_path_template=eval_kwargs.get(
                'occ_path_template', occ_io_cfg.get('path_template', '{scene_name}/{token}/labels.npz')),
            semantics_key=eval_kwargs.get(
                'semantics_key', occ_io_cfg.get('semantics_key', 'semantics')),
            mask_camera_key=eval_kwargs.get(
                'mask_camera_key', occ_io_cfg.get('mask_camera_key', 'mask_camera')),
            mask_lidar_key=eval_kwargs.get(
                'mask_lidar_key', occ_io_cfg.get('mask_lidar_key', 'mask_lidar')),
            mask_camera_bits_key=eval_kwargs.get(
                'mask_camera_bits_key', occ_io_cfg.get('mask_camera_bits_key', None)),
            camera_names_key=eval_kwargs.get(
                'camera_names_key', occ_io_cfg.get('camera_names_key', None)),
            mask_camera_select_names=eval_kwargs.get(
                'mask_camera_select_names', occ_io_cfg.get('mask_camera_select_names', None)),
        )

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        warnings.warn(
            'OccScanNetDataset.evaluate is deprecated; please use mmengine '
            'evaluators (Occ3DMetric) via val_evaluator/test_evaluator.',
            DeprecationWarning,
            stacklevel=2,
        )
        metric = self._build_metric(eval_kwargs)
        packed_results = [
            dict(pred=pred, sample_idx=i)
            for i, pred in enumerate(occ_results)
        ]
        return metric.compute_metrics(packed_results)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, f'{sample_token}.npz')
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')
