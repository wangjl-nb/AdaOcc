import os.path as osp

import numpy as np
from mmdet3d.registry import TRANSFORMS

from ._shared import DEFAULT_OCC3D_CLASS_NAMES


@TRANSFORMS.register_module()
class LoadOcc3DFromFile:

    def __init__(self,
                 occ_root,
                 ignore_class_names=None,
                 path_template='{scene_name}/{token}/labels.npz',
                 semantics_key='semantics',
                 raw_semantics_key='raw_semantics',
                 voxel_origin_key='voxel_origin',
                 voxel_size_key='voxel_size',
                 mask_camera_key='mask_camera',
                 mask_lidar_key='mask_lidar',
                 mask_camera_bits_key=None,
                 mask_cuboid_box_key='mask_cuboid_box',
                 mask_cuboid_occ_key='mask_cuboid_occ',
                 camera_names_key=None,
                 mask_camera_select_names=None,
                 class_names=None,
                 empty_label=None):
        self.occ_root = occ_root
        self.ignore_class_names = ignore_class_names or []
        self.path_template = path_template
        self.semantics_key = semantics_key
        self.raw_semantics_key = raw_semantics_key
        self.voxel_origin_key = voxel_origin_key
        self.voxel_size_key = voxel_size_key
        self.mask_camera_key = mask_camera_key
        self.mask_lidar_key = mask_lidar_key
        self.mask_camera_bits_key = mask_camera_bits_key
        self.mask_cuboid_box_key = mask_cuboid_box_key
        self.mask_cuboid_occ_key = mask_cuboid_occ_key
        self.camera_names_key = camera_names_key
        self.mask_camera_select_names = tuple(mask_camera_select_names or ())
        self.occ_class_names = class_names or list(DEFAULT_OCC3D_CLASS_NAMES)
        self.empty_label = len(self.occ_class_names) - 1 if empty_label is None else int(empty_label)

    def _build_occ_path(self, results):
        fmt = dict(results)
        if 'sample_token' in results:
            fmt.setdefault('token', results['sample_token'])
        if 'token' in results:
            fmt.setdefault('sample_token', results['token'])
        return osp.join(self.occ_root, self.path_template.format(**fmt))

    @staticmethod
    def _mask_from_bits(mask_camera_bits, camera_names, selected_names):
        selected = set(selected_names)
        mask = np.zeros_like(mask_camera_bits, dtype=np.bool_)
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_name in selected:
                mask |= (mask_camera_bits & (1 << cam_idx)) != 0
        return mask

    def __call__(self, results):
        occ_file = self._build_occ_path(results)
        occ_labels = np.load(occ_file)

        semantics = np.array(occ_labels[self.semantics_key], copy=True)
        mask_shape = semantics.shape
        mask_lidar = occ_labels[self.mask_lidar_key].astype(np.bool_) \
            if self.mask_lidar_key in occ_labels else np.ones(mask_shape, dtype=np.bool_)
        mask_camera = occ_labels[self.mask_camera_key].astype(np.bool_) \
            if self.mask_camera_key in occ_labels else np.ones(mask_shape, dtype=np.bool_)

        mask_camera_bits = None
        mask_cuboid_box = None
        mask_cuboid_occ = None
        camera_names = None
        raw_semantics = None
        voxel_origin = None
        voxel_size = None
        if self.mask_camera_bits_key and self.mask_camera_bits_key in occ_labels:
            mask_camera_bits = np.asarray(occ_labels[self.mask_camera_bits_key], dtype=np.uint8)
        if self.mask_cuboid_box_key and self.mask_cuboid_box_key in occ_labels:
            mask_cuboid_box = np.asarray(occ_labels[self.mask_cuboid_box_key], dtype=np.uint8)
        if self.mask_cuboid_occ_key and self.mask_cuboid_occ_key in occ_labels:
            mask_cuboid_occ = np.asarray(occ_labels[self.mask_cuboid_occ_key], dtype=np.uint8)
        if self.camera_names_key and self.camera_names_key in occ_labels:
            camera_names = [str(x) for x in occ_labels[self.camera_names_key].tolist()]
        if self.raw_semantics_key and self.raw_semantics_key in occ_labels:
            raw_semantics = np.asarray(occ_labels[self.raw_semantics_key], dtype=np.uint8)
        if self.voxel_origin_key and self.voxel_origin_key in occ_labels:
            voxel_origin = np.asarray(occ_labels[self.voxel_origin_key], dtype=np.float32).reshape(3)
        if self.voxel_size_key and self.voxel_size_key in occ_labels:
            voxel_size = np.asarray(occ_labels[self.voxel_size_key], dtype=np.float32).reshape(3)

        if mask_camera_bits is not None and camera_names is not None and self.mask_camera_select_names:
            mask_camera = self._mask_from_bits(
                mask_camera_bits,
                camera_names,
                self.mask_camera_select_names)

        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        if mask_camera_bits is not None:
            results['mask_camera_bits'] = np.ascontiguousarray(mask_camera_bits)
        if mask_cuboid_box is not None:
            results['mask_cuboid_box'] = np.ascontiguousarray(mask_cuboid_box)
        if mask_cuboid_occ is not None:
            results['mask_cuboid_occ'] = np.ascontiguousarray(mask_cuboid_occ)
        if camera_names is not None:
            results['mask_camera_names'] = list(camera_names)
        if raw_semantics is not None:
            results['raw_semantics'] = np.ascontiguousarray(raw_semantics)
        if voxel_origin is not None:
            results['voxel_origin'] = np.ascontiguousarray(voxel_origin)
        if voxel_size is not None:
            results['voxel_size'] = np.ascontiguousarray(voxel_size)

        if self.ignore_class_names:
            for class_id, class_name in enumerate(self.occ_class_names):
                if class_id == self.empty_label:
                    continue
                if class_name not in self.ignore_class_names:
                    continue
                semantics[semantics == class_id] = self.empty_label

        results['voxel_semantics'] = np.ascontiguousarray(semantics)
        return results
