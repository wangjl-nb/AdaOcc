import copy
from typing import Sequence

import numpy as np
import torch

from mmengine.structures import BaseDataElement
from mmdet3d.registry import TRANSFORMS

try:
    from mmdet3d.structures import Det3DDataSample
except Exception:  # pragma: no cover
    Det3DDataSample = BaseDataElement


@TRANSFORMS.register_module()
class PackOcc3DInputs:
    def __init__(self,
                 meta_keys: Sequence[str] = (),
                 extra_input_keys: Sequence[str] = ()):
        self.meta_keys = tuple(meta_keys)
        self.extra_input_keys = tuple(extra_input_keys)

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data

    def _pack_imgs(self, imgs):
        if isinstance(imgs, list):
            img_tensors = []
            for img in imgs:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if img.ndim == 3:
                    img = img.permute(2, 0, 1)
                img_tensors.append(img)
            return torch.stack(img_tensors, dim=0)
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if imgs.ndim == 3:
            imgs = imgs.permute(2, 0, 1)
        return imgs

    def __call__(self, results):
        inputs = {}
        if 'img' in results:
            inputs['img'] = self._pack_imgs(results['img'])
        if 'points' in results:
            points = results['points']
            if hasattr(points, 'tensor'):
                points = points.tensor
            inputs['points'] = self._to_tensor(points)
        for key in self.extra_input_keys:
            if key in results:
                inputs[key] = copy.deepcopy(results[key])

        data_sample = Det3DDataSample()
        if 'voxel_semantics' in results:
            voxel_semantics = self._to_tensor(results['voxel_semantics'])
            data_sample.voxel_semantics = voxel_semantics.long()
        if 'raw_semantics' in results:
            raw_semantics = self._to_tensor(results['raw_semantics'])
            data_sample.raw_semantics = raw_semantics.long()
        if 'voxel_origin' in results:
            voxel_origin = self._to_tensor(results['voxel_origin'])
            data_sample.voxel_origin = voxel_origin.to(dtype=torch.float32)
        if 'voxel_size' in results:
            voxel_size = self._to_tensor(results['voxel_size'])
            data_sample.voxel_size = voxel_size.to(dtype=torch.float32)
        if 'mask_camera' in results:
            mask_camera = self._to_tensor(results['mask_camera'])
            data_sample.mask_camera = mask_camera.bool()
        if 'mask_camera_bits' in results:
            mask_camera_bits = self._to_tensor(results['mask_camera_bits'])
            data_sample.mask_camera_bits = mask_camera_bits.to(dtype=torch.uint8)
        if 'mask_cuboid_box' in results:
            mask_cuboid_box = self._to_tensor(results['mask_cuboid_box'])
            data_sample.mask_cuboid_box = mask_cuboid_box.bool()
        if 'mask_cuboid_occ' in results:
            mask_cuboid_occ = self._to_tensor(results['mask_cuboid_occ'])
            data_sample.mask_cuboid_occ = mask_cuboid_occ.bool()
        if 'mask_lidar' in results:
            mask_lidar = self._to_tensor(results['mask_lidar'])
            data_sample.mask_lidar = mask_lidar.bool()

        meta = {}
        for key in self.meta_keys:
            if key in results:
                meta[key] = results[key]
        if 'mask_camera_names' in results:
            meta['mask_camera_names'] = copy.deepcopy(results['mask_camera_names'])
        if meta:
            data_sample.set_metainfo(meta)

        return dict(inputs=inputs, data_samples=data_sample)
