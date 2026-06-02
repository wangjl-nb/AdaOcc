import copy
import numpy as np
from numpy.linalg import inv
from mmdet3d.registry import TRANSFORMS

from ._shared import _build_cam_lookup, _path_keys


@TRANSFORMS.register_module()
class PackOnlineDepthInputs:
    """Save pre-augmentation images and camera metadata for online depth.

    This transform is intentionally lightweight: it runs before
    ``RandomTransformImage`` so the online DepthAnything branch can infer depth
    at the same resolution/camera calibration as the precomputed depth PNGs.
    The normal AdaOcc image pipeline may still resize/crop the ``img`` field
    afterwards for RADIO.
    """

    def __init__(self,
                 cam_types=None,
                 image_key='img',
                 output_key='online_depth',
                 strict=True,
                 enabled=True):
        self.cam_types = list(cam_types) if cam_types is not None else None
        self.image_key = image_key
        self.output_key = output_key
        self.strict = bool(strict)
        self.enabled = bool(enabled)

    @staticmethod
    def _copy_array(value, dtype=None):
        arr = np.asarray(value, dtype=dtype)
        return np.array(arr, copy=True)

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def __call__(self, results):
        if not self.enabled:
            return results
        images = results.get(self.image_key, None)
        filenames = results.get('filename', [])
        timestamps = results.get('img_timestamp', [])
        if images is None:
            if self.strict:
                raise KeyError(f'Missing image key: {self.image_key}')
            return results
        if not isinstance(images, list):
            images = list(images)
        if not isinstance(filenames, list):
            filenames = list(filenames)
        if not isinstance(timestamps, list):
            timestamps = list(timestamps)

        cam_lookup, _ = _build_cam_lookup(results, self.cam_types)
        lidar2occ = None
        if 'ego2lidar' in results and 'ego2occ' in results:
            lidar2occ = (
                np.asarray(results['ego2occ'], dtype=np.float32)
                @ inv(np.asarray(results['ego2lidar'], dtype=np.float32))
            ).astype(np.float32)
        views = []
        for idx, image in enumerate(images):
            image_path = filenames[idx] if idx < len(filenames) else None
            match = None
            for key in _path_keys(image_path):
                if key in cam_lookup:
                    match = cam_lookup[key]
                    break
            if match is None:
                if self.strict:
                    raise FileNotFoundError(
                        f'No camera metadata matched for online depth image: {image_path}')
                continue

            cam_name, cam_info = match
            view = dict(
                image=np.array(image, copy=True),
                filename=image_path,
                cam_name=cam_name,
                view_idx=int(idx),
                timestamp=self._as_float(
                    timestamps[idx] if idx < len(timestamps) else results.get('timestamp', 0.0)),
                cam_intrinsic=self._copy_array(cam_info['cam_intrinsic'], dtype=np.float32),
                sensor2lidar_rotation=self._copy_array(
                    cam_info['sensor2lidar_rotation'], dtype=np.float32),
                sensor2lidar_translation=self._copy_array(
                    cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3),
            )
            # Preserve optional paths/pose fields for diagnostics and future multi-frame work.
            for key in ('depth_path', 'data_path', 'sensor2global_rotation',
                        'sensor2global_translation'):
                if key in cam_info:
                    value = cam_info[key]
                    if isinstance(value, np.ndarray):
                        value = np.array(value, copy=True)
                    else:
                        value = copy.deepcopy(value)
                    view[key] = value
            views.append(view)

        if self.strict and not views:
            raise RuntimeError('PackOnlineDepthInputs did not pack any views')

        results[self.output_key] = dict(
            views=views,
            timestamp=self._as_float(results.get('timestamp', 0.0)),
            sample_token=copy.deepcopy(results.get('sample_token', None)),
            scene_name=copy.deepcopy(results.get('scene_name', None)),
            lidar2occ=lidar2occ,
        )
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(cam_types={self.cam_types}, '
                f'image_key={self.image_key!r}, output_key={self.output_key!r}, '
                f'strict={self.strict}, enabled={self.enabled})')
