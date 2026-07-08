import os.path as osp

import numpy as np
import cv2


DEFAULT_CAM_TYPES = [
    'CAM_LEFT', 'CAM_BACK', 'CAM_FRONT',
    'CAM_BOTTOM', 'CAM_TOP', 'CAM_RIGHT'
]

DEFAULT_OCC3D_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


def _infer_cam_types_from_sweeps(sweeps):
    for sweep in sweeps:
        if isinstance(sweep, dict) and sweep:
            return list(sweep.keys())
    return []


def _resolve_cam_types(results, configured_cam_types=None):
    if 'cam_types' in results and results['cam_types']:
        return list(results['cam_types'])

    if configured_cam_types:
        return list(configured_cam_types)

    cam_sweeps = results.get('cam_sweeps', {})
    if isinstance(cam_sweeps, dict):
        inferred = _infer_cam_types_from_sweeps(cam_sweeps.get('prev', []))
        if inferred:
            return inferred
        inferred = _infer_cam_types_from_sweeps(cam_sweeps.get('next', []))
        if inferred:
            return inferred

    sweeps = results.get('sweeps', {})
    if isinstance(sweeps, dict):
        inferred = _infer_cam_types_from_sweeps(sweeps.get('prev', []))
        if inferred:
            return inferred
        inferred = _infer_cam_types_from_sweeps(sweeps.get('next', []))
        if inferred:
            return inferred

    num_views = results.get('num_views', None)
    if num_views is None and isinstance(results.get('img', None), list):
        num_views = len(results['img'])
    if num_views is not None:
        return [f'CAM_{i}' for i in range(int(num_views))]
    return list(DEFAULT_CAM_TYPES)


def _resolve_num_views(results, cam_types):
    if cam_types:
        return int(len(cam_types))
    if 'num_views' in results and results['num_views'] is not None:
        return int(results['num_views'])
    if isinstance(results.get('img', None), list) and results['img']:
        return int(len(results['img']))
    return int(len(DEFAULT_CAM_TYPES))


def _timestamp_us_to_seconds(value):
    if value is None:
        return 0.0
    return float(value) / 1e6


def _transpose_rotation_matrix(value):
    return np.asarray(value, dtype=np.float32).T


def _path_keys(path):
    if not isinstance(path, str) or not path:
        return []
    keys = set()
    norm_path = osp.normpath(path)
    keys.add(norm_path)
    keys.add(osp.normpath(osp.abspath(path)))
    try:
        keys.add(osp.normpath(osp.relpath(path)))
    except Exception:
        pass
    return list(keys)


def _register_cam_info(lookup, cam_name, cam_info):
    if not isinstance(cam_info, dict):
        return
    image_path = cam_info.get('data_path', None)
    for key in _path_keys(image_path):
        if key not in lookup:
            lookup[key] = (cam_name, cam_info)


def _build_cam_lookup(results, configured_cam_types=None):
    lookup = {}
    cam_types = _resolve_cam_types(results, configured_cam_types)

    cams = results.get('cams', {})
    if isinstance(cams, dict):
        for cam_name in cam_types:
            if cam_name in cams:
                _register_cam_info(lookup, cam_name, cams[cam_name])
        for cam_name, cam_info in cams.items():
            _register_cam_info(lookup, cam_name, cam_info)

    cam_sweeps = results.get('cam_sweeps', {})
    if isinstance(cam_sweeps, dict):
        for side in ['prev', 'next']:
            sweeps = cam_sweeps.get(side, [])
            if not isinstance(sweeps, list):
                continue
            for sweep in sweeps:
                if not isinstance(sweep, dict):
                    continue
                for cam_name, cam_info in sweep.items():
                    _register_cam_info(lookup, cam_name, cam_info)
    elif isinstance(cam_sweeps, list):
        for sweep in cam_sweeps:
            if not isinstance(sweep, dict):
                continue
            for cam_name, cam_info in sweep.items():
                _register_cam_info(lookup, cam_name, cam_info)

    return lookup, cam_types


def _depth_candidates_from_image(image_path, include_same_stem=False):
    if not isinstance(image_path, str) or not image_path:
        return []
    path = osp.normpath(image_path)
    parent = osp.dirname(path)
    dirname = osp.basename(parent)
    stem, _ = osp.splitext(osp.basename(path))
    original_stem = stem

    if not stem.endswith('_depth'):
        stem = stem + '_depth'

    if dirname.startswith('image_'):
        depth_dir = osp.join(osp.dirname(parent), dirname.replace('image_', 'depth_', 1))
    else:
        depth_dir = parent
    candidates = [
        osp.join(depth_dir, stem + '.png'),
    ]
    if include_same_stem:
        same_stem_path = osp.join(parent, original_stem + '.png')
        if same_stem_path not in candidates:
            candidates.append(same_stem_path)
    return candidates


def _same_stem_depth_candidate_from_image(image_path):
    if not isinstance(image_path, str) or not image_path:
        return None
    path = osp.normpath(image_path)
    parent = osp.dirname(path)
    stem, _ = osp.splitext(osp.basename(path))
    return osp.join(parent, stem + '.png')


def _resolve_depth_path(cam_info,
                        image_path,
                        depth_key='depth_path',
                        fallback_depth_from_image_path=True,
                        fallback_same_stem_depth_from_image=False,
                        prefer_same_stem_depth_from_image=False):
    candidates = []
    depth_path = cam_info.get(depth_key, None) if isinstance(cam_info, dict) else None
    same_stem_path = _same_stem_depth_candidate_from_image(image_path) \
        if fallback_same_stem_depth_from_image else None
    if prefer_same_stem_depth_from_image and same_stem_path:
        candidates.append(same_stem_path)
    if isinstance(depth_path, str) and depth_path:
        candidates.append(depth_path)
    if fallback_depth_from_image_path:
        image_candidates = _depth_candidates_from_image(
            image_path,
            include_same_stem=fallback_same_stem_depth_from_image)
        if prefer_same_stem_depth_from_image and same_stem_path:
            image_candidates = [cand for cand in image_candidates
                                if osp.normpath(cand) != osp.normpath(same_stem_path)]
        candidates.extend(image_candidates)

    checked = set()
    for cand in candidates:
        if not isinstance(cand, str) or not cand:
            continue
        for key in _path_keys(cand):
            if key in checked:
                continue
            checked.add(key)
            if osp.exists(key):
                return key
    return candidates[0] if candidates else None


def _load_depth_png(depth_path,
                    depth_format='adaocc_rgba_float32',
                    raw_depth_scale=1000.0):
    if depth_path is None:
        return None
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None
    depth_format = str(depth_format or 'adaocc_rgba_float32').lower()
    if depth_format in ('raw_uint16_mm', 'uint16_mm', 'scannet_raw_mm'):
        if depth_img.ndim != 2:
            return None
        scale = float(raw_depth_scale or 1.0)
        if scale <= 0:
            raise ValueError(f'raw_depth_scale must be positive, got {scale}')
        return depth_img.astype(np.float32) / scale

    if depth_format not in ('adaocc_rgba_float32', 'rgba_float32', 'float32_rgba'):
        raise ValueError(
            'depth_format must be one of "adaocc_rgba_float32" or '
            f'"raw_uint16_mm", got {depth_format!r}')

    depth = depth_img.view('<f4').squeeze()
    if depth is None:
        return None
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        return None
    return depth.astype(np.float32)


def _load_rgba_depth(depth_path):
    return _load_depth_png(depth_path, depth_format='adaocc_rgba_float32')
