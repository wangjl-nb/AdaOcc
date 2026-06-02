import os.path as osp
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.logging import print_log
from mmengine.registry import METRICS

from models.utils import sparse2dense
from loaders.utils import install_numpy_pickle_compat_aliases


def _load_infos(ann_file):
    install_numpy_pickle_compat_aliases()
    data = load(ann_file)
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported annotation format: {type(data)}')


def _ordered_results(results, total_size):
    ordered = [None] * total_size
    seen = set()
    for item in results:
        idx = item.get('sample_idx', None)
        if idx is None:
            raise ValueError('All evaluation results must carry a valid sample_idx.')
        idx = int(idx)
        if idx < 0 or idx >= total_size:
            raise ValueError(
                f'Result sample_idx {idx} is out of range for dataset size {total_size}.')
        if idx in seen:
            raise ValueError(f'Duplicate sample_idx detected in evaluation results: {idx}.')
        seen.add(idx)
        ordered[idx] = item
    if len(results) != total_size:
        raise ValueError(
            f'Evaluation results are incomplete: got {len(results)} items for {total_size} samples.')
    if any(v is None for v in ordered):
        missing = [i for i, value in enumerate(ordered) if value is None]
        raise ValueError(
            f'Evaluation results are missing sample_idx entries: {missing[:8]}')
    return ordered


def _info_with_aliases(info):
    info = dict(info)
    if 'token' in info:
        info.setdefault('sample_token', info['token'])
    if 'sample_token' in info:
        info.setdefault('token', info['sample_token'])
    return info


def _build_occ_path(root, path_template, info):
    info = _info_with_aliases(info)
    return osp.join(root, path_template.format(**info))


def _safe_voxel_size(voxel_size):
    if voxel_size is None:
        return None
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
    if voxel_size.size == 1:
        return float(voxel_size[0])
    if voxel_size.size == 3:
        if np.max(voxel_size) - np.min(voxel_size) < 1e-8:
            return float(voxel_size[0])
        return voxel_size.tolist()
    raise ValueError(f'Unsupported voxel_size shape: {voxel_size.shape}')


def _compute_hist(gt, pred, num_classes, mask=None):
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)

    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_)
        gt = gt[mask]
        pred = pred[mask]

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    return np.bincount(
        num_classes * gt[valid] + pred[valid], minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def _to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, 'detach'):
        data = data.detach()
        if hasattr(data, 'cpu'):
            data = data.cpu()
        return data.numpy()
    return np.asarray(data)


def _sample_hist_from_sparse(occ_labels,
                             mask,
                             occ_loc,
                             sem_pred,
                             num_classes,
                             empty_label):
    occ_labels = np.asarray(occ_labels)
    if mask is None:
        mask = np.ones_like(occ_labels, dtype=np.bool_)
    else:
        mask = np.asarray(mask, dtype=np.bool_)

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    gt_masked = occ_labels[mask].astype(np.int64, copy=False)
    valid_gt_masked = (gt_masked >= 0) & (gt_masked < num_classes)
    if np.any(valid_gt_masked):
        gt_count = np.bincount(gt_masked[valid_gt_masked], minlength=num_classes)
        hist[:, empty_label] = gt_count[:num_classes]

    coords = _to_numpy(occ_loc)
    pred = _to_numpy(sem_pred)
    if coords.size == 0 or pred.size == 0:
        return hist

    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    if coords.shape[0] != pred.shape[0]:
        n = min(coords.shape[0], pred.shape[0])
        coords = coords[:n]
        pred = pred[:n]
    if coords.shape[0] == 0:
        return hist

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    valid_coord = (
        (x >= 0) & (x < occ_labels.shape[0]) &
        (y >= 0) & (y < occ_labels.shape[1]) &
        (z >= 0) & (z < occ_labels.shape[2])
    )
    if not np.any(valid_coord):
        return hist
    coords = coords[valid_coord]
    pred = pred[valid_coord]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Match dense assignment behavior: if duplicated voxel exists, keep last write.
    linear_idx = np.ravel_multi_index((x, y, z), dims=occ_labels.shape)
    order = np.argsort(linear_idx, kind='stable')
    linear_sorted = linear_idx[order]
    keep = np.ones(order.shape[0], dtype=np.bool_)
    keep[:-1] = linear_sorted[:-1] != linear_sorted[1:]
    selected = order[keep]
    coords = coords[selected]
    pred = pred[selected]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    in_mask = mask[x, y, z]
    if not np.any(in_mask):
        return hist
    coords = coords[in_mask]
    pred = pred[in_mask]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    gt_at_pred = occ_labels[x, y, z].astype(np.int64, copy=False)
    valid_gt = (gt_at_pred >= 0) & (gt_at_pred < num_classes)
    if np.any(valid_gt):
        dec = np.bincount(gt_at_pred[valid_gt], minlength=num_classes)
        hist[:, empty_label] -= dec[:num_classes]

    valid_cls = valid_gt & (pred >= 0) & (pred < num_classes)
    if not np.any(valid_cls):
        return hist
    gt_at_pred = gt_at_pred[valid_cls]
    pred = pred[valid_cls]

    pair_hist = np.bincount(
        num_classes * gt_at_pred + pred, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    hist += pair_hist

    return hist


def _sample_hist_from_raw_occscannet_sparse(raw_semantics,
                                            raw_occ_loc,
                                            sem_pred,
                                            num_classes,
                                            empty_label):
    raw_semantics = np.asarray(raw_semantics, dtype=np.int64)
    gt_dense = np.full(raw_semantics.shape, empty_label, dtype=np.int64)
    valid_sem = (raw_semantics >= 1) & (raw_semantics <= 11)
    gt_dense[valid_sem] = raw_semantics[valid_sem] - 1
    mask = raw_semantics != 255

    raw_occ_loc = _to_numpy(raw_occ_loc)
    sem_pred = _to_numpy(sem_pred)
    if raw_occ_loc.size == 0 or sem_pred.size == 0:
        return _sample_hist_from_sparse(
            gt_dense,
            mask,
            np.zeros((0, 3), dtype=np.int64),
            sem_pred,
            num_classes=num_classes,
            empty_label=empty_label)

    raw_occ_loc = np.asarray(raw_occ_loc, dtype=np.int64).reshape(-1, 3)
    sem_pred = np.asarray(sem_pred, dtype=np.int64).reshape(-1)
    if raw_occ_loc.shape[0] != sem_pred.shape[0]:
        n = min(raw_occ_loc.shape[0], sem_pred.shape[0])
        raw_occ_loc = raw_occ_loc[:n]
        sem_pred = sem_pred[:n]

    return _sample_hist_from_sparse(
        gt_dense,
        mask,
        raw_occ_loc,
        sem_pred,
        num_classes=num_classes,
        empty_label=empty_label)


def _compute_iou_vector(hist):
    hist = np.asarray(hist, dtype=np.float64)
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(
        np.diag(hist),
        denom,
        out=np.full(hist.shape[0], np.nan, dtype=np.float64),
        where=denom > 0,
    )


def _valid_class_indices(num_classes, empty_label):
    valid_indices = [idx for idx in range(num_classes)]
    if empty_label is not None and 0 <= empty_label < num_classes:
        valid_indices = [idx for idx in valid_indices if idx != empty_label]
    return valid_indices


def _compute_miou(hist, empty_label):
    iou = _compute_iou_vector(hist)
    valid_indices = _valid_class_indices(iou.shape[0], empty_label)
    if valid_indices:
        mean_iou = np.nanmean(iou[valid_indices])
    else:
        mean_iou = np.nanmean(iou)
    return round(float(mean_iou) * 100.0, 2)


def _compute_occupied_iou(hist, empty_label):
    hist = np.asarray(hist)
    if hist.ndim != 2 or hist.shape[0] != hist.shape[1]:
        return np.nan
    num_classes = hist.shape[0]
    if empty_label is None or empty_label < 0 or empty_label >= num_classes:
        return np.nan

    occ_mask = np.ones(num_classes, dtype=bool)
    occ_mask[empty_label] = False
    tp = hist[np.ix_(occ_mask, occ_mask)].sum()
    fp = hist[empty_label, occ_mask].sum()
    fn = hist[occ_mask, empty_label].sum()
    denom = tp + fp + fn
    if denom <= 0:
        return np.nan
    return float(tp) / float(denom)


@METRICS.register_module()
class Occ3DMetric(BaseMetric):
    default_prefix = 'occ'

    def __init__(self,
                 ann_file,
                 occ_root,
                 empty_label=17,
                 use_camera_mask=True,
                 use_raw_occscannet_gt=False,
                 compute_rayiou=True,
                 pc_range=None,
                 voxel_size=None,
                 class_names=None,
                 miou_num_workers=0,
                 occ_path_template='{scene_name}/{token}/labels.npz',
                 semantics_key='semantics',
                 raw_semantics_key='raw_semantics',
                 voxel_origin_key='voxel_origin',
                 raw_voxel_size_key='voxel_size',
                 mask_camera_key='mask_camera',
                 mask_lidar_key='mask_lidar',
                 mask_camera_bits_key=None,
                 camera_names_key=None,
                 mask_camera_select_names=None,
                 ray_num_workers=8,
                 ray_cfg=None,
                 focus_eval=None,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = int(empty_label)
        self.use_camera_mask = use_camera_mask
        self.use_raw_occscannet_gt = bool(use_raw_occscannet_gt)
        # Legacy args kept only for backward compatibility with old configs.
        self.compute_rayiou = compute_rayiou
        self.pc_range = pc_range
        self.voxel_size = _safe_voxel_size(voxel_size)
        self.class_names = class_names
        self.miou_num_workers = miou_num_workers

        self.occ_path_template = occ_path_template
        self.semantics_key = semantics_key
        self.raw_semantics_key = raw_semantics_key
        self.voxel_origin_key = voxel_origin_key
        self.raw_voxel_size_key = raw_voxel_size_key
        self.mask_camera_key = mask_camera_key
        self.mask_lidar_key = mask_lidar_key
        self.mask_camera_bits_key = mask_camera_bits_key
        self.camera_names_key = camera_names_key
        self.mask_camera_select_names = tuple(mask_camera_select_names or ())

        self.ray_num_workers = ray_num_workers
        self.ray_cfg = ray_cfg or {}
        self.focus_eval = focus_eval or {}
        self.focus_class_freq_ema = None

    @staticmethod
    def _mask_from_bits(mask_camera_bits, camera_names, selected_names):
        selected = set(selected_names)
        mask = np.zeros_like(mask_camera_bits, dtype=np.bool_)
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_name in selected:
                mask |= (mask_camera_bits & (1 << cam_idx)) != 0
        return mask


    def _class_name(self, class_idx):
        if self.class_names is not None and class_idx < len(self.class_names):
            return str(self.class_names[class_idx])
        return f'class_{class_idx}'

    def _update_focus_ema(self, class_freq, focus_cfg):
        class_freq = np.asarray(class_freq, dtype=np.float64)
        momentum = float(focus_cfg.get('ema_momentum', 0.9))
        momentum = min(max(momentum, 0.0), 1.0)
        if self.focus_class_freq_ema is None or self.focus_class_freq_ema.shape != class_freq.shape:
            self.focus_class_freq_ema = class_freq.copy()
        else:
            self.focus_class_freq_ema = momentum * self.focus_class_freq_ema + (1.0 - momentum) * class_freq
        return self.focus_class_freq_ema

    def _select_focus_indices(self, class_count, focus_cfg):
        num_classes = class_count.shape[0]
        valid_indices = _valid_class_indices(num_classes, self.empty_label)
        if not valid_indices:
            return []

        valid_count = class_count[valid_indices].astype(np.float64)
        total = float(valid_count.sum())
        if total > 0:
            class_freq = valid_count / total
        else:
            class_freq = np.ones_like(valid_count) / max(len(valid_count), 1)

        policy = focus_cfg.get('policy', 'ema_freq')
        if policy == 'ema_freq':
            effective_freq = self._update_focus_ema(class_freq, focus_cfg)
        else:
            effective_freq = class_freq

        freq_thr = float(focus_cfg.get('freq_thr', 0.02))
        min_classes = int(focus_cfg.get('min_classes', 1))
        max_classes = int(focus_cfg.get('max_classes', len(valid_indices)))
        max_classes = max(0, min(max_classes, len(valid_indices)))
        if max_classes > 0 and min_classes > max_classes:
            min_classes = max_classes

        selected_local = [idx for idx, freq in enumerate(effective_freq) if freq <= freq_thr]

        if len(selected_local) < min_classes:
            order = np.argsort(effective_freq)
            selected_set = set(selected_local)
            for idx in order:
                if idx not in selected_set:
                    selected_local.append(int(idx))
                    selected_set.add(int(idx))
                if len(selected_local) >= min_classes:
                    break

        if max_classes == 0:
            selected_local = []
        elif len(selected_local) > max_classes:
            selected_local = sorted(selected_local, key=lambda x: effective_freq[x])[:max_classes]

        if len(selected_local) == 0 and focus_cfg.get('fallback', 'rare_classes') == 'rare_classes':
            rare_classes = [int(x) for x in focus_cfg.get('rare_classes', [])]
            rare_local = [
                valid_indices.index(cls_idx)
                for cls_idx in rare_classes
                if cls_idx in valid_indices
            ]
            if rare_local:
                if max_classes > 0:
                    rare_local = rare_local[:max_classes]
                selected_local = rare_local

        selected_global = [valid_indices[idx] for idx in selected_local]
        return selected_global

    def _append_focus_metrics(self, metrics, hist, iou_vector):
        focus_cfg = self.focus_eval or {}
        if not focus_cfg.get('enabled', False):
            return

        class_count = hist.sum(axis=1)
        focus_indices = self._select_focus_indices(class_count, focus_cfg)
        valid_indices = _valid_class_indices(hist.shape[0], self.empty_label)
        head_indices = [idx for idx in valid_indices if idx not in focus_indices]

        if focus_indices:
            small_miou = float(np.nanmean(iou_vector[focus_indices])) * 100.0
        else:
            small_miou = np.nan
        if head_indices:
            head_miou = float(np.nanmean(iou_vector[head_indices])) * 100.0
        else:
            head_miou = np.nan

        metrics['mIoU_small'] = round(float(small_miou), 2)
        metrics['mIoU_head'] = round(float(head_miou), 2)

    def _log_per_class_iou(self, class_iou_items, metrics):
        if not class_iou_items:
            return

        lines = ['Per-class IoU (%):']
        for class_name, class_iou in class_iou_items:
            lines.append(f'  {class_name:<24} {class_iou:7.2f}\n')

        summary_parts = [f"mIoU={metrics.get('mIoU', np.nan):.2f}",
                         f"IoU={metrics.get('IoU', np.nan):.2f}"]
        if 'mIoU_small' in metrics:
            summary_parts.append(f"mIoU_small={metrics['mIoU_small']:.2f}\n")
        if 'mIoU_head' in metrics:
            summary_parts.append(f"mIoU_head={metrics['mIoU_head']:.2f}\n")
        lines.append('Summary: ' + ', '.join(summary_parts))

        print_log('\n'.join(lines), logger='current')

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', []) if isinstance(data_batch, dict) else []
        for idx, pred in enumerate(data_samples):
            sample = batch_samples[idx] if idx < len(batch_samples) else None
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                sample_token=meta.get('sample_token', None),
                scene_name=meta.get('scene_name', None),
            ))

    def _sample_hist(self, info, result_dict, num_classes):
        occ_file = _build_occ_path(self.occ_root, self.occ_path_template, info)
        with np.load(occ_file) as occ_infos:
            if self.use_raw_occscannet_gt:
                raw_semantics = np.asarray(occ_infos[self.raw_semantics_key], dtype=np.uint8)
                if 'raw_occ_loc' not in result_dict:
                    raise ValueError(
                        'use_raw_occscannet_gt=True requires prediction results to carry '
                        '"raw_occ_loc" for evaluation.')
                return _sample_hist_from_raw_occscannet_sparse(
                    raw_semantics,
                    result_dict['raw_occ_loc'],
                    result_dict['sem_pred'],
                    num_classes=num_classes,
                    empty_label=self.empty_label)

            occ_labels = np.asarray(occ_infos[self.semantics_key], dtype=np.uint8)
            if self.use_camera_mask:
                if self.mask_camera_bits_key in occ_infos and \
                        self.camera_names_key in occ_infos and \
                        self.mask_camera_select_names:
                    mask_camera_bits = np.asarray(
                        occ_infos[self.mask_camera_bits_key], dtype=np.uint8)
                    camera_names = [str(x) for x in occ_infos[self.camera_names_key].tolist()]
                    mask = self._mask_from_bits(
                        mask_camera_bits,
                        camera_names,
                        self.mask_camera_select_names)
                elif self.mask_camera_key in occ_infos:
                    mask = np.asarray(occ_infos[self.mask_camera_key], dtype=np.bool_)
                else:
                    mask = np.ones_like(occ_labels, dtype=np.bool_)
            else:
                mask = None

        return _sample_hist_from_sparse(
            occ_labels,
            mask,
            result_dict['occ_loc'],
            result_dict['sem_pred'],
            num_classes=num_classes,
            empty_label=self.empty_label)

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = _ordered_results(results, len(data_infos))

        num_classes = self.empty_label + 1
        if self.class_names is not None:
            num_classes = max(num_classes, len(self.class_names))

        hist = np.zeros((num_classes, num_classes), dtype=np.float64)
        total_samples = min(len(data_infos), len(ordered))
        num_workers = int(self.miou_num_workers or 0)

        if num_workers <= 1 or total_samples <= 1:
            for idx in range(total_samples):
                info = data_infos[idx]
                result_dict = ordered[idx]['pred']
                hist += self._sample_hist(info, result_dict, num_classes)
        else:
            max_workers = min(num_workers, total_samples)
            max_pending = max_workers * 2

            def _sample_args():
                for sample_idx in range(total_samples):
                    yield sample_idx, data_infos[sample_idx], ordered[sample_idx]['pred']

            arg_iter = iter(_sample_args())
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in range(max_pending):
                    try:
                        sample_idx, info, result_dict = next(arg_iter)
                    except StopIteration:
                        break
                    future = executor.submit(self._sample_hist, info, result_dict, num_classes)
                    futures[future] = sample_idx

                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        sample_idx = futures.pop(future)
                        try:
                            hist += future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f'Failed to compute mIoU histogram for sample index {sample_idx}.'
                            ) from exc

                    for _ in range(len(done)):
                        try:
                            sample_idx, info, result_dict = next(arg_iter)
                        except StopIteration:
                            break
                        future = executor.submit(self._sample_hist, info, result_dict, num_classes)
                        futures[future] = sample_idx

        iou_vector = _compute_iou_vector(hist)
        metrics = {
            'mIoU': _compute_miou(hist, self.empty_label),
            'IoU': round(_compute_occupied_iou(hist, self.empty_label) * 100, 2),
        }

        class_iou_items = []
        for class_idx in _valid_class_indices(num_classes, self.empty_label):
            class_name = self._class_name(class_idx)
            class_iou = round(float(iou_vector[class_idx] * 100.0), 2)
            metrics[f'IoU/{class_name}'] = class_iou
            class_iou_items.append((class_name, class_iou))

        self._append_focus_metrics(metrics, hist, iou_vector)
        self._log_per_class_iou(class_iou_items, metrics)

        return metrics


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    default_prefix = 'occ'

    def __init__(self,
                 ann_file,
                 occ_root,
                 empty_label=16,
                 occ_path_template='scene_{scene_token}/occupancy/{lidar_token}.npy',
                 src_class_names=None,
                 ignore_class_names=None,
                 class_names=None,
                 pc_range=None,
                 voxel_size=None,
                 collect_device='cpu',
                 prefix=None,
                 collect_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.occ_root = occ_root
        self.empty_label = int(empty_label)

        self.occ_path_template = occ_path_template
        self.src_class_names = src_class_names or [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]
        self.ignore_class_names = ignore_class_names or ['noise']
        self.class_names = class_names

        self.pc_range = np.asarray(
            pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3],
            dtype=np.float32)
        self.voxel_size = np.asarray(
            voxel_size if voxel_size is not None else [0.2, 0.2, 0.2],
            dtype=np.float32)

    def process(self, data_batch, data_samples):
        batch_samples = data_batch.get('data_samples', []) if isinstance(data_batch, dict) else []
        for idx, pred in enumerate(data_samples):
            sample = batch_samples[idx] if idx < len(batch_samples) else None
            meta = getattr(sample, 'metainfo', {}) if sample is not None else {}
            self.results.append(dict(
                pred=pred,
                sample_idx=meta.get('sample_idx', None),
                scene_token=meta.get('scene_token', None),
                lidar_token=meta.get('lidar_token', None),
            ))

    def compute_metrics(self, results):
        data_infos = _load_infos(self.ann_file)
        ordered = _ordered_results(results, len(data_infos))

        scene_size = self.pc_range[3:] - self.pc_range[:3]
        voxel_num = (scene_size / self.voxel_size).astype(np.int64)

        raw_empty_label = len(self.src_class_names)
        mapper = np.full(raw_empty_label + 1, self.empty_label, dtype=np.int64)

        active_class_names = [
            name for name in self.src_class_names
            if name not in self.ignore_class_names
        ]
        class_to_idx = {name: idx for idx, name in enumerate(active_class_names)}
        ignore_label_ids = []
        for src_idx, src_name in enumerate(self.src_class_names):
            if src_name in class_to_idx:
                mapper[src_idx] = class_to_idx[src_name]
            else:
                ignore_label_ids.append(src_idx)
        mapper[raw_empty_label] = self.empty_label

        num_classes = max(self.empty_label + 1, np.max(mapper) + 1)
        hist = np.zeros((num_classes, num_classes), dtype=np.float64)

        for idx, info in enumerate(data_infos):
            if idx >= len(ordered):
                break
            result_dict = ordered[idx]['pred']

            occ_file = _build_occ_path(self.occ_root, self.occ_path_template, info)
            occ_labels = np.load(occ_file)
            coors, labels = occ_labels[:, :3], occ_labels[:, 3].astype(np.int64)

            raw_dense = np.full(voxel_num, raw_empty_label, dtype=np.int64)
            raw_dense[coors[:, 2], coors[:, 1], coors[:, 0]] = labels

            mask = np.ones_like(raw_dense, dtype=np.bool_)
            for ignore_label in ignore_label_ids:
                mask &= raw_dense != ignore_label

            gt_dense = mapper[raw_dense]
            pred_dense, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                voxel_num,
                self.empty_label)
            pred_dense = np.asarray(pred_dense, dtype=np.int64)

            hist += _compute_hist(gt_dense, pred_dense, num_classes, mask=mask)

        return {
            'mIoU': _compute_miou(hist, self.empty_label),
            'IoU': round(_compute_occupied_iou(hist, self.empty_label) * 100, 2),
        }
