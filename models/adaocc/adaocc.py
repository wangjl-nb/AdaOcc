import time
import queue
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.dist import get_dist_info
from mmcv.cnn import ConvModule
from mmcv.ops import Voxelization
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from ..compat import auto_fp16, cast_tensor_type
from ..utils import (GridMask, pad_multiple, GpuPhotoMetricDistortion,
                     disable_all_fp16_function, debug_is_finite)
from .online_depth import OnlineDepthAnythingPoints


@MODELS.register_module()
class AdaOcc(MVXTwoStageDetector):
    '''
        specifically for 2D SECOND

        Adding:
            2D feature

    '''
    def __init__(self,
                 use_grid_mask=True,
                 data_aug=None,
                 stop_prev_grad=0,
                 train_view_dropout=None,
                 drop_lidar_feat=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_encoder=None,
                 img_feature_fusion=None,
                 use_external_img_encoder=False,
                 enable_pts_feature_branch=True,
                 enable_tpv_feature_branch=False,
                 tpv_encoder=None,
                 external_img_cache=False,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 second_out_dim=512,
                 pts_feat_dim=256,
                 voxel_head_phase_cfg=None,
                 online_depth=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs,
        )
        if pretrained is not None and init_cfg is None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.use_external_img_encoder = bool(use_external_img_encoder)
        self.enable_pts_feature_branch = bool(enable_pts_feature_branch)
        self.enable_tpv_feature_branch = bool(enable_tpv_feature_branch)
        self.external_img_cache = bool(external_img_cache)
        self.img_encoder = MODELS.build(img_encoder) if img_encoder is not None else None
        self.tpv_encoder = None
        if self.enable_tpv_feature_branch:
            if tpv_encoder is None:
                raise ValueError('tpv_encoder must be provided when enable_tpv_feature_branch=True')
            self.tpv_encoder = MODELS.build(tpv_encoder)
        if self.use_external_img_encoder and self.img_encoder is None:
            raise ValueError('img_encoder must be provided when use_external_img_encoder=True')
        self.img_feature_fusion_cfg = copy.deepcopy(img_feature_fusion) if img_feature_fusion else None
        self.use_img_feature_fusion = bool(
            self.img_feature_fusion_cfg is not None
            and self.img_encoder is not None
            and not self.use_external_img_encoder)
        img_encoder_cfg_freeze = False
        img_encoder_freeze_via_wrapper = False
        if isinstance(img_encoder, dict):
            img_encoder_cfg_freeze = bool(img_encoder.get('freeze', False))
            # Let the wrapper selectively freeze heavy submodules while keeping
            # lightweight adaptors/projections trainable.
            img_encoder_freeze_via_wrapper = bool(
                img_encoder.get('freeze_via_wrapper', False))
        fusion_cfg_freeze_img_encoder = bool(
            self.img_feature_fusion_cfg is not None
            and self.img_feature_fusion_cfg.get('freeze_img_encoder', True))
        self._freeze_img_encoder = bool(
            (img_encoder_cfg_freeze and not img_encoder_freeze_via_wrapper) or
            (fusion_cfg_freeze_img_encoder and not img_encoder_freeze_via_wrapper))

        self.img_fusion_proj = None
        self.img_fusion_concat_proj = None
        self.img_fusion_concat_act = None
        self.img_fusion_concat_se = None
        self.img_fusion_mode = 'weighted_sum'
        self.img_fusion_interp_mode = 'bilinear'
        self.img_fusion_align_corners = False
        if self.use_img_feature_fusion:
            if not self.with_img_backbone:
                raise ValueError('img_feature_fusion requires img_backbone to be enabled')
            fusion_out_channels = self._infer_img_fusion_out_channels(
                img_neck_cfg=img_neck,
                pts_bbox_head_cfg=pts_bbox_head)
            map_in_channels = int(self.img_feature_fusion_cfg.get('map_in_channels', 1024))
            self.img_fusion_proj = nn.Conv2d(
                map_in_channels,
                fusion_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)
            self.img_fusion_mode = str(
                self.img_feature_fusion_cfg.get('mode', 'weighted_sum')).lower()
            valid_fusion_modes = ('weighted_sum', 'concat_proj')
            if self.img_fusion_mode not in valid_fusion_modes:
                raise ValueError(
                    f'img_feature_fusion.mode must be one of {valid_fusion_modes}, '
                    f'got {self.img_fusion_mode}')
            if self.img_fusion_mode == 'concat_proj':
                self.img_fusion_concat_proj = nn.Conv2d(
                    fusion_out_channels * 2,
                    fusion_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
                if bool(self.img_feature_fusion_cfg.get('concat_use_act', True)):
                    self.img_fusion_concat_act = nn.ReLU(inplace=True)
                concat_se_reduction = int(
                    self.img_feature_fusion_cfg.get('concat_se_reduction', 16))
                concat_se_min_channels = int(
                    self.img_feature_fusion_cfg.get('concat_se_min_channels', 8))
                if concat_se_reduction <= 0:
                    raise ValueError(
                        f'concat_se_reduction must be positive, got {concat_se_reduction}')
                if concat_se_min_channels <= 0:
                    raise ValueError(
                        f'concat_se_min_channels must be positive, got {concat_se_min_channels}')
                concat_se_hidden = max(
                    fusion_out_channels // concat_se_reduction,
                    concat_se_min_channels)
                self.img_fusion_concat_se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(fusion_out_channels, concat_se_hidden, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(concat_se_hidden, fusion_out_channels, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
            self.img_fusion_interp_mode = self.img_feature_fusion_cfg.get('interp_mode', 'bilinear')
            self.img_fusion_align_corners = bool(
                self.img_feature_fusion_cfg.get('align_corners', False))
        if self._freeze_img_encoder and self.img_encoder is not None:
            self._freeze_module(self.img_encoder)

        self.pts_voxel_layer = None
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.drop_lidar_feat = bool(drop_lidar_feat)
        self.online_depth_cfg = copy.deepcopy(online_depth) if online_depth else {}
        self.online_depth_enabled = bool(self.online_depth_cfg.get('enabled', False))
        self.online_depth_mode = str(self.online_depth_cfg.get('mode', 'serial')).lower()
        if self.online_depth_mode not in ('serial', 'parallel'):
            raise ValueError(f'online_depth.mode must be serial/parallel, got {self.online_depth_mode}')
        online_depth_module_cfg = copy.deepcopy(self.online_depth_cfg)
        online_depth_module_cfg.pop('mode', None)
        self.online_depth = OnlineDepthAnythingPoints(**online_depth_module_cfg) \
            if self.online_depth_enabled else None
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.train_view_dropout = copy.deepcopy(train_view_dropout) if train_view_dropout else {}
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask

        self.memory = {}
        self.queue = queue.Queue()
        self.voxel_head_phase_cfg = copy.deepcopy(voxel_head_phase_cfg) if voxel_head_phase_cfg else {}
        self._voxel_head_phase_keep_paths = tuple()
        self._voxel_head_phase_enabled = False

        self.final_conv = None
        if self.enable_pts_feature_branch:
            self.final_conv = ConvModule(
                second_out_dim,
                pts_feat_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d')
            )

        self.pts_feat_dim=pts_feat_dim
        if not self.enable_pts_feature_branch:
            # In TPV mode, voxel+middle encoders are still used.
            if not self.enable_tpv_feature_branch:
                self._freeze_module(self.pts_voxel_encoder)
                self._freeze_module(self.pts_middle_encoder)
            else:
                # TPV branch uses middle encoder's encode_features, not its conv_out path.
                self._freeze_module(getattr(self.pts_middle_encoder, 'conv_out', None))
            self._freeze_module(self.pts_backbone)
            self._freeze_module(self.pts_neck)
            self._prune_unused_pts_modules()

        if self.enable_tpv_feature_branch:
            if self.pts_voxel_layer is None or self.pts_voxel_encoder is None or self.pts_middle_encoder is None:
                raise ValueError(
                    'TPV branch requires pts_voxel_layer, pts_voxel_encoder, and pts_middle_encoder.')
            if not bool(getattr(self.pts_middle_encoder, 'return_middle_feats', False)):
                raise ValueError(
                    'TPV branch requires pts_middle_encoder.return_middle_feats=True.')

        transformer = getattr(self.pts_bbox_head, 'transformer', None)
        if transformer is not None:
            if bool(getattr(transformer, 'use_tpv_sampling', False)) and (not self.enable_tpv_feature_branch):
                raise ValueError(
                    'transformer.use_tpv_sampling=True requires enable_tpv_feature_branch=True')
            if bool(getattr(transformer, 'use_pts_sampling', False)) and (not self.enable_pts_feature_branch):
                raise ValueError(
                    'transformer.use_pts_sampling=True requires enable_pts_feature_branch=True')
        self._apply_voxel_head_phase_freeze(self.voxel_head_phase_cfg)

    @staticmethod
    def _freeze_module(module):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()

    @staticmethod
    def _infer_img_fusion_out_channels(img_neck_cfg, pts_bbox_head_cfg):
        if isinstance(img_neck_cfg, dict):
            out_channels = img_neck_cfg.get('out_channels', None)
            if isinstance(out_channels, (list, tuple)) and len(out_channels) > 0:
                return int(out_channels[0])
            if out_channels is not None:
                return int(out_channels)
        if isinstance(pts_bbox_head_cfg, dict):
            in_channels = pts_bbox_head_cfg.get('in_channels', None)
            if in_channels is not None:
                return int(in_channels)
        return 256

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_img_encoder and self.img_encoder is not None:
            self.img_encoder.eval()
        if self.online_depth is not None:
            self.online_depth.eval()
        self._apply_voxel_head_phase_train_modes(mode=mode)
        return self

    @staticmethod
    def _unfreeze_module(module):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = True
        module.train()

    def _resolve_named_module(self, dotted_path):
        if not dotted_path:
            return self
        module = self
        for part in dotted_path.split('.'):
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and part.isdigit():
                module = module[int(part)]
            elif hasattr(module, part):
                module = getattr(module, part)
            else:
                raise AttributeError(f'Cannot resolve module path "{dotted_path}" at "{part}"')
        if not isinstance(module, nn.Module):
            raise TypeError(f'Resolved object for "{dotted_path}" is not nn.Module: {type(module)}')
        return module

    def _apply_voxel_head_phase_freeze(self, phase_cfg):
        cfg = copy.deepcopy(phase_cfg) if phase_cfg else {}
        self._voxel_head_phase_enabled = bool(cfg.get('enabled', False) and cfg.get('freeze_base', False))
        self._voxel_head_phase_keep_paths = tuple()
        if not self._voxel_head_phase_enabled:
            return

        for parameter in self.parameters():
            parameter.requires_grad = False

        keep_paths = list(cfg.get('keep_trainable_modules', []))
        unfreeze_last = int(cfg.get('unfreeze_decoder_last_n_layers', 0) or 0)
        if unfreeze_last > 0:
            decoder_layers = getattr(
                getattr(getattr(self.pts_bbox_head, 'transformer', None), 'decoder', None),
                'decoder_layers',
                None,
            )
            if decoder_layers is None:
                raise ValueError(
                    'voxel_head_phase_cfg.unfreeze_decoder_last_n_layers requires '
                    'pts_bbox_head.transformer.decoder.decoder_layers')
            total_layers = len(decoder_layers)
            unfreeze_last = min(unfreeze_last, total_layers)
            for layer_idx in range(total_layers - unfreeze_last, total_layers):
                keep_paths.append(f'pts_bbox_head.transformer.decoder.decoder_layers.{layer_idx}')

        dedup_keep_paths = []
        seen = set()
        for path in keep_paths:
            path = str(path)
            if path not in seen:
                dedup_keep_paths.append(path)
                seen.add(path)

        for path in dedup_keep_paths:
            module = self._resolve_named_module(path)
            self._unfreeze_module(module)

        self._voxel_head_phase_keep_paths = tuple(dedup_keep_paths)

    def _apply_voxel_head_phase_train_modes(self, mode=True):
        if not mode or not self._voxel_head_phase_enabled:
            return
        keep_paths = tuple(self._voxel_head_phase_keep_paths)
        if not keep_paths:
            return
        for module_path, module in self.named_modules():
            if module_path == '':
                continue
            if any(module_path == keep or keep.startswith(module_path + '.') for keep in keep_paths):
                continue
            module.eval()

    def _need_img_branch(self):
        return self.use_external_img_encoder or self.with_img_backbone

    def _prune_unused_pts_modules(self):
        if self.enable_pts_feature_branch:
            return
        if hasattr(self, 'pts_backbone'):
            self.pts_backbone = None
        if hasattr(self, 'pts_neck'):
            self.pts_neck = None
        self.final_conv = None

    def _maybe_drop_lidar_feat(self, pts_feats):
        if pts_feats is None:
            return None
        if not self.drop_lidar_feat:
            return pts_feats
        return torch.zeros_like(pts_feats)

    @staticmethod
    def _get_level_weights(weight_cfg, num_levels, name):
        if isinstance(weight_cfg, (float, int)):
            return [float(weight_cfg) for _ in range(num_levels)]
        if isinstance(weight_cfg, tuple):
            weight_cfg = list(weight_cfg)
        if not isinstance(weight_cfg, list):
            raise TypeError(
                f'{name} must be float/int or list/tuple of floats, got {type(weight_cfg)}')
        if len(weight_cfg) == 1:
            return [float(weight_cfg[0]) for _ in range(num_levels)]
        if len(weight_cfg) != num_levels:
            raise ValueError(
                f'{name} length mismatch: expected 1 or {num_levels}, got {len(weight_cfg)}')
        return [float(v) for v in weight_cfg]

    def _normalize_points_list(self, points):
        if points is None:
            return None
        if isinstance(points, torch.Tensor):
            if points.dim() == 2:
                return [points]
            if points.dim() != 3:
                raise TypeError(
                    'points Tensor input must have shape [N, C] or [B, N, C] when passed to external img encoder, '
                    f'but got {tuple(points.shape)}')
            return [points[i] for i in range(points.shape[0])]
        if isinstance(points, tuple):
            points = list(points)
        if not isinstance(points, list):
            raise TypeError(
                'points for external img encoder must be None, Tensor[B,N,C], or list of tensors, '
                f'but got {type(points)}')
        normalized = []
        for pts in points:
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            normalized.append(pts)
        return normalized

    @staticmethod
    def _normalize_img_batch_input(img):
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            return img
        if isinstance(img, tuple):
            img = list(img)
        if isinstance(img, list):
            tensors = []
            for item in img:
                if isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        f'Unsupported image batch item type: {type(item)}')
                tensors.append(item)
            if not tensors:
                raise ValueError('Empty image batch list is not supported')
            return torch.stack(tensors, dim=0)
        raise TypeError(f'Unsupported image batch type: {type(img)}')

    def _normalize_view_id_list(self, view_ids, batch_size):
        if view_ids is None:
            return None
        if isinstance(view_ids, torch.Tensor):
            if view_ids.dim() == 1 and batch_size == 1:
                return [view_ids]
            if view_ids.dim() == 2 and view_ids.shape[0] == batch_size:
                return [view_ids[i] for i in range(batch_size)]
            raise TypeError(
                f'depth_point_view_ids tensor must have shape [N] or [B, N], got {tuple(view_ids.shape)}')
        if isinstance(view_ids, np.ndarray):
            if view_ids.ndim == 1 and batch_size == 1:
                return [torch.from_numpy(view_ids)]
            if view_ids.ndim == 2 and view_ids.shape[0] == batch_size:
                return [torch.from_numpy(view_ids[i]) for i in range(batch_size)]
            raise TypeError(
                f'depth_point_view_ids ndarray must have shape [N] or [B, N], got {view_ids.shape}')
        if isinstance(view_ids, tuple):
            view_ids = list(view_ids)
        if not isinstance(view_ids, list):
            raise TypeError(
                f'depth_point_view_ids must be Tensor/ndarray/list, got {type(view_ids)}')
        if len(view_ids) != batch_size:
            raise ValueError(
                f'depth_point_view_ids batch size mismatch: got {len(view_ids)}, expected {batch_size}')
        normalized = []
        for item in view_ids:
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)
            elif not isinstance(item, torch.Tensor):
                item = torch.as_tensor(item)
            normalized.append(item.long())
        return normalized

    @staticmethod
    def _is_single_transform_matrix(value):
        if isinstance(value, np.ndarray):
            return value.ndim == 2 and value.shape == (4, 4)
        if isinstance(value, torch.Tensor):
            return value.dim() == 2 and tuple(value.shape) == (4, 4)
        return False

    @staticmethod
    def _slice_meta_value_by_indices(value, indices, total_views):
        if isinstance(value, list) and len(value) == total_views:
            return [copy.deepcopy(value[idx]) for idx in indices]
        if isinstance(value, tuple) and len(value) == total_views:
            return tuple(copy.deepcopy(value[idx]) for idx in indices)
        if AdaOcc._is_single_transform_matrix(value):
            return copy.deepcopy(value)
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_views:
            return np.array(value[indices], copy=True)
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == total_views:
            return value[indices]
        return copy.deepcopy(value)

    def _gather_img_metas_by_indices(self, img_metas, indices, total_views):
        gathered = []
        indices_list = [int(i) for i in indices]
        for meta in img_metas:
            if not isinstance(meta, dict):
                gathered.append(meta)
                continue
            gathered_meta = {}
            for key, value in meta.items():
                gathered_meta[key] = self._slice_meta_value_by_indices(
                    value, indices_list, total_views)
            gathered.append(gathered_meta)
        return gathered

    def _mask_inactive_ego2img(self, img_metas, total_views, active_indices,
                               kept_camera_ids, kept_camera_names):
        active_indices = {int(idx) for idx in active_indices}
        masked = []
        for meta in img_metas:
            if not isinstance(meta, dict):
                masked.append(meta)
                continue
            out = copy.deepcopy(meta)
            ego2img = out.get('ego2img', None)
            if isinstance(ego2img, list) and len(ego2img) == total_views:
                zero = np.zeros((4, 4), dtype=np.float32)
                new_ego2img = []
                for idx, item in enumerate(ego2img):
                    if idx in active_indices:
                        new_ego2img.append(np.asarray(item, dtype=np.float32))
                    else:
                        new_ego2img.append(zero.copy())
                out['ego2img'] = new_ego2img
            out['dynamic_view_keep_ids'] = [int(x) for x in kept_camera_ids]
            out['dynamic_view_keep_names'] = list(kept_camera_names)
            masked.append(out)
        return masked

    @staticmethod
    def _gather_view_extra_by_indices(view_extra, indices):
        if view_extra is None:
            return None
        indices_list = [int(i) for i in indices]
        gathered = []
        for item in view_extra:
            if item is None:
                gathered.append(None)
                continue
            out = copy.deepcopy(item)
            if 'views' in out and isinstance(out['views'], (list, tuple)):
                out['views'] = [copy.deepcopy(out['views'][idx]) for idx in indices_list]
            gathered.append(out)
        return gathered

    @staticmethod
    def _scatter_img_feats_to_full_views(img_feats, active_indices, full_total_views):
        scattered = []
        for feat in img_feats:
            if not isinstance(feat, torch.Tensor) or feat.dim() != 5:
                raise TypeError(
                    f'Image feature must be Tensor[B, TN, C, H, W], got {type(feat)} '
                    f'shape={getattr(feat, "shape", None)}')
            B, _, C, H, W = feat.shape
            feat_active_indices = torch.as_tensor(
                active_indices, dtype=torch.long, device=feat.device)
            full = feat.new_zeros((B, full_total_views, C, H, W))
            full[:, feat_active_indices] = feat
            scattered.append(full)
        return scattered

    @staticmethod
    def _points_non_empty(points):
        if points is None:
            return True
        return all(pts is not None and pts.shape[0] > 0 for pts in points)

    def _mask_has_non_empty_gt(self, voxel_semantics, mask_camera):
        if voxel_semantics is None or mask_camera is None:
            return True
        empty_label = int(getattr(getattr(self, 'pts_bbox_head', None), 'empty_label', -1))
        if empty_label < 0:
            return True
        visible_non_empty = (voxel_semantics.long() != empty_label) & mask_camera.bool()
        if visible_non_empty.dim() <= 1:
            return bool(visible_non_empty.any().item())
        valid_per_sample = visible_non_empty.reshape(visible_non_empty.shape[0], -1).any(dim=1)
        return bool(valid_per_sample.all().item())

    @staticmethod
    def _filter_points_by_view_ids(points, point_view_ids, active_indices):
        if points is None:
            return None
        if point_view_ids is None:
            raise ValueError('depth_point_view_ids are required when train_view_dropout is enabled')
        active_indices = active_indices.to(dtype=torch.long)
        filtered_points = []
        for pts, view_ids in zip(points, point_view_ids):
            pts = pts if isinstance(pts, torch.Tensor) else torch.as_tensor(pts)
            view_ids = view_ids.to(device=pts.device, dtype=torch.long)
            mask = (view_ids[:, None] == active_indices[None, :].to(device=pts.device)).any(dim=1)
            filtered_points.append(pts[mask])
        return filtered_points

    @staticmethod
    def _build_mask_from_bits(mask_camera_bits, camera_names, selected_names):
        bitmask = 0
        selected_names = set(selected_names)
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_name in selected_names:
                bitmask |= (1 << cam_idx)
        return (mask_camera_bits.to(dtype=torch.int64) & bitmask) != 0

    def _rebuild_mask_camera_from_bits(self, mask_camera_bits, mask_camera_names, kept_camera_names):
        rebuilt = []
        for bits, names in zip(mask_camera_bits, mask_camera_names):
            if names is None:
                raise ValueError(
                    'mask_camera_names metadata is required when train_view_dropout is enabled')
            rebuilt.append(self._build_mask_from_bits(bits, names, kept_camera_names))
        return torch.stack(rebuilt, dim=0)

    def _sample_kept_camera_ids(self, base_num_views, device):
        cfg = self.train_view_dropout or {}
        min_keep, max_keep = cfg.get('keep_count_range', [1, base_num_views])
        min_keep = max(1, min(int(min_keep), base_num_views))
        max_keep = max(min_keep, min(int(max_keep), base_num_views))
        keep_count = int(torch.randint(min_keep, max_keep + 1, (1,), device=device).item())
        kept = torch.randperm(base_num_views, device=device)[:keep_count]
        kept, _ = torch.sort(kept)
        return kept

    @staticmethod
    def _expand_active_tn_indices(kept_camera_ids, base_num_views, num_frames):
        chunks = []
        for frame_idx in range(num_frames):
            chunks.append(kept_camera_ids + frame_idx * base_num_views)
        return torch.cat(chunks, dim=0)

    def _prepare_train_view_dropout(self,
                                    img,
                                    points,
                                    depth_point_view_ids,
                                    view_extra,
                                    img_metas,
                                    voxel_semantics,
                                    mask_camera,
                                    mask_camera_bits,
                                    mask_camera_names):
        cfg = self.train_view_dropout or {}
        if (not self.training) or (not cfg.get('enabled', False)):
            return dict(
                img=img,
                points=points,
                view_extra=view_extra,
                encoder_img_metas=img_metas,
                head_img_metas=img_metas,
                runtime_num_views=None,
                mask_camera=mask_camera,
                active_tn_indices=None,
                kept_camera_ids=None,
                kept_camera_names=None,
            )

        if not self.use_external_img_encoder:
            raise ValueError('train_view_dropout currently requires use_external_img_encoder=True')
        if mask_camera_bits is None:
            raise ValueError(
                'train_view_dropout requires GT files to provide mask_camera_bits '
                '(load them via LoadOcc3DFromFile)')
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise TypeError(
                f'train_view_dropout expects img Tensor[B, TN, C, H, W], got {type(img)} '
                f'shape={getattr(img, "shape", None)}')

        B, total_views = img.shape[:2]
        base_num_views = self._get_num_views()
        if total_views % base_num_views != 0:
            raise ValueError(
                f'train_view_dropout expects TN divisible by num_views={base_num_views}, got TN={total_views}')
        num_frames = total_views // base_num_views
        camera_pool = list(cfg.get('camera_pool', [f'CAM_{i}' for i in range(base_num_views)]))
        if len(camera_pool) != base_num_views:
            raise ValueError(
                f'train_view_dropout.camera_pool length mismatch: expected {base_num_views}, got {len(camera_pool)}')

        points_list = self._normalize_points_list(points)
        view_id_list = self._normalize_view_id_list(depth_point_view_ids, B) if points_list is not None else None

        max_attempts = int(cfg.get('max_resample_attempts', 8))
        fallback_to_all = bool(cfg.get('fallback_to_all_if_empty_points', True))
        kept_camera_ids = None
        active_tn_indices = None
        filtered_points = points_list
        effective_mask_camera = mask_camera
        attempts_used = 0
        rejected_empty_points = 0
        rejected_empty_gt = 0
        fallback_used = False
        for _ in range(max_attempts):
            attempts_used += 1
            kept_camera_ids = self._sample_kept_camera_ids(base_num_views, img.device)
            active_tn_indices = self._expand_active_tn_indices(
                kept_camera_ids, base_num_views=base_num_views, num_frames=num_frames)
            filtered_points = self._filter_points_by_view_ids(
                points_list,
                view_id_list,
                active_tn_indices) if points_list is not None else None
            candidate_mask_camera = mask_camera
            if mask_camera_bits is not None:
                candidate_mask_camera = self._rebuild_mask_camera_from_bits(
                    mask_camera_bits,
                    mask_camera_names,
                    [camera_pool[int(idx)] for idx in kept_camera_ids.tolist()])
            points_ok = self._points_non_empty(filtered_points)
            gt_ok = self._mask_has_non_empty_gt(voxel_semantics, candidate_mask_camera)
            if not points_ok:
                rejected_empty_points += 1
            if not gt_ok:
                rejected_empty_gt += 1
            if points_ok and gt_ok:
                effective_mask_camera = candidate_mask_camera
                break
        else:
            if not fallback_to_all:
                raise FloatingPointError(
                    'train_view_dropout could not find a valid camera subset with non-empty '
                    'points and GT; set fallback_to_all_if_empty_points=True or relax keep_count_range')
            fallback_used = True
            kept_camera_ids = torch.arange(base_num_views, device=img.device, dtype=torch.long)
            active_tn_indices = self._expand_active_tn_indices(
                kept_camera_ids, base_num_views=base_num_views, num_frames=num_frames)
            filtered_points = self._filter_points_by_view_ids(
                points_list,
                view_id_list,
                active_tn_indices) if points_list is not None else None
            effective_mask_camera = mask_camera
            if mask_camera_bits is not None:
                effective_mask_camera = self._rebuild_mask_camera_from_bits(
                    mask_camera_bits,
                    mask_camera_names,
                    [camera_pool[int(idx)] for idx in kept_camera_ids.tolist()])

        kept_camera_names = [camera_pool[int(idx)] for idx in kept_camera_ids.tolist()]
        encoder_img = img.index_select(1, active_tn_indices.to(dtype=torch.long))
        encoder_img_metas = self._gather_img_metas_by_indices(
            img_metas, active_tn_indices.tolist(), total_views)
        encoder_view_extra = self._gather_view_extra_by_indices(
            view_extra, active_tn_indices.tolist())
        head_img_metas = self._mask_inactive_ego2img(
            img_metas,
            total_views=total_views,
            active_indices=active_tn_indices.tolist(),
            kept_camera_ids=kept_camera_ids.tolist(),
            kept_camera_names=kept_camera_names)

        return dict(
            img=encoder_img,
            points=filtered_points,
            view_extra=encoder_view_extra,
            encoder_img_metas=encoder_img_metas,
            head_img_metas=head_img_metas,
            runtime_num_views=int(kept_camera_ids.numel()),
            mask_camera=effective_mask_camera,
            active_tn_indices=active_tn_indices,
            kept_camera_ids=kept_camera_ids,
            kept_camera_names=kept_camera_names,
            full_total_views=total_views,
            view_dropout_keep_count=int(kept_camera_ids.numel()),
            view_dropout_attempts=attempts_used,
            view_dropout_fallback=int(fallback_used),
            view_dropout_reject_empty_points=rejected_empty_points,
            view_dropout_reject_empty_gt=rejected_empty_gt,
        )

    def _run_external_img_encoder(self, img, points, img_metas, view_extra=None,
                                  runtime_num_views=None):
        points = self._normalize_points_list(points)
        if self._freeze_img_encoder:
            with torch.no_grad():
                return self.img_encoder(
                    img,
                    points=points,
                    img_metas=img_metas,
                    view_extra=view_extra,
                    runtime_num_views=runtime_num_views)
        return self.img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            view_extra=view_extra,
            runtime_num_views=runtime_num_views)

    def _build_external_encoder_runtime_img_metas(self, img, img_metas, runtime_num_views=None):
        if img_metas is None or img is None:
            return None
        if not (self.use_external_img_encoder or self.use_img_feature_fusion):
            return None
        builder = getattr(self.img_encoder, 'build_runtime_img_metas', None)
        if not callable(builder):
            return None
        return builder(img=img, img_metas=img_metas, runtime_num_views=runtime_num_views)

    def _extract_external_img_feat_tensor(self, img, points, img_metas, view_extra=None,
                                          runtime_num_views=None):
        img_feats = self._run_external_img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            view_extra=view_extra,
            runtime_num_views=runtime_num_views)
        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) == 1 and isinstance(img_feats[0], torch.Tensor):
                img_feats = img_feats[0]
            else:
                raise ValueError(
                    'External img encoder must output a single Tensor[B, TN, C, H, W] '
                    'when img_feature_fusion is enabled.')
        if not isinstance(img_feats, torch.Tensor) or img_feats.dim() != 5:
            raise ValueError(
                'External img encoder output must be Tensor[B, TN, C, H, W], '
                f"but got type={type(img_feats)} shape={getattr(img_feats, 'shape', None)}")
        return img_feats

    def _extract_external_img_feat(self, img, points, img_metas, view_extra=None,
                                   runtime_num_views=None):
        img_feats = self._run_external_img_encoder(
            img,
            points=points,
            img_metas=img_metas,
            view_extra=view_extra,
            runtime_num_views=runtime_num_views)

        if isinstance(img_feats, (list, tuple)):
            if len(img_feats) == 1 and isinstance(img_feats[0], torch.Tensor):
                img_feats = img_feats[0]
            else:
                return list(img_feats)

        if not isinstance(img_feats, torch.Tensor) or img_feats.dim() != 5:
            raise ValueError(
                'External img encoder output must be Tensor[B, TN, C, H, W] or single-element list, '
                f"but got type={type(img_feats)} shape={getattr(img_feats, 'shape', None)}")

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        else:
            img_feats = [img_feats]

        if not isinstance(img_feats, (list, tuple)):
            raise TypeError(f'img_neck output must be list/tuple, got {type(img_feats)}')
        for lvl, feat in enumerate(img_feats):
            if not isinstance(feat, torch.Tensor) or feat.dim() != 5:
                raise ValueError(
                    f'Image feature at level {lvl} must be Tensor[B, TN, C, H, W], '
                    f"got type={type(feat)} shape={getattr(feat, 'shape', None)}")
        return list(img_feats)

    def _interpolate_external_feat(self, feat_2d, target_hw):
        if feat_2d.shape[-2:] == target_hw:
            return feat_2d
        if self.img_fusion_interp_mode in ('nearest', 'area', 'nearest-exact'):
            return F.interpolate(
                feat_2d,
                size=target_hw,
                mode=self.img_fusion_interp_mode)
        return F.interpolate(
            feat_2d,
            size=target_hw,
            mode=self.img_fusion_interp_mode,
            align_corners=self.img_fusion_align_corners)

    def _fuse_img_features(self, fpn_feats, external_feat):
        if not isinstance(fpn_feats, (list, tuple)) or len(fpn_feats) == 0:
            raise ValueError('fpn_feats must be a non-empty list of image features')
        if self.img_fusion_proj is None:
            return list(fpn_feats)
        if not isinstance(external_feat, torch.Tensor) or external_feat.dim() != 5:
            raise ValueError(
                'external_feat must be Tensor[B, TN, C, H, W], '
                f'but got type={type(external_feat)} shape={getattr(external_feat, "shape", None)}')

        B_ext, TN_ext, C_ext, H_ext, W_ext = external_feat.shape
        ext_feat_2d = external_feat.reshape(B_ext * TN_ext, C_ext, H_ext, W_ext)
        ext_feat_2d = ext_feat_2d.to(dtype=self.img_fusion_proj.weight.dtype)
        ext_feat_proj_2d = self.img_fusion_proj(ext_feat_2d)
        proj_channels = ext_feat_proj_2d.shape[1]

        num_levels = len(fpn_feats)
        alpha = self._get_level_weights(
            self.img_feature_fusion_cfg.get('alpha', 1.0),
            num_levels=num_levels,
            name='img_feature_fusion.alpha')
        beta = self._get_level_weights(
            self.img_feature_fusion_cfg.get('beta', 0.0),
            num_levels=num_levels,
            name='img_feature_fusion.beta')

        fused_feats = []
        for lvl, fpn_feat in enumerate(fpn_feats):
            if not isinstance(fpn_feat, torch.Tensor) or fpn_feat.dim() != 5:
                raise ValueError(
                    f'FPN feature level {lvl} must be Tensor[B, TN, C, H, W], '
                    f'got type={type(fpn_feat)} shape={getattr(fpn_feat, "shape", None)}')
            B_fpn, TN_fpn, C_fpn, H_fpn, W_fpn = fpn_feat.shape
            if B_fpn != B_ext or TN_fpn != TN_ext:
                raise ValueError(
                    f'Feature batch/TN mismatch at level {lvl}: '
                    f'fpn={tuple(fpn_feat.shape[:2])}, external={(B_ext, TN_ext)}')
            if C_fpn != proj_channels:
                raise ValueError(
                    f'Feature channel mismatch at level {lvl}: '
                    f'fpn C={C_fpn}, projected external C={proj_channels}. '
                    'Please check img_neck out_channels and img_feature_fusion.map_in_channels.')
            ext_lvl_2d = self._interpolate_external_feat(
                ext_feat_proj_2d,
                target_hw=(H_fpn, W_fpn))
            ext_lvl = ext_lvl_2d.view(B_fpn, TN_fpn, C_fpn, H_fpn, W_fpn)
            ext_lvl = ext_lvl.to(dtype=fpn_feat.dtype)
            if self.img_fusion_mode == 'weighted_sum':
                fused = alpha[lvl] * fpn_feat + beta[lvl] * ext_lvl
            else:
                if self.img_fusion_concat_proj is None:
                    raise RuntimeError(
                        'img_fusion_concat_proj is not initialized while mode=concat_proj')
                concat_2d = torch.cat(
                    [
                        (alpha[lvl] * fpn_feat).reshape(B_fpn * TN_fpn, C_fpn, H_fpn, W_fpn),
                        (beta[lvl] * ext_lvl).reshape(B_fpn * TN_fpn, C_fpn, H_fpn, W_fpn),
                    ],
                    dim=1)
                concat_2d = concat_2d.to(dtype=self.img_fusion_concat_proj.weight.dtype)
                fused_2d = self.img_fusion_concat_proj(concat_2d)
                if self.img_fusion_concat_act is not None:
                    fused_2d = self.img_fusion_concat_act(fused_2d)
                if self.img_fusion_concat_se is None:
                    raise RuntimeError(
                        'img_fusion_concat_se is not initialized while mode=concat_proj')
                fused_2d = fused_2d * self.img_fusion_concat_se(fused_2d)
                fused = fused_2d.view(B_fpn, TN_fpn, C_fpn, H_fpn, W_fpn).to(dtype=fpn_feat.dtype)
            fused_feats.append(fused)
        return fused_feats

    def _extract_pts_feat_for_head(self, points):
        if (not self.enable_pts_feature_branch) or (not self.with_pts_backbone):
            return None
        pts_feats = self.extract_pts_feat(points)
        return self.final_conv(pts_feats[0])

    def _extract_tpv_feat_for_head(self, points):
        if not self.enable_tpv_feature_branch:
            return None
        if self.tpv_encoder is None:
            raise RuntimeError('TPV branch is enabled but tpv_encoder is not initialized')
        if points is None:
            raise ValueError('points must be provided when enable_tpv_feature_branch=True')
        sparse_feats = self._extract_sparse_3d_feat_for_tpv(points)
        return self.tpv_encoder(sparse_feats)

    def _extract_sparse_3d_feat_for_tpv(self, points):
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = self._resolve_sparse_batch_size(points, coors)
        middle_out = self.pts_middle_encoder(voxel_features, coors, batch_size)

        if (not isinstance(middle_out, (tuple, list))) or len(middle_out) != 2:
            raise RuntimeError(
                'pts_middle_encoder must return (spatial_features, encode_features) '
                'for TPV branch. Please set return_middle_feats=True.')
        _, encode_features = middle_out
        if not isinstance(encode_features, (list, tuple)) or len(encode_features) == 0:
            raise RuntimeError('Invalid encode_features from pts_middle_encoder.')

        high_sparse = encode_features[-1]
        if not hasattr(high_sparse, 'dense'):
            raise TypeError('encode_features items must be sparse tensors supporting dense().')
        high_dense = high_sparse.dense()
        high_indices = getattr(high_sparse, 'indices', None)
        if high_indices is None:
            raise TypeError('encode_features items must expose sparse indices for TPV refinement.')

        # Pick the nearest higher-resolution stage as skip for 3D FPN fusion.
        skip_sparse = None
        high_shape = tuple(int(v) for v in high_dense.shape[-3:])
        for feat in reversed(encode_features[:-1]):
            feat_shape = getattr(feat, 'spatial_shape', None)
            if feat_shape is None:
                continue
            feat_shape = tuple(int(v) for v in feat_shape)
            if any(a > b for a, b in zip(feat_shape, high_shape)):
                skip_sparse = feat
                break
        skip_dense = skip_sparse.dense() if skip_sparse is not None else None
        return dict(
            high=high_dense,
            skip=skip_dense,
            high_indices=high_indices,
        )

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization on a batch of points."""
        assert self.pts_voxel_layer is not None, 'pts_voxel_layer is not set'
        voxels, num_points, coors = [], [], []
        for i, res in enumerate(points):
            if hasattr(res, 'tensor'):
                res = res.tensor
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            num_points.append(res_num_points)
            coors.append(res_coors)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)
        return voxels, num_points, coors

    @staticmethod
    def _resolve_sparse_batch_size(points, coors):
        if isinstance(points, (list, tuple)):
            return len(points)
        if coors is None or coors.numel() == 0:
            return 0
        return int(coors[:, 0].max().item()) + 1

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat_(self, img):
        # import pdb; pdb.set_trace()
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_img_feat(self, img, img_metas, points=None, view_extra=None,
                         runtime_num_views=None):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5
        img_for_external = img

        if self.use_external_img_encoder:
            return self._extract_external_img_feat(
                img_for_external,
                points=points,
                img_metas=img_metas,
                view_extra=view_extra,
                runtime_num_views=runtime_num_views)

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            num_views = self._get_num_views()
            img = img.reshape(B, -1, num_views, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat_(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat_(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, num_views, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat_(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        if self.use_img_feature_fusion:
            external_feat = self._extract_external_img_feat_tensor(
                img_for_external,
                points=points,
                img_metas=img_metas,
                view_extra=view_extra,
                runtime_num_views=runtime_num_views)
            img_feats_reshaped = self._fuse_img_features(img_feats_reshaped, external_feat)

        return img_feats_reshaped
    
    @auto_fp16(apply_to=('pts'), out_fp32=True)
    def extract_pts_feat(self, pts):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = self._resolve_sparse_batch_size(pts, coors)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        return self._forward(inputs, data_samples)

    def _forward(self, inputs, data_samples=None):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        img = self._normalize_img_batch_input(img)
        img_metas = self._collect_img_metas(data_samples)
        return self.extract_feat(img, img_metas)

    def loss(self, inputs, data_samples):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        img = self._normalize_img_batch_input(img)
        points = inputs.get('points') if isinstance(inputs, dict) else None
        depth_point_view_ids = inputs.get('depth_point_view_ids') if isinstance(inputs, dict) else None
        view_extra = inputs.get('view_extra') if isinstance(inputs, dict) else None
        online_depth_inputs = inputs.get('online_depth') if isinstance(inputs, dict) else None
        runtime_num_views = inputs.get('runtime_num_views') if isinstance(inputs, dict) else None
        view_dropout_preprocessed = bool(inputs.get('view_dropout_preprocessed', False)) \
            if isinstance(inputs, dict) else False
        preprocessed_view_dropout_state = inputs.get('view_dropout_state', None) \
            if isinstance(inputs, dict) else None
        if isinstance(img, torch.Tensor) and img.dim() >= 1:
            view_extra = self._normalize_view_extra(
                view_extra, batch_size=int(img.shape[0]))
        img_metas = self._collect_img_metas(data_samples)
        voxel_semantics = self._stack_data_samples(data_samples, 'voxel_semantics')
        raw_semantics = self._stack_data_samples(data_samples, 'raw_semantics')
        raw_voxel_origin = self._stack_data_samples(data_samples, 'voxel_origin')
        raw_voxel_size = self._stack_data_samples(data_samples, 'voxel_size')
        mask_camera = self._stack_data_samples(data_samples, 'mask_camera')
        mask_camera_bits = self._stack_data_samples(data_samples, 'mask_camera_bits')
        mask_camera_names = [sample.metainfo.get('mask_camera_names', None) for sample in data_samples] \
            if data_samples else []
        debug_is_finite('input.img', img)
        debug_is_finite('input.points', points)
        debug_is_finite('input.voxel_semantics', voxel_semantics)
        debug_is_finite('input.mask_camera', mask_camera)
        if img_metas:
            for i, meta in enumerate(img_metas):
                if 'ego2img' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2img', np.asarray(meta['ego2img']))
                if 'ego2occ' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2occ', np.asarray(meta['ego2occ']))
                if 'ego2lidar' in meta:
                    debug_is_finite(f'img_metas[{i}].ego2lidar', np.asarray(meta['ego2lidar']))
        encoder_img_metas = None
        if view_dropout_preprocessed:
            view_dropout_state = dict(preprocessed_view_dropout_state or {})
            if runtime_num_views is None:
                runtime_num_views = self._resolve_runtime_num_views(
                    img=img, img_metas=img_metas, runtime_num_views=None)
            runtime_num_views = int(runtime_num_views)
            view_dropout_state.setdefault('runtime_num_views', runtime_num_views)
            view_dropout_state.setdefault(
                'view_dropout_keep_count',
                int(runtime_num_views))
        else:
            if self.online_depth_enabled:
                depth_point_view_ids = None
            view_dropout_state = self._prepare_train_view_dropout(
                img=img,
                points=points,
                depth_point_view_ids=depth_point_view_ids,
                view_extra=view_extra,
                img_metas=img_metas,
                voxel_semantics=voxel_semantics,
                mask_camera=mask_camera,
                mask_camera_bits=mask_camera_bits,
                mask_camera_names=mask_camera_names)
            img = view_dropout_state['img']
            points = view_dropout_state['points']
            view_extra = view_dropout_state['view_extra']
            img_metas = view_dropout_state['head_img_metas']
            encoder_img_metas = view_dropout_state['encoder_img_metas']
            runtime_num_views = view_dropout_state['runtime_num_views']
            mask_camera = view_dropout_state['mask_camera']
        return self.forward_train(
            img=img,
            points=points,
            view_extra=view_extra,
            online_depth_inputs=online_depth_inputs,
            img_metas=img_metas,
            encoder_img_metas=encoder_img_metas,
            runtime_num_views=runtime_num_views,
            view_dropout_state=view_dropout_state,
            raw_semantics=raw_semantics,
            raw_voxel_origin=raw_voxel_origin,
            raw_voxel_size=raw_voxel_size,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
        )

    def predict(self, inputs, data_samples, rescale=False):
        img = inputs.get('img') if isinstance(inputs, dict) else inputs
        img = self._normalize_img_batch_input(img)
        points = inputs.get('points') if isinstance(inputs, dict) else None
        view_extra = inputs.get('view_extra') if isinstance(inputs, dict) else None
        online_depth_inputs = inputs.get('online_depth') if isinstance(inputs, dict) else None
        raw_semantics = None
        raw_voxel_origin = None
        raw_voxel_size = None
        if getattr(self.pts_bbox_head, '_use_raw_occscannet_gt', lambda: False)():
            raw_semantics = self._stack_data_samples(data_samples, 'raw_semantics')
            raw_voxel_origin = self._stack_data_samples(data_samples, 'voxel_origin')
            raw_voxel_size = self._stack_data_samples(data_samples, 'voxel_size')
        if isinstance(img, torch.Tensor) and img.dim() >= 1:
            view_extra = self._normalize_view_extra(
                view_extra, batch_size=int(img.shape[0]))
        img_metas = self._collect_img_metas(data_samples)
        return self.simple_test(
            img_metas,
            img,
            points,
            rescale=rescale,
            view_extra=view_extra,
            online_depth_inputs=online_depth_inputs,
            raw_semantics=raw_semantics,
            raw_voxel_origin=raw_voxel_origin,
            raw_voxel_size=raw_voxel_size)

    def forward_train(self,
                      points=None,
                      view_extra=None,
                      online_depth_inputs=None,
                      img_metas=None,
                      encoder_img_metas=None,
                      runtime_num_views=None,
                      view_dropout_state=None,
                      raw_semantics=None,
                      raw_voxel_origin=None,
                      raw_voxel_size=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      voxel_semantics=None,
                      mask_camera=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if isinstance(img, torch.Tensor) and img.dim() >= 1:
            view_extra = self._normalize_view_extra(
                view_extra, batch_size=int(img.shape[0]))
        runtime_num_views = self._resolve_runtime_num_views(
            img=img,
            img_metas=encoder_img_metas if encoder_img_metas is not None else img_metas,
            runtime_num_views=runtime_num_views)
        source_head_img_metas = encoder_img_metas if encoder_img_metas is not None else img_metas
        runtime_head_img_metas = self._build_external_encoder_runtime_img_metas(
            img=img,
            img_metas=source_head_img_metas,
            runtime_num_views=runtime_num_views,
        )
        if self.online_depth_enabled:
            # If precomputed-depth points are still present, keep using them for
            # RADIO/LiDAR injection while the online branch runs, then replace
            # points before TPV/head.  If the depth loader is skipped, RADIO sees
            # ``None`` and the online points become the sole TPV/head input.
            img_points = points
            online_depth_stream = None
            online_depth_result = None
            can_parallel_online_depth = (
                self.online_depth_mode == 'parallel'
                and torch.cuda.is_available()
                and isinstance(img, torch.Tensor)
                and img.is_cuda
            )
            if can_parallel_online_depth:
                online_depth_stream = torch.cuda.Stream(device=img.device)
                with torch.cuda.stream(online_depth_stream):
                    online_depth_result = self._apply_online_depth_if_enabled(
                        points, online_depth_inputs, img=img)
            else:
                online_depth_result = self._apply_online_depth_if_enabled(
                    points, online_depth_inputs, img=img)
        else:
            img_points = points
            online_depth_stream = None
            online_depth_result = None

        img_feats = self.extract_img_feat(
            img,
            source_head_img_metas,
            points=img_points,
            view_extra=view_extra,
            runtime_num_views=runtime_num_views) \
            if self._need_img_branch() else None
        head_img_metas = runtime_head_img_metas if runtime_head_img_metas is not None \
            else source_head_img_metas
        if online_depth_stream is not None:
            torch.cuda.current_stream(device=img.device).wait_stream(online_depth_stream)
            points, _ = online_depth_result
        elif self.online_depth_enabled:
            points, _ = online_depth_result
        pts_feats = self._extract_pts_feat_for_head(points)
        tpv_feats = self._extract_tpv_feat_for_head(points)
        if pts_feats is not None:
            pts_feats = self._maybe_drop_lidar_feat(pts_feats)
        debug_is_finite('img_feats', img_feats)
        debug_is_finite('pts_feats', pts_feats)
        debug_is_finite('tpv_feats', tpv_feats)

        # forward occ head
        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats, tpv_feats=tpv_feats,
                                  img_metas=head_img_metas, points=points,
                                  runtime_num_views=runtime_num_views)
        if mask_camera is None:
            mask_camera = torch.ones_like(voxel_semantics)
        losses = self.pts_bbox_head.loss(
            voxel_semantics,
            mask_camera,
            outs,
            raw_semantics=raw_semantics,
            raw_voxel_origin=raw_voxel_origin,
            raw_voxel_size=raw_voxel_size,
            img_metas=img_metas,
        )
        runtime_num_query = outs.get('runtime_num_query', None)
        if runtime_num_query is not None:
            stat_device = voxel_semantics.device if isinstance(voxel_semantics, torch.Tensor) else img.device
            losses['train_runtime_num_query'] = torch.tensor(
                float(runtime_num_query),
                device=stat_device)
        if view_dropout_state is not None and view_dropout_state.get('kept_camera_ids', None) is not None:
            stat_device = voxel_semantics.device if isinstance(voxel_semantics, torch.Tensor) else img.device
            losses['train_view_dropout_keep_count'] = torch.tensor(
                float(view_dropout_state.get('view_dropout_keep_count', 0)),
                device=stat_device)
            losses['train_view_dropout_attempts'] = torch.tensor(
                float(view_dropout_state.get('view_dropout_attempts', 0)),
                device=stat_device)
            fallback = float(view_dropout_state.get('view_dropout_fallback', 0))
            reject_empty_points = float(
                view_dropout_state.get('view_dropout_reject_empty_points', 0))
            reject_empty_gt = float(
                view_dropout_state.get('view_dropout_reject_empty_gt', 0))
            if fallback > 0:
                losses['train_view_dropout_fallback'] = torch.tensor(
                    fallback, device=stat_device)
            if reject_empty_points > 0:
                losses['train_view_dropout_reject_empty_points'] = torch.tensor(
                    reject_empty_points, device=stat_device)
            if reject_empty_gt > 0:
                losses['train_view_dropout_reject_empty_gt'] = torch.tensor(
                    reject_empty_gt, device=stat_device)

        return losses

    @disable_all_fp16_function
    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        return self.simple_test(img_metas[0], img[0], points[0], **kwargs)

    def _get_num_views(self):
        transformer = getattr(self.pts_bbox_head, 'transformer', None)
        if transformer is not None and hasattr(transformer, 'num_views'):
            return int(transformer.num_views)
        decoder = getattr(transformer, 'decoder', None) if transformer is not None else None
        if decoder is not None and hasattr(decoder, 'num_views'):
            return int(decoder.num_views)
        return 6

    def _get_num_frames(self):
        transformer = getattr(self.pts_bbox_head, 'transformer', None)
        if transformer is not None and hasattr(transformer, 'num_frames'):
            return int(transformer.num_frames)
        decoder = getattr(transformer, 'decoder', None) if transformer is not None else None
        if decoder is not None and hasattr(decoder, 'num_frames'):
            return int(decoder.num_frames)
        return 1

    def _get_online_depth_device(self, img=None, points=None):
        if isinstance(img, torch.Tensor):
            return img.device
        if isinstance(points, torch.Tensor):
            return points.device
        if isinstance(points, (list, tuple)) and points:
            first = points[0]
            if hasattr(first, 'tensor'):
                first = first.tensor
            if isinstance(first, torch.Tensor):
                return first.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _apply_online_depth_if_enabled(self, points, online_depth_inputs, img=None):
        if not self.online_depth_enabled:
            return points, None
        if self.online_depth is None:
            raise RuntimeError('online_depth is enabled but module is not initialized')
        device = self._get_online_depth_device(img=img, points=points)
        batch_size = int(img.shape[0]) if isinstance(img, torch.Tensor) and img.dim() >= 1 else None
        online_points, online_view_ids = self.online_depth(
            online_depth_inputs, device=device, dtype=torch.float32, batch_size=batch_size)
        return online_points, online_view_ids

    def _resolve_runtime_num_views(self, img=None, img_metas=None, runtime_num_views=None):
        if runtime_num_views is not None:
            runtime_num_views = int(runtime_num_views)
            if runtime_num_views > 0:
                return runtime_num_views

        total_views = None
        if isinstance(img, torch.Tensor) and img.dim() == 5:
            total_views = int(img.shape[1])
        elif isinstance(img_metas, list) and img_metas and isinstance(img_metas[0], dict):
            filenames = img_metas[0].get('filename', None)
            if isinstance(filenames, list) and filenames:
                total_views = len(filenames)

        num_frames = self._get_num_frames()
        if total_views is not None and num_frames > 0 and total_views % num_frames == 0:
            resolved = total_views // num_frames
            if resolved > 0:
                return int(resolved)

        fallback = self._get_num_views()
        return int(fallback) if fallback > 0 else None

    def _can_use_online_test(self, img_metas, img):
        if get_dist_info()[1] != 1:
            return False
        if not isinstance(img_metas, list) or len(img_metas) != 1:
            return False
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            return False
        num_views = self._get_num_views()
        return num_views > 0 and img.shape[1] % num_views == 0

    def simple_test(self,
                    img_metas,
                    img=None,
                    points=None,
                    rescale=False,
                    view_extra=None,
                    online_depth_inputs=None,
                    runtime_num_views=None,
                    raw_semantics=None,
                    raw_voxel_origin=None,
                    raw_voxel_size=None):
        if self._can_use_online_test(img_metas, img):
            return self.simple_test_online(
                img_metas,
                img,
                points,
                rescale,
                view_extra=view_extra,
                online_depth_inputs=online_depth_inputs,
                runtime_num_views=runtime_num_views,
                raw_semantics=raw_semantics,
                raw_voxel_origin=raw_voxel_origin,
                raw_voxel_size=raw_voxel_size)
        return self.simple_test_offline(
            img_metas,
            img,
            points,
            rescale,
            view_extra=view_extra,
            online_depth_inputs=online_depth_inputs,
            runtime_num_views=runtime_num_views,
            raw_semantics=raw_semantics,
            raw_voxel_origin=raw_voxel_origin,
            raw_voxel_size=raw_voxel_size)

    def _collect_img_metas(self, data_samples):
        if not data_samples:
            return []
        return [sample.metainfo for sample in data_samples]

    def _stack_data_samples(self, data_samples, key):
        if not data_samples:
            return None
        values = [getattr(sample, key) for sample in data_samples if hasattr(sample, key)]
        if not values:
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values, dim=0)
        return torch.tensor(values)

    def _collect_data_sample_list(self, data_samples, key):
        if not data_samples:
            return None
        values = [getattr(sample, key, None) for sample in data_samples]
        if all(value is None for value in values):
            return None
        return values

    def _normalize_view_extra(self, view_extra, batch_size):
        def _is_batched_leaf(key, value):
            if key in {'intrinsics', 'camera_poses', 'camera_pose_quats', 'camera_pose_trans'}:
                return False
            if isinstance(value, (list, tuple)) and len(value) == batch_size:
                return True
            if key == 'is_metric_scale':
                if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                    return True
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
                    return True
            if isinstance(value, torch.Tensor) and value.dim() >= 3 and value.shape[0] == batch_size:
                return True
            if isinstance(value, np.ndarray) and value.ndim >= 3 and value.shape[0] == batch_size:
                return True
            return False

        def _select_batched_leaf(key, value, sample_idx):
            if key in {'intrinsics', 'camera_poses', 'camera_pose_quats', 'camera_pose_trans'}:
                return copy.deepcopy(value)
            if isinstance(value, list):
                return copy.deepcopy(value[sample_idx])
            if isinstance(value, tuple):
                return copy.deepcopy(value[sample_idx])
            if isinstance(value, torch.Tensor):
                return value[sample_idx]
            if isinstance(value, np.ndarray):
                return np.array(value[sample_idx], copy=True)
            return copy.deepcopy(value)

        def _decollate_from_dict(extra_dict):
            if not isinstance(extra_dict, dict):
                return None
            if 'views' not in extra_dict:
                return None
            views = extra_dict.get('views', None)
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                return None
            views = list(views)
            if not isinstance(views[0], dict):
                return None

            has_batched_payload = False
            for view in views:
                if not isinstance(view, dict):
                    continue
                for view_key, value in view.items():
                    if _is_batched_leaf(view_key, value):
                        has_batched_payload = True
                        break
                if has_batched_payload:
                    break
            if not has_batched_payload:
                return None

            decollated = [dict() for _ in range(batch_size)]
            for key, value in extra_dict.items():
                if key == 'views':
                    for sample_idx in range(batch_size):
                        sample_views = []
                        for view in views:
                            if not isinstance(view, dict):
                                sample_views.append(copy.deepcopy(view))
                                continue
                            sample_view = {}
                            for view_key, view_value in view.items():
                                if _is_batched_leaf(view_key, view_value):
                                    sample_view[view_key] = _select_batched_leaf(
                                        view_key, view_value, sample_idx)
                                else:
                                    sample_view[view_key] = copy.deepcopy(view_value)
                            sample_views.append(sample_view)
                        decollated[sample_idx]['views'] = sample_views
                    continue

                if _is_batched_leaf(key, value):
                    for sample_idx in range(batch_size):
                        decollated[sample_idx][key] = _select_batched_leaf(key, value, sample_idx)
                else:
                    for sample_idx in range(batch_size):
                        decollated[sample_idx][key] = copy.deepcopy(value)
            return decollated

        if view_extra is None:
            return None
        if isinstance(view_extra, dict):
            decollated = _decollate_from_dict(view_extra)
            if decollated is not None:
                return decollated
            return [copy.deepcopy(view_extra) for _ in range(batch_size)]
        if isinstance(view_extra, tuple):
            view_extra = list(view_extra)
        if not isinstance(view_extra, list):
            raise TypeError(
                'view_extra must be None, dict, or list of dict, '
                f'but got {type(view_extra)}')
        if len(view_extra) != batch_size:
            raise ValueError(
                f'view_extra batch size mismatch: got {len(view_extra)}, '
                f'expected {batch_size}')
        normalized = []
        for item in view_extra:
            if item is None:
                normalized.append(None)
            elif isinstance(item, dict):
                normalized.append(copy.deepcopy(item))
            else:
                raise TypeError(
                    f'Each view_extra item must be dict/None, got {type(item)}')
        return normalized

    def _slice_view_extra(self, view_extra, img_indices, total_views):
        if view_extra is None:
            return None

        sliced = []
        for sample_extra in view_extra:
            if sample_extra is None:
                sliced.append(None)
                continue
            if not isinstance(sample_extra, dict):
                raise TypeError(
                    f'Each view_extra item must be dict/None, got {type(sample_extra)}')
            sample_out = copy.deepcopy(sample_extra)
            views = sample_out.get('views', None)
            if isinstance(views, (list, tuple)) and len(views) == total_views:
                sample_out['views'] = [views[j] for j in img_indices]
            sliced.append(sample_out)
        return sliced

    def simple_test_offline(self,
                            img_metas,
                            img=None,
                            points=None,
                            rescale=False,
                            view_extra=None,
                            online_depth_inputs=None,
                            runtime_num_views=None,
                            raw_semantics=None,
                            raw_voxel_origin=None,
                            raw_voxel_size=None):
        if isinstance(img, torch.Tensor) and img.dim() >= 1:
            view_extra = self._normalize_view_extra(
                view_extra, batch_size=int(img.shape[0]))
        runtime_num_views = self._resolve_runtime_num_views(
            img=img, img_metas=img_metas, runtime_num_views=runtime_num_views)
        runtime_img_metas = self._build_external_encoder_runtime_img_metas(
            img=img,
            img_metas=img_metas,
            runtime_num_views=runtime_num_views,
        )
        if self.online_depth_enabled:
            points, _ = self._apply_online_depth_if_enabled(
                points, online_depth_inputs, img=img)
        img_feats = self.extract_img_feat(
            img,
            img_metas,
            points=points,
            view_extra=view_extra,
            runtime_num_views=runtime_num_views) \
            if self._need_img_branch() else None
        pts_feats = self._extract_pts_feat_for_head(points)
        tpv_feats = self._extract_tpv_feat_for_head(points)
        head_img_metas = runtime_img_metas if runtime_img_metas is not None else img_metas

        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats, tpv_feats=tpv_feats,
                                  img_metas=head_img_metas, points=points,
                                  runtime_num_views=runtime_num_views)
        if getattr(self.pts_bbox_head, '_use_raw_occscannet_gt', lambda: False)():
            return self.pts_bbox_head.get_occ(
                outs,
                head_img_metas,
                rescale=rescale,
                raw_semantics=raw_semantics,
                raw_voxel_origin=raw_voxel_origin,
                raw_voxel_size=raw_voxel_size)
        return self.pts_bbox_head.get_occ(outs, head_img_metas[0], rescale=rescale)

    def simple_test_online(self,
                           img_metas,
                           img=None,
                           points=None,
                           rescale=False,
                           view_extra=None,
                           online_depth_inputs=None,
                           runtime_num_views=None,
                           raw_semantics=None,
                           raw_voxel_origin=None,
                           raw_voxel_size=None):
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        view_extra = self._normalize_view_extra(view_extra, B)
        runtime_num_views = self._resolve_runtime_num_views(
            img=img, img_metas=img_metas, runtime_num_views=runtime_num_views)
        if self.online_depth_enabled:
            points, _ = self._apply_online_depth_if_enabled(
                points, online_depth_inputs, img=img)
        num_views = self._get_num_views()
        img = img.reshape(B, N // num_views, num_views, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // num_views
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        uses_external_encoder = self.use_external_img_encoder or self.use_img_feature_fusion
        allow_cache = not (uses_external_encoder and not self.external_img_cache)
        if view_extra is not None:
            # Optional extra modalities can change feature semantics even for same filename.
            allow_cache = False
        for i in range(num_frames):
            img_indices = list(np.arange(i * num_views, (i + 1) * num_views))
            extra_curr = self._slice_view_extra(
                view_extra,
                img_indices=img_indices,
                total_views=num_views * num_frames)

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                item = img_metas[0][k]
                if isinstance(item, list) and (len(item) == num_views * num_frames):
                    img_metas_curr[0][k] = [item[j] for j in img_indices]
                else:
                    img_metas_curr[0][k] = item

            if allow_cache and img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_img_feat(
                    img[:, i],
                    img_metas_curr,
                    points=points,
                    view_extra=extra_curr,
                    runtime_num_views=runtime_num_views)
                if allow_cache:
                    self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                    self.queue.put(img_filenames[img_indices[0]])
                    while self.queue.qsize() >= 16:  # avoid OOM
                        pop_key = self.queue.get()
                        self.memory.pop(pop_key)

            runtime_img_metas_curr = self._build_external_encoder_runtime_img_metas(
                img=img[:, i],
                img_metas=img_metas_curr,
                runtime_num_views=runtime_num_views,
            )

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(
                runtime_img_metas_curr if runtime_img_metas_curr is not None else img_metas_curr
            )

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # extract points features
        pts_feats = self._extract_pts_feat_for_head(points)
        tpv_feats = self._extract_tpv_feat_for_head(points)
        if pts_feats is not None:
                pts_feats = self._maybe_drop_lidar_feat(pts_feats)

        # run occupancy predictor
        outs = self.pts_bbox_head(mlvl_feats=img_feats, pts_feats=pts_feats, tpv_feats=tpv_feats,
                                  img_metas=img_metas, points=points,
                                  runtime_num_views=runtime_num_views)
        if getattr(self.pts_bbox_head, '_use_raw_occscannet_gt', lambda: False)():
            return self.pts_bbox_head.get_occ(
                outs,
                img_metas,
                rescale=rescale,
                raw_semantics=raw_semantics,
                raw_voxel_origin=raw_voxel_origin,
                raw_voxel_size=raw_voxel_size)
        return self.pts_bbox_head.get_occ(outs, img_metas[0], rescale=rescale)
