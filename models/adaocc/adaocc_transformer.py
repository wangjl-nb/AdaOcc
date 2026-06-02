import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmengine.model import BaseModule, bias_init_with_prob
from mmcv.cnn import Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops import knn
from mmdet3d.registry import MODELS
from .adaocc_sampling import sampling_4d, sampling_pts_feats, sampling_tpv_feats
from ..bbox.utils import decode_bbox, decode_points, encode_points
from ..utils import inverse_sigmoid, DUMP, debug_is_finite
from ..checkpoint import checkpoint as cp
from ..csrc.wrapper import MSMV_CUDA

'''
    adding feature:
        enable PT feature sampling
        enable PT feature and image feature fusion
        the fusion way:cat+proj
        lidar feat: group mixing
        
'''

@MODELS.register_module()
class AdaOccTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 ffn_feedforward_channels=512,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 feature_dims=0,
                 scales=[1.0],
                 score_mode='semantic',
                 occ_out_channels=1,
                 use_pts_sampling=True,
                 use_tpv_sampling=False,
                 tpv_fusion_mode='query_attn',
                 split_decoder=None,
                 query_allocator=None,
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines
        self.feature_dims = int(feature_dims)
        self.score_mode = str(score_mode).lower()
        self.occ_out_channels = int(occ_out_channels)
        if self.score_mode not in ('semantic', 'binary_occ'):
            raise ValueError(
                f'Unsupported score_mode={score_mode}, expected "semantic" or "binary_occ"')
        if self.score_mode == 'binary_occ' and self.occ_out_channels <= 0:
            raise ValueError('occ_out_channels must be positive when score_mode="binary_occ"')

        self.decoder = AdaOccTransformerDecoder(
            embed_dims, ffn_feedforward_channels, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, self.feature_dims, scales,
            score_mode=self.score_mode,
            occ_out_channels=self.occ_out_channels,
            use_pts_sampling=use_pts_sampling,
            use_tpv_sampling=use_tpv_sampling,
            tpv_fusion_mode=tpv_fusion_mode,
            split_decoder=split_decoder,
            query_allocator=query_allocator, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, img_metas,
                tpv_feats=None, runtime_num_views=None):
        score_preds, refine_pts, feat_scores = self.decoder(
            query_points, query_feat, mlvl_feats, pts_feats, img_metas,
            tpv_feats=tpv_feats, runtime_num_views=runtime_num_views)

        score_preds = [torch.nan_to_num(score) for score in score_preds]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]
        if feat_scores is not None:
            feat_scores = [torch.nan_to_num(feat) for feat in feat_scores]

        return score_preds, refine_pts, feat_scores


class AdaOccTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 ffn_feedforward_channels=512,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 feature_dims=0,
                 scales=[1.0],
                 score_mode='semantic',
                 occ_out_channels=1,
                 use_pts_sampling=True,
                 use_tpv_sampling=False,
                 tpv_fusion_mode='query_attn',
                 split_decoder=None,
                 query_allocator=None,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.feature_dims = int(feature_dims)
        self.ffn_feedforward_channels = int(ffn_feedforward_channels)
        self.score_mode = str(score_mode).lower()
        self.occ_out_channels = int(occ_out_channels)
        self.use_pts_sampling = bool(use_pts_sampling)
        self.use_tpv_sampling = bool(use_tpv_sampling)
        if self.use_pts_sampling and self.use_tpv_sampling:
            raise ValueError('use_pts_sampling and use_tpv_sampling cannot both be True')
        self.query_allocator = query_allocator or {}
        self.split_decoder = copy.deepcopy(split_decoder) if split_decoder else {}

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_split_cfg = self._resolve_layer_split_cfg(i, num_layers)
            self.decoder_layers.append(
                AdaOccTransformerDecoderLayer(
                    embed_dims, self.ffn_feedforward_channels, num_frames, num_views, num_points, num_levels, num_classes, 
                    num_groups, num_refines[i], last_refines[i], layer_idx=i,
                    feature_dims=self.feature_dims,
                    score_mode=self.score_mode,
                    occ_out_channels=self.occ_out_channels,
                    scale=scales[i], pc_range=pc_range,
                    use_pts_sampling=self.use_pts_sampling,
                    use_tpv_sampling=self.use_tpv_sampling,
                    tpv_fusion_mode=tpv_fusion_mode,
                    split_decoder=layer_split_cfg)
            )

    def _resolve_layer_split_cfg(self, layer_idx, num_layers):
        split_cfg = self.split_decoder or {}
        if not split_cfg.get('enabled', False):
            return None

        enabled = False
        enabled_layers = split_cfg.get('enabled_layers', None)
        if enabled_layers is not None:
            layer_ids = set()
            for idx in enabled_layers:
                idx = int(idx)
                if idx < 0:
                    idx = num_layers + idx
                layer_ids.add(idx)
            enabled = layer_idx in layer_ids
        elif 'start_layer' in split_cfg:
            start_layer = int(split_cfg.get('start_layer', 0))
            if start_layer < 0:
                start_layer = num_layers + start_layer
            enabled = layer_idx >= start_layer
        else:
            last_n_layers = int(split_cfg.get('last_n_layers', 1))
            last_n_layers = max(0, min(num_layers, last_n_layers))
            enabled = layer_idx >= (num_layers - last_n_layers)

        if not enabled:
            return None

        layer_cfg = copy.deepcopy(split_cfg)
        layer_cfg['enabled'] = True
        layer_cfg.pop('enabled_layers', None)
        layer_cfg.pop('start_layer', None)
        layer_cfg.pop('last_n_layers', None)
        return layer_cfg

    @torch.no_grad()
    def init_weights(self):
        for layer in self.decoder_layers:
            if hasattr(layer, 'init_weights'):
                layer.init_weights()

    def _apply_query_allocator(self, query_points, query_feat, score_pred, layer_idx):
        allocator_cfg = self.query_allocator or {}
        if not allocator_cfg.get('enabled', False):
            return query_points, query_feat

        switch_layer = int(allocator_cfg.get('switch_layer', 2))
        if layer_idx != switch_layer - 1:
            return query_points, query_feat

        B, Q, P, Cxyz = query_points.shape
        if Q <= 1:
            return query_points, query_feat

        score_prob = score_pred.sigmoid()
        if self.score_mode == 'binary_occ':
            occ_prob = score_prob.squeeze(-1).mean(dim=-1)
            nonempty_score = occ_prob
            uncertainty_score = 1.0 - torch.abs(2.0 * occ_prob - 1.0)
        else:
            nonempty_score = score_prob.max(dim=-1).values.mean(dim=-1)  # [B, Q]

            num_classes = max(score_prob.shape[-1], 1)
            entropy = -(score_prob.clamp(min=1e-6, max=1 - 1e-6) *
                        torch.log(score_prob.clamp(min=1e-6, max=1 - 1e-6))).sum(dim=-1)
            entropy = entropy / max(np.log(float(num_classes)), 1.0)
            uncertainty_score = entropy.mean(dim=-1)  # [B, Q]

        score_weights = allocator_cfg.get('score_weights', {})
        nonempty_w = float(score_weights.get('nonempty', 0.5))
        uncertainty_w = float(score_weights.get('uncertainty', 0.5))

        context_ratio = float(allocator_cfg.get('context_ratio', 0.6))
        context_ratio = min(max(context_ratio, 0.0), 1.0)
        num_context = int(round(Q * context_ratio))
        num_context = max(0, min(Q, num_context))
        num_detail = Q - num_context

        if self.score_mode == 'binary_occ':
            detail_score = nonempty_w * nonempty_score + uncertainty_w * uncertainty_score
        else:
            detail_score = nonempty_w * (1.0 - nonempty_score) + uncertainty_w * uncertainty_score
        context_rank = torch.argsort(nonempty_score, dim=-1, descending=True)
        detail_rank = torch.argsort(detail_score, dim=-1, descending=True)

        if num_context > 0:
            context_idx = context_rank[:, :num_context]
        else:
            context_idx = context_rank[:, :0]

        detail_indices = []
        for batch_idx in range(B):
            detail_idx = detail_rank[batch_idx]
            if num_context > 0:
                used = torch.zeros(Q, dtype=torch.bool, device=detail_idx.device)
                used[context_idx[batch_idx]] = True
                detail_idx = detail_idx[~used[detail_idx]]
            if detail_idx.numel() < num_detail:
                short = num_detail - detail_idx.numel()
                detail_idx = torch.cat([detail_idx, detail_rank[batch_idx, :short]], dim=0)
            detail_indices.append(detail_idx[:num_detail])

        if num_detail > 0:
            detail_idx = torch.stack(detail_indices, dim=0)
            gather_idx = torch.cat([context_idx, detail_idx], dim=1)
        else:
            gather_idx = context_idx

        gather_points_idx = gather_idx[..., None, None].expand(B, Q, P, Cxyz)
        gather_feat_idx = gather_idx[..., None].expand(B, Q, query_feat.shape[-1])
        new_query_points = torch.gather(query_points, dim=1, index=gather_points_idx)
        new_query_feat = torch.gather(query_feat, dim=1, index=gather_feat_idx)

        detail_jitter_std = float(allocator_cfg.get('detail_jitter_std', 0.0))
        if self.training and detail_jitter_std > 0 and num_detail > 0:
            detail_slice = slice(num_context, Q)
            detail_points = decode_points(new_query_points[:, detail_slice], self.pc_range)
            detail_points = detail_points + torch.randn_like(detail_points) * detail_jitter_std

            pc_range = detail_points.new_tensor(self.pc_range)
            lower = pc_range[:3]
            upper = pc_range[3:]
            detail_points = detail_points.clamp(min=lower, max=upper)

            new_query_points = new_query_points.clone()
            new_query_points[:, detail_slice] = encode_points(detail_points, self.pc_range)

        return new_query_points, new_query_feat

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, img_metas,
                tpv_feats=None, runtime_num_views=None):
        score_preds, refine_pts = [], []
        feat_scores = [] if self.feature_dims > 0 else None
        effective_num_views = self.num_views if runtime_num_views is None else int(runtime_num_views)
        if effective_num_views <= 0:
            raise ValueError(f'runtime_num_views must be positive, got {effective_num_views}')

        ego2img = np.asarray([m['ego2img'] for m in img_metas]).astype(np.float32)
        ego2img = query_feat.new_tensor(ego2img) # [B, N, 4, 4]
        ego2occ = np.asarray([m['ego2occ'] for m in img_metas]).astype(np.float32)
        ego2occ = query_feat.new_tensor(ego2occ)
        occ2ego = torch.inverse(ego2occ) # [B, 4, 4]
        occ2img = ego2img @ occ2ego[:, None].expand_as(ego2img)

        occ2lidar = None
        if self.use_pts_sampling:
            ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
            ego2lidar = query_feat.new_tensor(ego2lidar) # [B, 4, 4]
            occ2lidar = ego2lidar @ occ2ego
            debug_is_finite('ego2lidar', ego2lidar)
        debug_is_finite('ego2img', ego2img)
        debug_is_finite('ego2occ', ego2occ)
        debug_is_finite('occ2img', occ2img)
        debug_is_finite('occ2lidar', occ2lidar)

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = effective_num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        if self.use_pts_sampling:
            if pts_feats is None:
                raise ValueError('pts_feats must be provided when use_pts_sampling=True')
            # group pts features in advance for sampling
            B, GC, H, W = pts_feats.shape
            G, C = self.num_groups, GC // self.num_groups
            if GC % G != 0:
                raise ValueError(
                    f'pts_feats channel={GC} is not divisible by num_groups={G}')
            pts_feats = pts_feats.reshape(B, G, C, H, W).reshape(B * G, C, H, W)
        elif self.use_tpv_sampling:
            if tpv_feats is None:
                raise ValueError('tpv_feats must be provided when use_tpv_sampling=True')
            if not isinstance(tpv_feats, dict):
                raise TypeError(f'tpv_feats must be dict, got {type(tpv_feats)}')
            grouped_tpv = {}
            expected_batch_size = None
            for plane_key in ['xy', 'xz', 'yz']:
                if plane_key not in tpv_feats:
                    raise KeyError(f'tpv_feats missing key "{plane_key}"')
                feat = tpv_feats[plane_key]
                if not isinstance(feat, torch.Tensor) or feat.dim() != 4:
                    raise ValueError(
                        f'tpv_feats["{plane_key}"] must be Tensor[B, GC, H, W], '
                        f'got {type(feat)} shape={getattr(feat, "shape", None)}')
                B, GC, H, W = feat.shape
                if expected_batch_size is None:
                    expected_batch_size = B
                elif B != expected_batch_size:
                    raise ValueError(
                        f'tpv_feats batch mismatch: expected {expected_batch_size}, got {B} for "{plane_key}"')
                G, C = self.num_groups, GC // self.num_groups
                if GC % G != 0:
                    raise ValueError(
                        f'tpv_feats["{plane_key}"] channel={GC} is not divisible by num_groups={G}')
                grouped_tpv[plane_key] = feat.reshape(B, G, C, H, W).reshape(B * G, C, H, W)
            tpv_feats = grouped_tpv

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            query_feat, score_pred, query_points, feat_score = decoder_layer(
                query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar,
                img_metas, tpv_feats=tpv_feats, runtime_num_views=effective_num_views)

            debug_is_finite(f'decoder_layer[{i}].score_pred', score_pred)
            debug_is_finite(f'decoder_layer[{i}].query_points', query_points)
            score_preds.append(score_pred)
            refine_pts.append(query_points)
            if feat_scores is not None:
                feat_scores.append(feat_score)
            query_points, query_feat = self._apply_query_allocator(
                query_points,
                query_feat,
                score_pred,
                layer_idx=i)

        return score_preds, refine_pts, feat_scores


class AdaOccTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 ffn_feedforward_channels=512,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 feature_dims=0,
                 score_mode='semantic',
                 occ_out_channels=1,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 use_pts_sampling=True,
                 use_tpv_sampling=False,
                 tpv_fusion_mode='query_attn',
                 split_decoder=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.ffn_feedforward_channels = int(ffn_feedforward_channels)
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.feature_dims = int(feature_dims)
        self.score_mode = str(score_mode).lower()
        self.occ_out_channels = int(occ_out_channels)
        self.layer_idx = layer_idx
        self.scale = scale
        self.use_pts_sampling = bool(use_pts_sampling)
        self.use_tpv_sampling = bool(use_tpv_sampling)
        self.split_decoder = copy.deepcopy(split_decoder) if split_decoder else {}
        self.split_enabled = bool(self.split_decoder.get('enabled', False))
        if self.score_mode not in ('semantic', 'binary_occ'):
            raise ValueError(
                f'Unsupported score_mode={score_mode}, expected "semantic" or "binary_occ"')
        self.score_out_channels = num_classes if self.score_mode == 'semantic' else self.occ_out_channels

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = AdaOccSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = AdaOccSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                     num_groups=num_groups, num_points=num_points, 
                                     num_levels=num_levels, pc_range=pc_range,
                                     use_pts_sampling=self.use_pts_sampling,
                                     use_tpv_sampling=self.use_tpv_sampling,
                                     tpv_fusion_mode=tpv_fusion_mode)
        
        mixing_points = num_points * (
            num_frames + (1 if (self.use_pts_sampling or self.use_tpv_sampling) else 0))
        self.img_pts_mixing=AdaptiveMixing(
            in_dim=embed_dims, in_points=mixing_points, n_groups=num_groups)

        self.ffn = FFN(
            embed_dims,
            feedforward_channels=self.ffn_feedforward_channels,
            ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        if self.split_enabled:
            self._init_split_decoder(num_cls_fcs=num_cls_fcs, num_reg_fcs=num_reg_fcs)
        else:
            self.score_branch = self._build_branch(
                input_dim=self.embed_dims,
                hidden_dim=self.embed_dims,
                out_dim=self.score_out_channels * self.num_refines,
                num_fcs=num_cls_fcs,
                use_norm=True)
            self.reg_branch = self._build_branch(
                input_dim=self.embed_dims,
                hidden_dim=self.embed_dims,
                out_dim=3 * self.num_refines,
                num_fcs=num_reg_fcs,
                use_norm=False)
            if self.feature_dims > 0:
                self.feat_branch = self._build_branch(
                    input_dim=self.embed_dims,
                    hidden_dim=self.embed_dims,
                    out_dim=self.feature_dims * self.num_refines,
                    num_fcs=num_cls_fcs,
                    use_norm=True)
            else:
                self.feat_branch = None

    @staticmethod
    def _build_branch(input_dim, hidden_dim, out_dim, num_fcs=2, use_norm=True):
        layers = []
        curr_dim = input_dim
        for _ in range(num_fcs):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)

    def _init_split_decoder(self, num_cls_fcs=2, num_reg_fcs=2):
        self.split_num_children = int(self.split_decoder.get('num_children', 8))
        if self.split_num_children <= 0:
            raise ValueError(f'split_decoder.num_children must be positive, got {self.split_num_children}')
        if self.num_refines % self.split_num_children != 0:
            raise ValueError(
                f'num_refines={self.num_refines} must be divisible by split num_children={self.split_num_children}')
        self.split_child_points = self.num_refines // self.split_num_children
        self.split_child_dim = int(self.split_decoder.get('child_dim', self.embed_dims))
        if self.split_child_dim <= 0:
            raise ValueError(f'split_decoder.child_dim must be positive, got {self.split_child_dim}')
        self.split_child_radius_scale = float(self.split_decoder.get('child_radius_scale', 1.0))
        self.split_leaf_radius_scale = float(self.split_decoder.get('leaf_radius_scale', 0.5))
        self.split_min_extent_ratio = float(self.split_decoder.get('min_extent_ratio', 0.005))
        self.split_fuse_child_to_query = bool(self.split_decoder.get('fuse_child_to_query', True))
        self.split_context_mode = str(self.split_decoder.get('context', 'auto')).lower()
        if self.split_context_mode == 'auto':
            if self.use_tpv_sampling:
                self.split_context_mode = 'tpv'
            elif self.use_pts_sampling:
                self.split_context_mode = 'pts'
            else:
                self.split_context_mode = 'none'
        if self.split_context_mode not in ('none', 'tpv', 'pts'):
            raise ValueError(
                f'Unsupported split_decoder.context={self.split_context_mode}, expected "auto|none|tpv|pts"')

        self.score_branch = None
        self.reg_branch = None
        self.feat_branch = None

        self.child_seed_branch = self._build_branch(
            input_dim=self.embed_dims,
            hidden_dim=self.embed_dims,
            out_dim=self.split_num_children * (3 + self.split_child_dim),
            num_fcs=max(num_reg_fcs, 1),
            use_norm=True)
        self.child_pos_encoder = nn.Sequential(
            nn.Linear(3, self.split_child_dim),
            nn.LayerNorm(self.split_child_dim),
            nn.ReLU(inplace=True),
        )
        self.child_refine = self._build_branch(
            input_dim=self.split_child_dim,
            hidden_dim=self.split_child_dim,
            out_dim=self.split_child_dim,
            num_fcs=1,
            use_norm=True)
        if self.split_fuse_child_to_query:
            self.child_query_fuse = nn.Sequential(
                nn.Linear(self.split_child_dim, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )
        else:
            self.child_query_fuse = None

        self.group_feature_dims = None
        self.child_ctx_proj = None
        if self.split_context_mode != 'none':
            if self.embed_dims % self.num_groups != 0:
                raise ValueError(
                    f'embed_dims={self.embed_dims} must be divisible by num_groups={self.num_groups} '
                    'when split_decoder local context is enabled')
            self.group_feature_dims = self.embed_dims // self.num_groups
            self.child_ctx_proj = nn.Sequential(
                nn.Linear(self.group_feature_dims, self.split_child_dim),
                nn.LayerNorm(self.split_child_dim),
                nn.ReLU(inplace=True),
            )

        self.split_score_branch = self._build_branch(
            input_dim=self.split_child_dim,
            hidden_dim=self.split_child_dim,
            out_dim=self.score_out_channels * self.split_child_points,
            num_fcs=num_cls_fcs,
            use_norm=True)
        self.split_reg_branch = self._build_branch(
            input_dim=self.split_child_dim,
            hidden_dim=self.split_child_dim,
            out_dim=3 * self.split_child_points,
            num_fcs=num_reg_fcs,
            use_norm=False)
        if self.feature_dims > 0:
            self.split_feat_branch = self._build_branch(
                input_dim=self.split_child_dim,
                hidden_dim=self.split_child_dim,
                out_dim=self.feature_dims * self.split_child_points,
                num_fcs=num_cls_fcs,
                use_norm=True)
        else:
            self.split_feat_branch = None

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.img_pts_mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        if self.split_enabled:
            nn.init.constant_(self.split_score_branch[-1].bias, bias_init)
        else:
            nn.init.constant_(self.score_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def _decode_with_split(self, query_points, query_feat, pts_feats, occ2lidar, tpv_feats):
        B, Q = query_points.shape[:2]
        proposal_points = decode_points(query_points, self.pc_range)
        proposal_center = proposal_points.mean(dim=2)  # [B, Q, 3]
        proposal_spread = proposal_points.std(dim=2, unbiased=False)
        pc_range = proposal_points.new_tensor(self.pc_range)
        min_extent = (pc_range[3:] - pc_range[:3]) * self.split_min_extent_ratio
        proposal_spread = proposal_spread.clamp(min=min_extent.view(1, 1, 3))

        child_seed = self.child_seed_branch(query_feat)
        child_seed = child_seed.reshape(B, Q, self.split_num_children, 3 + self.split_child_dim)
        child_offset = torch.tanh(child_seed[..., :3])
        child_offset = child_offset * proposal_spread[:, :, None, :] * self.split_child_radius_scale
        child_center = proposal_center[:, :, None, :] + child_offset

        child_feat = child_seed[..., 3:]
        child_feat = child_feat + self.child_pos_encoder(child_offset)

        if self.child_ctx_proj is not None:
            sample_points = child_center[:, :, None, :, :].expand(
                B, Q, self.num_groups, self.split_num_children, 3)
            if self.split_context_mode == 'tpv':
                if tpv_feats is None:
                    raise ValueError('tpv_feats must be provided when split_decoder.context="tpv"')
                child_ctx = sampling_tpv_feats(sample_points, tpv_feats, self.pc_range)
            else:
                if pts_feats is None or occ2lidar is None:
                    raise ValueError('pts_feats and occ2lidar must be provided when split_decoder.context="pts"')
                child_ctx = sampling_pts_feats(sample_points, pts_feats, occ2lidar, self.pc_range)
            child_ctx = child_ctx.mean(dim=2)
            child_feat = child_feat + self.child_ctx_proj(child_ctx)

        child_feat = child_feat + self.child_refine(child_feat)
        if self.child_query_fuse is not None:
            query_feat = query_feat + self.child_query_fuse(child_feat.mean(dim=2))

        leaf_offset = self.scale * self.split_reg_branch(child_feat)
        leaf_offset = leaf_offset.reshape(B, Q, self.split_num_children, self.split_child_points, 3)
        leaf_offset = torch.tanh(leaf_offset)
        leaf_offset = leaf_offset * proposal_spread[:, :, None, None, :] * self.split_leaf_radius_scale

        score_pred = self.split_score_branch(child_feat)
        score_pred = score_pred.reshape(
            B, Q, self.split_num_children, self.split_child_points, self.score_out_channels)

        feat_score = None
        if self.split_feat_branch is not None:
            feat_score = self.split_feat_branch(child_feat)
            feat_score = feat_score.reshape(
                B, Q, self.split_num_children, self.split_child_points, self.feature_dims)

        refine_pt = child_center[:, :, :, None, :] + leaf_offset
        refine_pt = refine_pt.reshape(B, Q, self.num_refines, 3)
        refine_pt = encode_points(refine_pt, self.pc_range)

        score_pred = score_pred.reshape(B, Q, self.num_refines, self.score_out_channels)
        if feat_score is not None:
            feat_score = feat_score.reshape(B, Q, self.num_refines, self.feature_dims)

        return query_feat, score_pred, refine_pt, feat_score

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar,
                img_metas, tpv_feats=None, runtime_num_views=None):
        """
        query_points: [B, Q, 3] [x, y, z]
        pts_feats:[B,C,dy,dx]
        """
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_img_feat, sampled_pts_feat = self.sampling(
            query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar,
            img_metas, tpv_feats=tpv_feats, runtime_num_views=runtime_num_views)
        if sampled_pts_feat is None:
            sampled_feat = sampled_img_feat
        else:
            sampled_feat = torch.cat([sampled_img_feat, sampled_pts_feat], dim=-2) # B,Q,G,(T+1)P,C1
        query_feat = self.norm1(self.img_pts_mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        if self.split_enabled:
            query_feat, score_pred, refine_pt, feat_score = self._decode_with_split(
                query_points, query_feat, pts_feats, occ2lidar, tpv_feats)
            # Keep the split-layer query stream graph-connected to the decoder outputs
            # so DDP does not treat late self-attention params as unused.
            graph_anchor = query_feat.sum(dim=-1, keepdim=True)[..., None] * 0.0
            score_pred = score_pred + graph_anchor
        else:
            B, Q = query_points.shape[:2]
            score_pred = self.score_branch(query_feat)
            reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]
            score_pred = score_pred.reshape(B, Q, self.num_refines, self.score_out_channels)
            feat_score = None
            if self.feat_branch is not None:
                feat_score = self.feat_branch(query_feat)
                feat_score = feat_score.reshape(B, Q, self.num_refines, self.feature_dims)
            refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, score_pred, refine_pt, feat_score


class AdaOccSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.max_tau = 10.0
        self.min_attn_bias = -60.0

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points).float()
        tau = self.gen_tau(query_feat).float()  # [B, Q, 8]
        tau = torch.nan_to_num(
            tau,
            nan=0.0,
            posinf=self.max_tau,
            neginf=0.0,
        ).clamp(min=0.0, max=self.max_tau)

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        # The distance bias is an additive logit mask.  It should only dampen
        # far-away queries; letting a trained negative/oversized tau create a
        # large positive fp16 mask can overflow inside softmax and make an
        # entire sample's decoder stream NaN.  Saturating at exp(-60) preserves
        # the intended "effectively ignored" behavior without introducing +/-inf.
        attn_mask = torch.nan_to_num(
            attn_mask,
            nan=0.0,
            posinf=0.0,
            neginf=self.min_attn_bias,
        ).clamp(min=self.min_attn_bias, max=0.0)

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        debug_is_finite('self_attn.attn_mask', attn_mask)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class AdaOccSampling(BaseModule):
    """
        Adaptive Spatio-temporal Sampling
        
        adding feat:
            return sampling pts 
    
    """
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 use_pts_sampling=True,
                 use_tpv_sampling=False,
                 tpv_fusion_mode='query_attn',
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.use_pts_sampling = bool(use_pts_sampling)
        self.use_tpv_sampling = bool(use_tpv_sampling)
        if self.use_pts_sampling and self.use_tpv_sampling:
            raise ValueError('use_pts_sampling and use_tpv_sampling cannot both be True')
        self.tpv_fusion_mode = tpv_fusion_mode

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)
        self.tpv_plane_weights = None
        if self.use_tpv_sampling:
            if self.tpv_fusion_mode != 'query_attn':
                raise ValueError(
                    f'Unsupported tpv_fusion_mode={self.tpv_fusion_mode}, expected "query_attn"')
            self.tpv_plane_weights = nn.Linear(embed_dims, num_groups * num_points * 3)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar,
                      img_metas, tpv_feats=None, runtime_num_views=None):
        '''
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)

        img_sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        img_sampling_points = img_sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)
        pts_sampling_points = sampling_points.clone()

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        effective_num_views = self.num_views if runtime_num_views is None else int(runtime_num_views)
        if effective_num_views <= 0:
            raise ValueError(f'runtime_num_views must be positive, got {effective_num_views}')

        sampled_img_feats = sampling_4d(
            img_sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h, image_w,
            effective_num_views
        )  # [B, Q, G, FP, C]

        sampled_pts_feats = None
        if self.use_pts_sampling:
            if pts_feats is None:
                raise ValueError('pts_feats is None while use_pts_sampling=True')
            if occ2lidar is None:
                raise ValueError('occ2lidar is None while use_pts_sampling=True')
            sampled_pts_feats = sampling_pts_feats(
                pts_sampling_points,
                pts_feats,
                occ2lidar,
                self.pc_range
            )  # [B, Q, G, P, C]
        elif self.use_tpv_sampling:
            if tpv_feats is None:
                raise ValueError('tpv_feats is None while use_tpv_sampling=True')
            plane_weights = None
            if self.tpv_fusion_mode == 'query_attn':
                plane_weights = self.tpv_plane_weights(query_feat).view(
                    B, Q, self.num_groups, self.num_points, 3)
            sampled_pts_feats = sampling_tpv_feats(
                pts_sampling_points,
                tpv_feats,
                self.pc_range,
                plane_weights=plane_weights,
            )  # [B, Q, G, P, C]

        return sampled_img_feats, sampled_pts_feats

    def forward(self, query_points, query_feat, mlvl_feats, pts_feats, occ2img, occ2lidar,
                img_metas, tpv_feats=None, runtime_num_views=None):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats, pts_feats,
                      occ2img, occ2lidar, img_metas, tpv_feats, runtime_num_views,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, mlvl_feats, pts_feats,
                                      occ2img, occ2lidar, img_metas, tpv_feats=tpv_feats,
                                      runtime_num_views=runtime_num_views)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)
        self.eval_out_proj_chunk = 256

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def _project_back_to_query(self, out):
        out = out.contiguous()

        # NOTE:
        # On the current H20 / torch2.2 / cuda12.1 stack, eval can hit a CUDA
        # SIGFPE on the single large out_proj GEMM here (observed with
        # shape [1, 960, 2048] -> [1, 960, 256]). Training does not trigger the
        # same failing kernel because it runs with a different batch/graph path.
        # Chunking the query dimension keeps the math identical while avoiding
        # the bad eval-time kernel selection.
        if (not self.training
                and out.is_cuda
                and out.shape[0] == 1
                and out.shape[1] > self.eval_out_proj_chunk):
            chunks = []
            for start in range(0, out.shape[1], self.eval_out_proj_chunk):
                chunk = out[:, start:start + self.eval_out_proj_chunk].contiguous()
                chunks.append(F.linear(chunk, self.out_proj.weight, self.out_proj.bias))
            return torch.cat(chunks, dim=1)

        return self.out_proj(out)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self._project_back_to_query(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
