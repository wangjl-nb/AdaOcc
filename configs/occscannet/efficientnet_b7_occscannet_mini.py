"""AdaOcc experimental config: OccScanNet-mini + frozen EfficientNet-B7 image backbone.

This config is a direct copy of the RADIO mini baseline with only the image
backbone/preprocessing path changed for the EfficientNet experiment.  Keep the
RADIO main config unchanged for baseline reproducibility.
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

default_scope = "mmdet3d"
custom_imports = dict(
    imports=["models", "loaders", "models.backbones.timm_feature_backbone"],
    allow_failed_imports=False,
)

_config_dir = _Path("{{ fileDirname }}").resolve()
_repo_root = _Path(_os.getenv("ADAOCC_REPO_ROOT", _config_dir.parents[1])).expanduser().resolve()
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))


def _bool_env(name, default=True):
    value = _os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


# -------------------- Fixed repository-relative runtime paths --------------------
# Public reproduction uses a fixed layout so users only need to arrange files
# under the tree documented in README.md.  Path overrides are intentionally not
# part of the normal open-source workflow.
dataset_root = str(_repo_root / "data" / "OccScanNet")
occ_root = str(_Path(dataset_root) / "gts_camvisbits")
train_ann_file = str(_Path(dataset_root) / "train_occscannet_mini.pkl")
val_ann_file = str(_Path(dataset_root) / "val_occscannet_mini.pkl")
test_ann_file = str(_Path(dataset_root) / "test_occscannet_mini.pkl")
load_from = str(_repo_root / "pretrain" / "fusion_pretrain_model.pth")
radio_model_id = str(_repo_root / "pretrain" / "radio" / "C-RADIOv3-B")
radio_local_files_only = True
efficientnet_checkpoint_path = str(_repo_root / "pretrain" / "timm" / "tf_efficientnet_b7_ns-1dbc32de.pth")
output_root = str(_repo_root / "outputs")
depth_anything_model_path = str(
    _repo_root / "pretrain" / "depth_anything" / "finetune_scannet_depthanythingv2.pth"
)

# -------------------- Dataset --------------------
dataset_type = "OccScanNetDataset"
use_raw_occscannet_gt = True
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)
object_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
occ_names = [
    "ceiling", "floor", "wall", "window", "chair", "bed", "sofa", "table",
    "tvs", "furniture", "objects",
]
class_names = occ_names + ["free"]
empty_label = len(occ_names)
rare_classes = []
cls_weights = [4, 1, 1, 3, 1, 2, 2, 1, 5, 1, 2]

# -------------------- Geometry --------------------
cam_types = ["CAM_FRONT"]
num_views = len(cam_types)
point_cloud_range = [-3.20, -4.80, -5.60, 7.20, 4.80, 5.60]
pc_voxel_size = [0.08, 0.08, 0.08]
voxel_size = [0.08, 0.08, 0.08]
scan_raw_h = 968
scan_raw_w = 1296
scan_final_dim = (720, 960)
scan_resize = scan_final_dim[0] / float(scan_raw_h)
sparse_shape = [
    int(round((point_cloud_range[5] - point_cloud_range[2]) / pc_voxel_size[2])) + 1,
    int(round((point_cloud_range[4] - point_cloud_range[1]) / pc_voxel_size[1])),
    int(round((point_cloud_range[3] - point_cloud_range[0]) / pc_voxel_size[0])),
]
dataset_cfg = dict(
    cam_types=cam_types,
    num_views=num_views,
    occ_io=dict(
        path_template="{token}/labels.npz",
        semantics_key="semantics",
        mask_camera_key="mask_camera",
        mask_lidar_key="mask_lidar",
        mask_camera_bits_key="mask_camera_bits",
        camera_names_key="camera_names",
        mask_camera_select_names=cam_types,
    ),
    class_names=class_names,
    empty_label=empty_label,
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
)

# -------------------- Model capacity / schedule --------------------
embed_dims = 512
feature_dims = 256
ffn_feedforward_channels = 1024
num_layers = 6
num_points = 4
num_refines = [1, 2, 4, 8, 16, 32]
num_levels = 1
image_feature_channels = 512
image_feature_output_divisor = 16
query_budget_per_view = 500
num_query = query_budget_per_view * num_views
initial_num_query = 100
grow_num_query = 100
grow_every_epochs = 40
total_epochs = 200
val_interval = 20
global_batch_size = 64
batch_size = global_batch_size
selected_frame_indices = [0]
selected_num_frames = len(selected_frame_indices)
online_depth_enabled = _bool_env("ADAOCC_ONLINE_DEPTH", False)
precomputed_depth_for_online = _bool_env("ADAOCC_PRECOMPUTED_DEPTH_FOR_ONLINE", False)
precomputed_depth_points_enabled = (not online_depth_enabled) or precomputed_depth_for_online
raw_depth_from_images = _bool_env("ADAOCC_RAW_DEPTH_FROM_IMAGES", True)
eval_padding = True
eval_score_thr = 0.4
eval_ctr_dist_thr = 1.0
eval_voxel_topk = 10
occ_out_channels = 1
prototype_npz_path = str(_Path(dataset_root) / "occscannet_prototypes_pca256.npz")
prototype_bridge_path = str(_Path(dataset_root) / "occscannet_bridge.json")

query_init_mix_cfg = dict(
    enabled=True,
    lidar_ratio=0.7,
    random_ratio=0.3,
    random_mode="uniform_pc_range",
)
train_view_dropout_cfg = dict(
    enabled=False,
    keep_count_range=[1, 1],
    camera_pool=cam_types,
    fallback_to_all_if_empty_points=True,
    max_resample_attempts=1,
)
online_depth_cfg = dict(
    enabled=online_depth_enabled,
    mode="parallel",
    model_path=depth_anything_model_path,
    depth_anything_root=str(_repo_root / "Depth_Anything_V2" / "metric_depth"),
    encoder="vitb",
    features=128,
    out_channels=[96, 192, 384, 768],
    max_depth=20.0,
    input_size=518,
    depth_min=0.1,
    depth_max=7.5,
    sample_stride=1,
    max_points_total=0,
    point_cloud_range=point_cloud_range,
    # mmcv image loaders output BGR by default; PackOnlineDepthInputs runs
    # before RandomTransformImage so online DA sees the original 968x1296 image.
    image_is_bgr=True,
)

# -------------------- Image / TPV branches --------------------
img_backbone = None
img_neck = None
img_feature_fusion = None
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True,
)
img_encoder = dict(
    type="ModularOccEncoder",
    num_views=num_views,
    num_frames=selected_num_frames,
    chunk_by_frame=True,
    preprocess_cfg=dict(
        resize_mode="patch_aligned_pad_bottom_right",
        patch_size=16,
        norm_type="imagenet",
    ),
    image_backbone_cfg=dict(
        type="TimmFeatureBackbone",
        model_name="tf_efficientnet_b7_ns",
        checkpoint_path=efficientnet_checkpoint_path,
        pretrained=False,
        features_only=True,
        out_indices=(3,),
        freeze=True,
        strict_checkpoint=True,
        expected_output_stride=image_feature_output_divisor,
        out_channels=image_feature_channels,
    ),
    # EfficientNet features are projected to the same 512-channel, stride-16 image
    # level consumed by the existing AdaOcc decoder/head.
    pyramid_adapter_cfg=dict(
        enabled=True,
        view_batch_size=4,
        output_in_channels=image_feature_channels,
        output_channels=None,
        upsample_output_divisor=image_feature_output_divisor,
        pyramid=dict(
            output_divisors=[image_feature_output_divisor],
            downsample_mode="bilinear",
            num_levels=num_levels,
            align_corners=False,
        ),
    ),
)

pts_voxel_layer = dict(
    max_num_points=10,
    voxel_size=pc_voxel_size,
    deterministic=False,
    max_voxels=(90000, 120000),
    point_cloud_range=point_cloud_range,
)
pts_voxel_encoder = dict(type="HardSimpleVFE", num_features=5)
pts_middle_encoder = dict(
    type="SparseEncoderTPVOnly",
    in_channels=5,
    sparse_shape=sparse_shape,
    output_channels=128,
    order=("conv", "norm", "act"),
    encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
    encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
    block_type="basicblock",
    return_middle_feats=True,
)
pts_backbone = dict(
    type="SECOND",
    in_channels=1152,
    out_channels=[128, 256],
    layer_nums=[5, 5],
    layer_strides=[1, 2],
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    conv_cfg=dict(type="Conv2d", bias=False),
)
pts_neck = dict(
    type="SECONDFPN",
    in_channels=[128, 256],
    out_channels=[256, 256],
    upsample_strides=[1, 2],
    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)
tpv_encoder = dict(
    type="TPVLiteEncoder",
    in_channels=128,
    skip_in_channels=64,
    fpn_channels=128,
    out_channels=embed_dims,
    use_skip=True,
)

model = dict(
    type="AdaOcc",
    data_preprocessor=dict(type="BaseDataPreprocessor"),
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=16),
    ),
    stop_prev_grad=0,
    train_view_dropout=train_view_dropout_cfg,
    online_depth=online_depth_cfg,
    use_external_img_encoder=True,
    enable_tpv_feature_branch=True,
    enable_pts_feature_branch=False,
    img_backbone=img_backbone,
    img_neck=img_neck,
    img_encoder=img_encoder,
    img_feature_fusion=img_feature_fusion,
    pts_voxel_layer=pts_voxel_layer,
    pts_voxel_encoder=pts_voxel_encoder,
    pts_middle_encoder=pts_middle_encoder,
    pts_backbone=pts_backbone,
    pts_neck=pts_neck,
    tpv_encoder=tpv_encoder,
    pts_bbox_head=dict(
        type="AdaOccHead",
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        query_budget_per_view=query_budget_per_view,
        pc_range=point_cloud_range,
        empty_label=empty_label,
        voxel_size=voxel_size,
        init_pos_lidar="curr",
        transformer=dict(
            type="AdaOccTransformer",
            embed_dims=embed_dims,
            feature_dims=0,
            ffn_feedforward_channels=ffn_feedforward_channels,
            score_mode="semantic",
            occ_out_channels=occ_out_channels,
            num_frames=selected_num_frames,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            use_pts_sampling=False,
            use_tpv_sampling=True,
            tpv_fusion_mode="query_attn",
            query_allocator=dict(
                enabled=False,
                switch_layer=2,
                context_ratio=0.5,
                detail_jitter_std=0.01,
            ),
            pc_range=point_cloud_range,
        ),
        feature_supervision=dict(
            enabled=False,
            prototype_decode=False,
            feature_dims=feature_dims,
            prototype_npz_path=prototype_npz_path,
            prototype_bridge_path=prototype_bridge_path,
        ),
        loss_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_pts=dict(type="SmoothL1Loss", _scope_="mmdet", beta=0.2, loss_weight=0.5),
    ),
    train_cfg=dict(
        pts=dict(
            use_raw_occscannet_gt=use_raw_occscannet_gt,
            point_loss_mode="dcd",
            cls_weights=cls_weights,
            rare_classes=rare_classes,
            rare_weights=12,
            hard_camera_mask=True,
            tail_focus=dict(
                enabled=False,
                policy="ema_freq",
                ema_momentum=0.9,
                freq_thr=0.02,
                min_tail_classes=8,
                max_tail_classes=24,
                sync_stats=True,
                fallback="rare_classes",
                tail_weight=12,
            ),
            hard_mining=dict(enabled=False, pred_topk_ratio=0.5, min_keep=1024),
            gt_balance=dict(enabled=False, per_class_cap=8192, tail_min_keep=128, sample_mode="random"),
            occ_target_cfg=dict(pos_dist_thr=0.10, neg_dist_thr=0.25),
            containment_loss=dict(
                enabled=True,
                layers=["d4", "d5"],
                weight=0.10,
                target_margin=0.04,
                topk_boxes=4,
                beta=0.04,
                outside_margin=0.0,
                use_camera_mask=False,
                weight_schedule=dict(
                    enabled=True,
                    start_epoch=20,
                    end_epoch=60,
                    start_weight=0.10,
                    end_weight=0.30,
                ),
            ),
            empty_dist_thr=0.04,
            cls_ignore_dist_thr=0.08,
            decoder_loss_decay=0.9,
            query_init_mix=query_init_mix_cfg,
            progressive_query_schedule=dict(
                enabled=True,
                initial_num_query=initial_num_query,
                grow_every_epochs=grow_every_epochs,
                grow_num_query=grow_num_query,
                start_epoch=1,
                max_active_query=num_query,
                apply_in_eval=False,
            ),
        )
    ),
    test_cfg=dict(
        pts=dict(
            score_thr=eval_score_thr,
            padding=eval_padding,
            ctr_dist_thr=eval_ctr_dist_thr,
            voxel_topk=eval_voxel_topk,
        )
    ),
)

# -------------------- Data/Pipeline --------------------
ida_aug_conf = dict(
    resize_lim=(scan_resize, scan_resize),
    final_dim=scan_final_dim,
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=scan_raw_h,
    W=scan_raw_w,
    rand_flip=False,
)
imdecode_backend = "pillow"
depth_points_cfg = dict(
    type="LoadPointsFromMultiViewDepth",
    sample_stride=1,
    sample_stride_current=1,
    sample_stride_history=1,
    max_points_total=0,
    depth_min=0.1,
    depth_max=7.5,
    depth_format="raw_uint16_mm" if raw_depth_from_images else "adaocc_rgba_float32",
    raw_depth_scale=1000.0,
    scale_intrinsic_to_depth=raw_depth_from_images,
    coord_convention="opencv",
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    time_dim=4,
    strict_depth_exist=True,
    fallback_depth_from_image_path=True,
    fallback_same_stem_depth_from_image=raw_depth_from_images,
    prefer_same_stem_depth_from_image=raw_depth_from_images,
    output_view_ids_key="depth_point_view_ids",
)
pack_meta_keys = (
    "sample_idx", "sample_token", "scene_name",
    "filename", "ori_shape", "img_shape", "pad_shape",
    "ego2occ", "ego2img", "ego2lidar",
    "ego2global_translation", "ego2global_rotation",
    "img_timestamp",
)

_occ_loader = dict(
    type="LoadOcc3DFromFile",
    occ_root=occ_root,
    path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    mask_camera_bits_key=dataset_cfg["occ_io"]["mask_camera_bits_key"],
    camera_names_key=dataset_cfg["occ_io"]["camera_names_key"],
    mask_camera_select_names=dataset_cfg["occ_io"]["mask_camera_select_names"],
    class_names=dataset_cfg["class_names"],
    empty_label=dataset_cfg["empty_label"],
)
_select_current_frame = dict(type="SelectTemporalFrames", frame_indices=selected_frame_indices)
_depth_point_steps = (
    [
        depth_points_cfg,
        dict(type="LiDARToOccSpace"),
    ]
    if precomputed_depth_points_enabled else []
)
_point_filter_steps = (
    [dict(type="PointsRangeFilterWithViewIds", point_cloud_range=point_cloud_range)]
    if precomputed_depth_points_enabled else []
)

train_pipeline = [
    dict(type="AdaOccLoadMultiViewImageFromFiles", to_float32=False, color_type="color", imdecode_backend=imdecode_backend),
    dict(type="LoadMultiViewImageFromMultiSweeps", sweeps_num=8, imdecode_backend=imdecode_backend, cam_types=cam_types),
    _select_current_frame,
    dict(type="PackOnlineDepthInputs", cam_types=cam_types, strict=True, enabled=online_depth_enabled),
    *_depth_point_steps,
    _occ_loader,
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=True),
    *_point_filter_steps,
    dict(
        type="PackOcc3DInputs",
        meta_keys=pack_meta_keys,
        extra_input_keys=("depth_point_view_ids", "online_depth"),
    ),
]

test_pipeline = [
    dict(type="AdaOccLoadMultiViewImageFromFiles", to_float32=False, color_type="color", imdecode_backend=imdecode_backend),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=8,
        test_mode=True,
        force_offline=True,
        imdecode_backend=imdecode_backend,
        cam_types=cam_types,
    ),
    _select_current_frame,
    dict(type="PackOnlineDepthInputs", cam_types=cam_types, strict=True, enabled=online_depth_enabled),
    *_depth_point_steps,
    _occ_loader,
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=False),
    *_point_filter_steps,
    dict(
        type="PackOcc3DInputs",
        meta_keys=pack_meta_keys,
        extra_input_keys=("depth_point_view_ids", "online_depth"),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type="BatchAlignedDefaultSampler",
        shuffle=True,
        round_up=True,
        batch_size=global_batch_size,
        batch_size_scale="global",
    ),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=False,
    ),
    collate_fn=dict(type="pseudo_collate"),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        occ_root=occ_root,
        dataset_cfg=dataset_cfg,
        test_mode=True,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

_focus_eval_cfg = dict(
    enabled=True,
    policy="ema_freq",
    ema_momentum=0.9,
    freq_thr=0.02,
    min_classes=2,
    max_classes=2,
    fallback="rare_classes",
    rare_classes=rare_classes,
)
_evaluator_common = dict(
    type="Occ3DMetric",
    use_raw_occscannet_gt=use_raw_occscannet_gt,
    occ_root=occ_root,
    occ_path_template=dataset_cfg["occ_io"]["path_template"],
    semantics_key=dataset_cfg["occ_io"]["semantics_key"],
    mask_camera_key=dataset_cfg["occ_io"]["mask_camera_key"],
    mask_lidar_key=dataset_cfg["occ_io"]["mask_lidar_key"],
    mask_camera_bits_key=dataset_cfg["occ_io"]["mask_camera_bits_key"],
    camera_names_key=dataset_cfg["occ_io"]["camera_names_key"],
    mask_camera_select_names=dataset_cfg["occ_io"]["mask_camera_select_names"],
    empty_label=dataset_cfg["empty_label"],
    use_camera_mask=True,
    pc_range=dataset_cfg["pc_range"],
    voxel_size=dataset_cfg["voxel_size"],
    class_names=dataset_cfg["class_names"],
    miou_num_workers=32,
    focus_eval=_focus_eval_cfg,
)
val_evaluator = dict(_evaluator_common, ann_file=val_ann_file)
test_evaluator = dict(_evaluator_common, ann_file=test_ann_file)

# -------------------- Optimization / runtime --------------------
optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={
        "img_backbone": dict(lr_mult=0.1),
        "sampling_offset": dict(lr_mult=0.1),
    }),
    loss_scale=512.0,
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", T_max=total_epochs, by_epoch=True, begin=0, end=total_epochs, eta_min=2e-4 * 1e-3),
]
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="AdaOccIterTimerHook"),
    logger=dict(type="LoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=1, save_last=True),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
log_processor = dict(type="AdaOccLogProcessor", window_size=1, by_epoch=True)
visualizer = dict(type="Visualizer", vis_backends=[dict(type="TensorboardVisBackend")])
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
randomness = dict(seed=0, deterministic=False)
resume_from = None

# Keep MMEngine's config dump/visualizer path valid: imported helper objects are
# only needed while evaluating this file and should not become config fields.
del _Path, _os, _sys, _config_dir, _repo_root
del _bool_env
