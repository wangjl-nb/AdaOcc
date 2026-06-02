import torch
import torch.nn.functional as F
from ..bbox.utils import decode_points, encode_points
from ..utils import rotation_3d_in_axis, DUMP, debug_is_finite
from ..csrc.wrapper import msmv_sampling, msmv_sampling_pytorch


def make_sample_points(query_points, offset, pc_range):
    '''
    query_points: [B, Q, P, 3] (x, y, z)
    offset: [B, Q, G, P, 3]
    '''
    xyz = decode_points(query_points, pc_range)  # [B, Q, 3]
    xyz = xyz[..., None, None, :]  # [B, Q, 1, 1, 3]
    sample_xyz = xyz + offset  # [B, Q, G, P, 3]
    return sample_xyz


def sampling_4d(sample_points, mlvl_feats, scale_weights, occ2img, image_h, image_w, num_views=6, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        occ2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    stage = getattr(DUMP, 'stage_count', 'na')
    debug_is_finite(f'sampling_4d.stage{stage}.sample_points', sample_points)
    debug_is_finite(f'sampling_4d.stage{stage}.occ2img', occ2img)
    debug_is_finite(f'sampling_4d.stage{stage}.scale_weights', scale_weights)

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
    N = num_views
    
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    # get the projection matrix
    occ2img = occ2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    occ2img = occ2img.expand(B, T*N, Q, G * P, 4, 4)
    occ2img = occ2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views (force fp32 to avoid fp16 overflow)
    with torch.cuda.amp.autocast(enabled=False):
        sample_points_cam = torch.matmul(occ2img.float(), sample_points.float()).squeeze(-1)  # [B, T, N, Q, GP, 4]

        # homo coord -> pixel coord
        homo = sample_points_cam[..., 2:3]
        homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
        sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
        sample_points_cam = torch.nan_to_num(sample_points_cam, nan=0.0, posinf=2.0, neginf=-2.0)
    debug_is_finite(f'sampling_4d.stage{stage}.homo', homo)

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h
    sample_points_cam = sample_points_cam.clamp(-2.0, 2.0)
    debug_is_finite(f'sampling_4d.stage{stage}.sample_points_cam', sample_points_cam)

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # for visualization only
    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1).cpu(),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask.cpu(),
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    # guard against N==1 to avoid division by zero
    if N > 1:
        view_coord = i_view[..., None].float() / (N - 1)
    else:
        view_coord = torch.zeros_like(i_view[..., None], dtype=sample_points_cam.dtype)
    sample_points_cam = torch.cat([sample_points_cam, view_coord], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)
    sample_points_cam = sample_points_cam.contiguous()
    sample_points_cam = sample_points_cam.to(mlvl_feats[0].dtype)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)
    scale_weights = scale_weights.contiguous()
    scale_weights = scale_weights.to(mlvl_feats[0].dtype)

    # multi-scale multi-view grid sample
    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)
    debug_is_finite(f'sampling_4d.stage{stage}.final', final)

    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final


def sampling_pts_feats(sample_points, pts_feats, occ2lidar, pc_range):
    stage = getattr(DUMP, 'stage_count', 'na')
    debug_is_finite(f'sampling_pts_feats.stage{stage}.sample_points', sample_points)
    debug_is_finite(f'sampling_pts_feats.stage{stage}.occ2lidar', occ2lidar)
    C = pts_feats.shape[1]
    B, Q, G, P, _ = sample_points.shape  # [B, Q, G, P, 3]
    sample_points = sample_points.permute(0, 2, 1, 3, 4)
    sample_points = sample_points.reshape(B*G, Q, P, 3)  # [BG, Q, P, 3]

    occ2lidar = occ2lidar[:, None, None, None, :, :]  # [B, 1, 1, 1, 4, 4]
    occ2lidar = occ2lidar.expand(B, G, Q, P, 4, 4)
    occ2lidar = occ2lidar.reshape(B*G, Q, P, 4, 4)

    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)[..., None] # [BG, Q, P, 4, 1]
    with torch.cuda.amp.autocast(enabled=False):
        sample_points = torch.matmul(occ2lidar.float(), sample_points.float()).squeeze(-1)

    norm_sample_points = encode_points(sample_points[..., :3], pc_range)
    norm_sample_points = norm_sample_points[..., :2] * 2 - 1 # [BG, Q, P, 2]
    norm_sample_points = torch.nan_to_num(norm_sample_points, nan=0.0, posinf=2.0, neginf=-2.0)
    norm_sample_points = norm_sample_points.clamp(-2.0, 2.0)
    debug_is_finite(f'sampling_pts_feats.stage{stage}.norm_sample_points', norm_sample_points)
    norm_sample_points = norm_sample_points.to(pts_feats.dtype)

    feat = F.grid_sample(pts_feats, norm_sample_points, padding_mode='zeros', align_corners=True) 
    debug_is_finite(f'sampling_pts_feats.stage{stage}.feat', feat)
    feat = feat.reshape(B, G, C, Q, P)
    feat = feat.permute(0, 3, 1, 4, 2)  # [B, Q, G, P, C]

    return feat


def sampling_tpv_feats(sample_points, tpv_feats, pc_range, plane_weights=None):
    """Sample TPV features (XY/XZ/YZ planes) at query-centered 3D points.

    Args:
        sample_points (Tensor): [B, Q, G, P, 3] in pc_range coordinate space.
        tpv_feats (dict[str, Tensor]): {
            'xy': [B*G, C, Ny, Nx],
            'xz': [B*G, C, Nz, Nx],
            'yz': [B*G, C, Nz, Ny],
        }
        pc_range (list[float] or Tensor): [x_min, y_min, z_min, x_max, y_max, z_max].
        plane_weights (Tensor | None): [B, Q, G, P, 3], soft weights for xy/xz/yz.

    Returns:
        Tensor: [B, Q, G, P, C]
    """
    stage = getattr(DUMP, 'stage_count', 'na')
    debug_is_finite(f'sampling_tpv_feats.stage{stage}.sample_points', sample_points)

    required = ['xy', 'xz', 'yz']
    if not isinstance(tpv_feats, dict):
        raise TypeError(f'tpv_feats must be dict, got {type(tpv_feats)}')
    for key in required:
        if key not in tpv_feats:
            raise KeyError(f'tpv_feats missing key "{key}"')
        if not isinstance(tpv_feats[key], torch.Tensor) or tpv_feats[key].dim() != 4:
            raise ValueError(
                f'tpv_feats["{key}"] must be Tensor[B*G, C, H, W], '
                f'got {type(tpv_feats[key])} shape={getattr(tpv_feats[key], "shape", None)}')

    xy_feat = tpv_feats['xy']
    xz_feat = tpv_feats['xz']
    yz_feat = tpv_feats['yz']
    C = xy_feat.shape[1]

    B, Q, G, P, _ = sample_points.shape
    sample_points = sample_points.permute(0, 2, 1, 3, 4).reshape(B * G, Q, P, 3)  # [BG, Q, P, 3]
    norm_sample_points = encode_points(sample_points, pc_range)  # [BG, Q, P, 3]
    norm_sample_points = torch.nan_to_num(norm_sample_points, nan=0.0, posinf=1.0, neginf=0.0)
    norm_sample_points = norm_sample_points.clamp(0.0, 1.0)

    grid_xy = torch.stack([norm_sample_points[..., 0], norm_sample_points[..., 1]], dim=-1)
    grid_xz = torch.stack([norm_sample_points[..., 0], norm_sample_points[..., 2]], dim=-1)
    grid_yz = torch.stack([norm_sample_points[..., 1], norm_sample_points[..., 2]], dim=-1)
    grid_xy = (grid_xy * 2.0 - 1.0).to(dtype=xy_feat.dtype)
    grid_xz = (grid_xz * 2.0 - 1.0).to(dtype=xz_feat.dtype)
    grid_yz = (grid_yz * 2.0 - 1.0).to(dtype=yz_feat.dtype)

    sampled_xy = F.grid_sample(
        xy_feat, grid_xy, padding_mode='zeros', align_corners=True)  # [BG, C, Q, P]
    sampled_xz = F.grid_sample(
        xz_feat, grid_xz, padding_mode='zeros', align_corners=True)  # [BG, C, Q, P]
    sampled_yz = F.grid_sample(
        yz_feat, grid_yz, padding_mode='zeros', align_corners=True)  # [BG, C, Q, P]
    debug_is_finite(f'sampling_tpv_feats.stage{stage}.sampled_xy', sampled_xy)
    debug_is_finite(f'sampling_tpv_feats.stage{stage}.sampled_xz', sampled_xz)
    debug_is_finite(f'sampling_tpv_feats.stage{stage}.sampled_yz', sampled_yz)

    sampled_xy = sampled_xy.reshape(B, G, C, Q, P).permute(0, 3, 1, 4, 2)  # [B, Q, G, P, C]
    sampled_xz = sampled_xz.reshape(B, G, C, Q, P).permute(0, 3, 1, 4, 2)
    sampled_yz = sampled_yz.reshape(B, G, C, Q, P).permute(0, 3, 1, 4, 2)

    if plane_weights is None:
        return (sampled_xy + sampled_xz + sampled_yz) / 3.0

    if plane_weights.shape != (B, Q, G, P, 3):
        raise ValueError(
            f'plane_weights shape mismatch: expected {(B, Q, G, P, 3)}, '
            f'got {tuple(plane_weights.shape)}')
    weights = torch.softmax(plane_weights, dim=-1).to(sampled_xy.dtype)[..., None]
    sampled = torch.stack([sampled_xy, sampled_xz, sampled_yz], dim=-2)  # [B, Q, G, P, 3, C]
    return (sampled * weights).sum(dim=-2)
