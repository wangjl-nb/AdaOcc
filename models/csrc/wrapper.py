import os

import torch
import torch.nn.functional as F


def _env_flag(name, default='0'):
    value = os.getenv(name, default)
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

if _env_flag('ADAOCC_DISABLE_MSMV_CUDA'):
    # The released single-level RADIO baseline supports this PyTorch fallback.
    # Do not try to import the optional extension when users explicitly disable
    # it; otherwise smoke runs show a scary but harmless ImportError warning.
    MSMV_CUDA = False
else:
    try:
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_forward, _ms_deform_attn_cuda_c2345_backward
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_forward, _ms_deform_attn_cuda_c23456_backward
        MSMV_CUDA = True
    except ImportError as e:
        print('Warning: optional MSMV CUDA extension is unavailable; using PyTorch fallback.')
        print('Error message:', e)
        MSMV_CUDA = False


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 3]
    scale_weights: [B, Q, P, 4]
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, C, _, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    final = torch.zeros(
        [B, C, Q, P],
        device=mlvl_feats[0].device,
        dtype=mlvl_feats[0].dtype,
    )

    def _safe_grid_sample(feat, grid):
        kwargs = dict(mode='bilinear', padding_mode='zeros', align_corners=True)
        if feat.requires_grad and grid.requires_grad:
            base = F.grid_sample(feat.detach(), grid.detach(), **kwargs)
            grad_feat = F.grid_sample(feat, grid.detach(), **kwargs)
            grad_grid = F.grid_sample(feat.detach(), grid, **kwargs)
            return grad_feat + grad_grid - base
        return F.grid_sample(feat, grid, **kwargs)

    for lvl, feat in enumerate(mlvl_feats):
        if feat.shape[2] == 1:
            # For single-view runs, use the much more stable 2D grid_sample path
            # instead of the 5D volume sampler with a degenerate depth axis.
            grid_2d = sampling_locations[..., :2] * 2 - 1  # [B, Q, P, 2]
            out = _safe_grid_sample(feat[:, :, 0], grid_2d)  # [B, C, Q, P]
        else:
            grid_3d = sampling_locations * 2 - 1
            grid_3d = grid_3d[:, :, :, None, :]  # [B, Q, P, 1, 3]
            out = _safe_grid_sample(feat, grid_3d)[..., 0]  # [B, C, Q, P]
        out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final += out

    return final.permute(0, 2, 1, 3)


class MSMVSamplingC2345(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c2345_forward)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c2345_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c2345_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight


class MSMVSamplingC23456(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c23456_forward)
        return _ms_deform_attn_cuda_c23456_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c23456_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c23456_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    def _to_cuda_layout(feat):
        if feat.dim() != 5:
            raise ValueError(f'Expected 5D feature tensor, got shape={tuple(feat.shape)}')

        # Training code paths already convert features to channel-last
        # [B, N, H, W, C] before calling the CUDA extension. Standalone
        # benchmarks may still pass channel-first [B, C, N, H, W].
        if feat.shape[1] <= 8 and feat.shape[-1] > 8:
            return feat.contiguous()
        if feat.shape[2] <= 8 and feat.shape[1] > 8:
            return feat.permute(0, 2, 3, 4, 1).contiguous()
        raise ValueError(
            'Unable to infer MSMV feature layout; expected [B,C,N,H,W] or [B,N,H,W,C], '
            f'got shape={tuple(feat.shape)}')

    def _to_pytorch_layout(feat):
        if feat.dim() != 5:
            raise ValueError(f'Expected 5D feature tensor, got shape={tuple(feat.shape)}')

        # Channel-last layout used by the fused CUDA path.
        if feat.shape[1] <= 8 and feat.shape[-1] > 8:
            return feat.permute(0, 4, 1, 2, 3).contiguous()
        # Native channel-first layout expected by the PyTorch fallback.
        if feat.shape[2] <= 8 and feat.shape[1] > 8:
            return feat.contiguous()
        raise ValueError(
            'Unable to infer MSMV feature layout; expected [B,C,N,H,W] or [B,N,H,W,C], '
            f'got shape={tuple(feat.shape)}')

    # The fused CUDA backward is not stable when gradients need to flow back
    # into image feature values (for example, trainable external image encoders).
    # Keep the fast path for frozen-image-feature runs, and fall back only when
    # mlvl_feats themselves require gradients.
    need_value_grad = any(bool(getattr(feat, 'requires_grad', False)) for feat in mlvl_feats)

    if len(mlvl_feats) == 4 and MSMV_CUDA and not need_value_grad:
        mlvl_feats_cuda = [_to_cuda_layout(feat) for feat in mlvl_feats]
        out_dtype = mlvl_feats[0].dtype
        if out_dtype != torch.float32 or sampling_locations.dtype != torch.float32 or scale_weights.dtype != torch.float32:
            mlvl_feats_cuda = [feat.float() for feat in mlvl_feats_cuda]
            sampling_locations = sampling_locations.float()
            scale_weights = scale_weights.float()
            out = MSMVSamplingC2345.apply(*mlvl_feats_cuda, sampling_locations, scale_weights)
            return out.to(out_dtype)
        return MSMVSamplingC2345.apply(*mlvl_feats_cuda, sampling_locations, scale_weights)
    elif len(mlvl_feats) == 5 and MSMV_CUDA and not need_value_grad:
        mlvl_feats_cuda = [_to_cuda_layout(feat) for feat in mlvl_feats]
        out_dtype = mlvl_feats[0].dtype
        if out_dtype != torch.float32 or sampling_locations.dtype != torch.float32 or scale_weights.dtype != torch.float32:
            mlvl_feats_cuda = [feat.float() for feat in mlvl_feats_cuda]
            sampling_locations = sampling_locations.float()
            scale_weights = scale_weights.float()
            out = MSMVSamplingC23456.apply(*mlvl_feats_cuda, sampling_locations, scale_weights)
            return out.to(out_dtype)
        return MSMVSamplingC23456.apply(*mlvl_feats_cuda, sampling_locations, scale_weights)
    else:
        mlvl_feats = [_to_pytorch_layout(feat) for feat in mlvl_feats]
        out_dtype = mlvl_feats[0].dtype

        # The PyTorch grid_sample fallback is the stable path when gradients must
        # flow back into image feature values, but running its 5D CUDA backward in
        # fp16/bf16 can still trigger floating-point exceptions on H20/Hopper-class
        # GPUs. Keep the fallback in fp32 and cast the fused result back afterward.
        if out_dtype != torch.float32:
            mlvl_feats = [feat.float() for feat in mlvl_feats]
        if sampling_locations.dtype != torch.float32:
            sampling_locations = sampling_locations.float()
        if scale_weights.dtype != torch.float32:
            scale_weights = scale_weights.float()

        out = msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights)
        return out if out_dtype == torch.float32 else out.to(out_dtype)
