from mmengine.runner import amp
from mmdet3d.registry import MODELS

try:
    from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
except ImportError:  # pragma: no cover
    from mmdet3d.models.middle_encoders import SparseEncoder  # type: ignore
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE  # type: ignore

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class SparseEncoderTPVOnly(SparseEncoder):
    """SparseEncoder variant that skips unused BEV dense output in TPV-only mode.

    When `return_middle_feats=True`, AdaOcc TPV path only consumes `encode_features`.
    The parent `SparseEncoder` still runs `conv_out + dense()` to materialize BEV
    features that are then discarded. This variant avoids that work and returns
    `(None, encode_features)` directly.
    """

    def __init__(self, *args, skip_bev_output=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_bev_output = bool(skip_bev_output)
        if self.skip_bev_output and self.return_middle_feats:
            self.conv_out = None

    @amp.autocast(enabled=False)
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        if self.return_middle_feats and self.skip_bev_output:
            return None, encode_features

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()
        n, c, d, h, w = spatial_features.shape
        spatial_features = spatial_features.view(n, c * d, h, w)

        if self.return_middle_feats:
            return spatial_features, encode_features
        return spatial_features
