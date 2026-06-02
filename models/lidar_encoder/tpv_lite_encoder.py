import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TPVLiteEncoder(nn.Module):
    """Convert sparse 3D backbone features into TPV planes (XY/XZ/YZ)."""

    def __init__(self,
                 in_channels=128,
                 skip_in_channels=64,
                 fpn_channels=64,
                 out_channels=256,
                 use_skip=True,
                 **kwargs):
        super().__init__()
        self.in_channels = int(in_channels)
        self.skip_in_channels = int(skip_in_channels)
        self.fpn_channels = int(fpn_channels)
        self.out_channels = int(out_channels)
        self.use_skip = bool(use_skip)

        self.high_proj = nn.Sequential(
            nn.Conv3d(self.in_channels, self.fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.high_refine = nn.Sequential(
            nn.Conv3d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )

        self.skip_proj = nn.Sequential(
            nn.Conv3d(self.skip_in_channels, self.fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_residual = nn.Sequential(
            nn.Conv3d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.fpn_channels),
        )
        self.fpn_act = nn.ReLU(inplace=True)

        self.xy_head = self._make_plane_head()
        self.xz_head = self._make_plane_head()
        self.yz_head = self._make_plane_head()
        if kwargs:
            # Keep compatibility with previous config fields without changing behavior.
            self._unused_kwargs = kwargs

    def _make_plane_head(self):
        return nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _parse_input(feats):
        if isinstance(feats, dict):
            high = feats.get('high', None)
            skip = feats.get('skip', None)
        elif isinstance(feats, (tuple, list)):
            if len(feats) == 0:
                raise ValueError('TPVLiteEncoder input tuple/list cannot be empty')
            high = feats[0]
            skip = feats[1] if len(feats) > 1 else None
        else:
            high, skip = feats, None

        if not isinstance(high, torch.Tensor) or high.dim() != 5:
            raise TypeError(
                'TPVLiteEncoder expects high-resolution 3D feature tensor [B, C, D, H, W], '
                f'got type={type(high)} shape={getattr(high, "shape", None)}')
        if skip is not None:
            if not isinstance(skip, torch.Tensor) or skip.dim() != 5:
                raise TypeError(
                    'TPVLiteEncoder skip feature must be Tensor[B, C, D, H, W], '
                    f'got type={type(skip)} shape={getattr(skip, "shape", None)}')
            if skip.shape[0] != high.shape[0]:
                raise ValueError(
                    f'Skip batch size mismatch: high B={high.shape[0]}, skip B={skip.shape[0]}')
        return high, skip

    def forward(self, feats):
        high, skip = self._parse_input(feats)

        x = self.high_proj(high)
        x = self.high_refine(x)

        if self.use_skip and skip is not None:
            skip = self.skip_proj(skip)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(
                    x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = x + skip
            x = self.fpn_act(x + self.fpn_residual(x))

        # x: [B, C, D, H, W], where D/Z, H/Y, W/X.
        xy = x.mean(dim=2)  # [B, C, H, W]
        xz = x.mean(dim=3)  # [B, C, D, W]
        yz = x.mean(dim=4)  # [B, C, D, H]

        return dict(
            xy=self.xy_head(xy),
            xz=self.xz_head(xz),
            yz=self.yz_head(yz),
        )
