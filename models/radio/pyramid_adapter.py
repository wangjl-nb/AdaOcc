import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ImagePyramidAdapter(BaseModule):
    """Expand a single spatial feature map into AdaOcc pyramid levels."""

    def __init__(self, view_batch_size=None, output_in_channels=None, output_channels=None,
                 upsample_output_divisor=1, per_level_output_proj=False, pyramid=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.view_batch_size = None if view_batch_size is None else int(view_batch_size)
        if self.view_batch_size is not None and self.view_batch_size <= 0:
            self.view_batch_size = None

        self.output_in_channels = None if output_in_channels is None else int(output_in_channels)
        self.output_channels = None if output_channels is None else int(output_channels)
        self.per_level_output_proj = bool(per_level_output_proj)

        pyramid_cfg = dict(pyramid or {})
        divisors = pyramid_cfg.get("output_divisors", [4, 8, 16, 32])
        self.output_divisors = [int(v) for v in divisors]
        self.num_levels = int(pyramid_cfg.get("num_levels", len(self.output_divisors)))
        self.output_divisors = self.output_divisors[: self.num_levels]
        self.downsample_mode = str(pyramid_cfg.get("downsample_mode", "bilinear"))
        self.align_corners = bool(pyramid_cfg.get("align_corners", False))
        self.upsample_output_divisor = int(upsample_output_divisor)

        self.output_proj = None
        if self.output_channels is not None:
            if self.per_level_output_proj:
                if self.output_in_channels is None:
                    self.output_proj = torch.nn.ModuleList(
                        [
                            torch.nn.LazyConv2d(self.output_channels, kernel_size=1, bias=True)
                            for _ in range(self.num_levels)
                        ]
                    )
                else:
                    self.output_proj = torch.nn.ModuleList(
                        [
                            torch.nn.Conv2d(
                                self.output_in_channels,
                                self.output_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True,
                            )
                            for _ in range(self.num_levels)
                        ]
                    )
            elif self.output_in_channels is None:
                self.output_proj = torch.nn.LazyConv2d(self.output_channels, kernel_size=1, bias=True)
            else:
                self.output_proj = torch.nn.Conv2d(
                    self.output_in_channels,
                    self.output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )

    def _compute_level_sizes(self, height, width):
        return [
            (
                max((int(height) + divisor - 1) // divisor, 1),
                max((int(width) + divisor - 1) // divisor, 1),
            )
            for divisor in self.output_divisors
        ]

    def _compute_upsample_size(self, height, width):
        if self.upsample_output_divisor <= 1:
            return int(height), int(width)
        divisor = int(self.upsample_output_divisor)
        return (
            max((int(height) + divisor - 1) // divisor, 1),
            max((int(width) + divisor - 1) // divisor, 1),
        )

    def _interpolate(self, feature, size):
        if tuple(feature.shape[-2:]) == tuple(size):
            return feature
        if self.downsample_mode in ("nearest", "area", "nearest-exact"):
            return F.interpolate(feature, size=size, mode=self.downsample_mode)
        return F.interpolate(
            feature,
            size=size,
            mode=self.downsample_mode,
            align_corners=self.align_corners,
        )

    def _project(self, feature, level_idx=None):
        if self.output_proj is None:
            return feature
        if not feature.is_contiguous():
            feature = feature.contiguous()
        restore_shape = None
        if feature.dim() == 5:
            batch_size, total_views, channels, height, width = feature.shape
            feature = feature.reshape(batch_size * total_views, channels, height, width)
            restore_shape = (batch_size, total_views)
        elif feature.dim() != 4:
            raise ValueError(
                "projection expects Tensor[B, TN, C, H, W] or Tensor[N, C, H, W], "
                f'got shape={tuple(feature.shape)}'
            )
        in_dtype = feature.dtype
        proj = self.output_proj
        if isinstance(proj, torch.nn.ModuleList):
            if level_idx is None:
                raise ValueError("level_idx is required when using per-level output projection")
            proj = proj[level_idx]
        proj_dtype = proj.weight.dtype
        if in_dtype != proj_dtype:
            feature = feature.to(dtype=proj_dtype)
        feature = proj(feature)
        if feature.dtype != in_dtype:
            feature = feature.to(dtype=in_dtype)
        if restore_shape is not None:
            batch_size, total_views = restore_shape
            feature = feature.reshape(batch_size, total_views, *feature.shape[1:])
        return feature

    def forward(self, features, processed_views_batch):
        if isinstance(features, (list, tuple)):
            if len(features) != self.num_levels:
                raise ValueError(
                    f"precomputed level count mismatch: expected {self.num_levels}, got {len(features)}"
                )
            outputs = []
            for level_idx, level_feat in enumerate(features):
                if not isinstance(level_feat, torch.Tensor) or level_feat.dim() != 5:
                    raise ValueError(
                        "precomputed pyramid levels must be Tensor[B, TN, C, H, W], "
                        f'got type={type(level_feat)} shape={getattr(level_feat, "shape", None)}'
                    )
                outputs.append(self._project(level_feat, level_idx=level_idx).contiguous())
            return outputs

        if not isinstance(features, torch.Tensor) or features.dim() != 5:
            raise ValueError(
                "features must be Tensor[B, TN, C, H, W], "
                f'got type={type(features)} shape={getattr(features, "shape", None)}'
            )
        if len(processed_views_batch) != int(features.shape[0]):
            raise ValueError(
                f"processed_views_batch length mismatch: {len(processed_views_batch)} vs B={features.shape[0]}"
            )

        batch_size, total_views = features.shape[:2]
        per_sample_levels = []
        for sample_idx in range(batch_size):
            sample_views = processed_views_batch[sample_idx]
            if len(sample_views) != total_views:
                raise ValueError(
                    f"sample {sample_idx} processed-view count mismatch: {len(sample_views)} vs {total_views}"
                )

            image_hw = [tuple(int(v) for v in sample_views[view_idx]["img"].shape[-2:]) for view_idx in range(total_views)]
            ref_hw = image_hw[0]
            if any(hw != ref_hw for hw in image_hw[1:]):
                raise ValueError(f"sample {sample_idx} has inconsistent processed image sizes: {image_hw}")

            target_hw = self._compute_upsample_size(*ref_hw)
            level_sizes = self._compute_level_sizes(*ref_hw)
            upsampled = self._interpolate(features[sample_idx], size=target_hw)

            sample_levels = []
            for level_size in level_sizes:
                level = self._interpolate(upsampled, size=level_size)
                level = self._project(level)
                sample_levels.append(level.unsqueeze(0))
            per_sample_levels.append(sample_levels)

        outputs = []
        for level_idx in range(self.num_levels):
            outputs.append(
                torch.cat([sample_levels[level_idx] for sample_levels in per_sample_levels], dim=0)
            )
        return outputs
