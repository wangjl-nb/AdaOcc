import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .input_adapter import AdaOccViewInputAdapter


_IMAGE_NORMALIZATIONS = {
    # RADIO's Hugging Face wrapper expects RGB in [0, 1].
    "radio": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    "identity": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    # Kept for defensive compatibility with old configs, but AdaOcc's public
    # baseline uses only norm_type="radio".
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


@MODELS.register_module()
class RadioViewPreprocessor(BaseModule):
    """Prepare AdaOcc image tensors into RADIO-ready view dictionaries.

    The current baseline only needs deterministic image padding plus RADIO
    normalization; depth, intrinsics, and poses are consumed elsewhere for
    point construction and decoder projection, not by the image backbone.
    Keeping this local avoids vendoring unrelated external preprocessing code.
    """

    def __init__(
        self,
        preprocess_cfg=None,
        lidar_injection="shared",
        tn_align_mode="strict",
        preserve_meta_keys=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.preprocess_cfg = dict(preprocess_cfg or {})
        self.input_adapter = AdaOccViewInputAdapter(
            lidar_injection=lidar_injection,
            tn_align_mode=tn_align_mode,
            preserve_meta_keys=preserve_meta_keys,
        )

    @staticmethod
    def _normalize_resize_mode(resize_mode):
        return str(resize_mode or "fixed_mapping")

    @staticmethod
    def _image_hw(img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and int(img.shape[-1]) == 3:
                return int(img.shape[0]), int(img.shape[1])
            if img.dim() == 3 and int(img.shape[0]) in (1, 3):
                return int(img.shape[1]), int(img.shape[2])
            if img.dim() == 2:
                return int(img.shape[0]), int(img.shape[1])
        if isinstance(img, np.ndarray):
            if img.ndim >= 2:
                return int(img.shape[0]), int(img.shape[1])
        raise TypeError(f"Unsupported image type for shape inference: {type(img)}")

    @staticmethod
    def _resolve_patch_aligned_hw(height, width, patch_size):
        patch = max(int(patch_size), 1)
        target_h = int(math.ceil(float(height) / float(patch)) * patch)
        target_w = int(math.ceil(float(width) / float(patch)) * patch)
        return target_h, target_w

    @staticmethod
    def _pad_tensor_hw_bottom_right(value, target_h, target_w, fill_value=0.0):
        if value.dim() == 2:
            pad_h = max(int(target_h) - int(value.shape[0]), 0)
            pad_w = max(int(target_w) - int(value.shape[1]), 0)
            if pad_h == 0 and pad_w == 0:
                return value
            return F.pad(value, (0, pad_w, 0, pad_h), value=float(fill_value))

        if value.dim() == 3:
            # HWC tensors are used by AdaOccViewInputAdapter.
            if int(value.shape[-1]) == 3:
                chw = value.permute(2, 0, 1).contiguous()
                pad_h = max(int(target_h) - int(chw.shape[-2]), 0)
                pad_w = max(int(target_w) - int(chw.shape[-1]), 0)
                if pad_h == 0 and pad_w == 0:
                    return value
                padded = F.pad(chw, (0, pad_w, 0, pad_h), value=float(fill_value))
                return padded.permute(1, 2, 0).contiguous()

            # Generic CHW tensors.
            pad_h = max(int(target_h) - int(value.shape[-2]), 0)
            pad_w = max(int(target_w) - int(value.shape[-1]), 0)
            if pad_h == 0 and pad_w == 0:
                return value
            return F.pad(value, (0, pad_w, 0, pad_h), value=float(fill_value))

        raise ValueError(
            f"Expected 2D/3D tensor for bottom-right padding, got shape={tuple(value.shape)}"
        )

    @staticmethod
    def _pad_array_hw_bottom_right(value, target_h, target_w, fill_value=0.0):
        if value.ndim == 2:
            pad_h = max(int(target_h) - int(value.shape[0]), 0)
            pad_w = max(int(target_w) - int(value.shape[1]), 0)
            if pad_h == 0 and pad_w == 0:
                return value
            return np.pad(
                value,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=fill_value,
            )

        if value.ndim == 3:
            pad_h = max(int(target_h) - int(value.shape[0]), 0)
            pad_w = max(int(target_w) - int(value.shape[1]), 0)
            if pad_h == 0 and pad_w == 0:
                return value
            return np.pad(
                value,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=fill_value,
            )

        raise ValueError(
            f"Expected 2D/3D ndarray for bottom-right padding, got shape={value.shape}"
        )

    @classmethod
    def _pad_hw_bottom_right(cls, value, target_h, target_w, fill_value=0.0):
        if isinstance(value, torch.Tensor):
            return cls._pad_tensor_hw_bottom_right(
                value, target_h=target_h, target_w=target_w, fill_value=fill_value
            )
        if isinstance(value, np.ndarray):
            return cls._pad_array_hw_bottom_right(
                value, target_h=target_h, target_w=target_w, fill_value=fill_value
            )
        raise TypeError(f"Unsupported type for bottom-right padding: {type(value)}")

    @staticmethod
    def _to_chw_float01_image(img, view_idx):
        if isinstance(img, torch.Tensor):
            if img.dim() != 3:
                raise ValueError(
                    f"Expected 3D image tensor for view {view_idx}, got shape={tuple(img.shape)}"
                )
            if int(img.shape[-1]) == 3:
                chw = img.permute(2, 0, 1).contiguous()
            elif int(img.shape[0]) == 3:
                chw = img.contiguous()
            else:
                raise ValueError(
                    f"Expected HWC/CHW 3-channel image tensor for view {view_idx}, "
                    f"got shape={tuple(img.shape)}"
                )

            if chw.dtype == torch.uint8:
                return chw.to(dtype=torch.float32).div_(255.0)

            chw = chw.to(dtype=torch.float32)
            max_value = float(chw.detach().max().item()) if chw.numel() else 0.0
            if max_value <= 1.0:
                return (chw * 255.0).clamp_(0.0, 255.0).to(torch.uint8).to(torch.float32).div_(255.0)
            return chw.clamp_(0.0, 255.0).to(torch.uint8).to(torch.float32).div_(255.0)

        if isinstance(img, np.ndarray):
            if img.ndim != 3 or int(img.shape[-1]) != 3:
                raise ValueError(
                    f"Expected HWC 3-channel ndarray for view {view_idx}, got shape={img.shape}"
                )
            arr = img
            if arr.dtype == np.uint8:
                arr01 = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32, copy=False)
                if arr.size and float(np.max(arr)) <= 1.0:
                    arr01 = ((arr * 255.0).clip(0, 255).astype(np.uint8)).astype(np.float32) / 255.0
                else:
                    arr01 = (arr.clip(0, 255).astype(np.uint8)).astype(np.float32) / 255.0
            return torch.from_numpy(arr01).permute(2, 0, 1).contiguous()

        raise TypeError(f"Unsupported image type for view {view_idx}: {type(img)}")

    @staticmethod
    def _normalize_image(chw_float01, norm_type):
        try:
            mean, std = _IMAGE_NORMALIZATIONS[str(norm_type)]
        except KeyError as exc:
            raise ValueError(
                f'Unsupported image normalization "{norm_type}". '
                f"Supported: {sorted(_IMAGE_NORMALIZATIONS)}"
            ) from exc

        mean = chw_float01.new_tensor(mean).view(3, 1, 1)
        std = chw_float01.new_tensor(std).view(3, 1, 1)
        return ((chw_float01 - mean) / std).unsqueeze(0).contiguous()

    def _preprocess_inputs(self, input_views, resize_mode="fixed_size", size=None,
                           norm_type="radio", **kwargs):
        if kwargs:
            # Keep old config compatibility without silently accepting typos.
            ignored = {"patch_size", "resolution_set", "verbose"}
            unexpected = sorted(set(kwargs) - ignored)
            if unexpected:
                raise TypeError(f"Unexpected preprocess_cfg keys: {unexpected}")

        resize_mode = self._normalize_resize_mode(resize_mode)
        if resize_mode != "fixed_size":
            raise ValueError(
                "AdaOcc local preprocessing only supports fixed_size after "
                f"patch alignment, got resize_mode={resize_mode!r}"
            )

        if not input_views:
            raise ValueError("input_views cannot be empty")

        processed_views = []
        for view_idx, view in enumerate(input_views):
            if "img" not in view:
                raise KeyError(f'View {view_idx} missing required "img" key')

            chw = self._to_chw_float01_image(view["img"], view_idx=view_idx)
            if size is not None:
                if not isinstance(size, (tuple, list)) or len(size) != 2:
                    raise ValueError(f"fixed_size expects size=(width, height), got {size}")
                expected_hw = (int(size[1]), int(size[0]))
                actual_hw = tuple(int(v) for v in chw.shape[-2:])
                if actual_hw != expected_hw:
                    raise ValueError(
                        f"View {view_idx} image size mismatch after patch alignment: "
                        f"expected H,W={expected_hw}, got {actual_hw}"
                    )

            processed_view = {
                "img": self._normalize_image(chw, norm_type=norm_type),
                "data_norm_type": [norm_type],
            }

            # Preserve non-geometry metadata for debugging.  Geometry keys are
            # intentionally omitted because the RADIO image backbone does not
            # consume depth/rays/poses in the promoted baseline.
            for key, value in view.items():
                if key not in {"img", "depth_z", "intrinsics", "ray_directions", "camera_poses"}:
                    processed_view[key] = copy.deepcopy(value)
            processed_views.append(processed_view)
        return processed_views

    def _prepare_views_for_preprocess(self, sample_views):
        preprocess_cfg = dict(self.preprocess_cfg)
        resize_mode = self._normalize_resize_mode(preprocess_cfg.get("resize_mode", None))
        if resize_mode != "patch_aligned_pad_bottom_right":
            return sample_views, preprocess_cfg

        patch_size = int(preprocess_cfg.get("patch_size", 14))
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        heights = []
        widths = []
        for view_idx, view in enumerate(sample_views):
            if "img" not in view:
                raise KeyError(f'View {view_idx} missing required "img" key')
            height, width = self._image_hw(view["img"])
            heights.append(height)
            widths.append(width)

        target_h, target_w = self._resolve_patch_aligned_hw(
            max(heights), max(widths), patch_size=patch_size
        )
        if all(height == target_h and width == target_w for height, width in zip(heights, widths)):
            preprocess_cfg["resize_mode"] = "fixed_size"
            preprocess_cfg["size"] = (target_w, target_h)
            return sample_views, preprocess_cfg

        prepared_views = []
        for view_idx, view in enumerate(sample_views):
            prepared_view = copy.deepcopy(view)
            prepared_view["img"] = self._pad_hw_bottom_right(
                prepared_view["img"], target_h=target_h, target_w=target_w, fill_value=0.0
            )
            if "depth_z" in prepared_view:
                prepared_view["depth_z"] = self._pad_hw_bottom_right(
                    prepared_view["depth_z"],
                    target_h=target_h,
                    target_w=target_w,
                    fill_value=0.0,
                )
            if "ray_directions" in prepared_view and "intrinsics" not in prepared_view:
                raise ValueError(
                    "patch_aligned_pad_bottom_right requires intrinsics-based inputs; "
                    f'view {view_idx} provided ray_directions without intrinsics'
                )
            prepared_views.append(prepared_view)

        preprocess_cfg["resize_mode"] = "fixed_size"
        preprocess_cfg["size"] = (target_w, target_h)
        return prepared_views, preprocess_cfg

    def build_runtime_img_metas(self, img, img_metas, runtime_num_views=None):
        resize_mode = self._normalize_resize_mode(self.preprocess_cfg.get("resize_mode", None))
        if resize_mode != "patch_aligned_pad_bottom_right":
            return None
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            return None
        if img_metas is None:
            return None

        batch_size, total_views, _, height, width = img.shape
        target_h, target_w = self._resolve_patch_aligned_hw(
            int(height),
            int(width),
            patch_size=int(self.preprocess_cfg.get("patch_size", 14)),
        )
        if target_h == int(height) and target_w == int(width):
            return copy.deepcopy(img_metas)

        runtime_img_shape = (target_h, target_w, int(img.shape[2]))
        runtime_input_shape = (target_h, target_w)
        adapted = []
        for meta in img_metas:
            if not isinstance(meta, dict):
                adapted.append(meta)
                continue

            out = copy.deepcopy(meta)
            for key in ("img_shape", "ori_shape", "pad_shape"):
                if key in out:
                    out[key] = [runtime_img_shape for _ in range(total_views)]
            if "input_shape" in out:
                out["input_shape"] = runtime_input_shape
            adapted.append(out)

        if len(adapted) != batch_size:
            raise ValueError(
                f"img_metas length mismatch for runtime adaptation: {len(adapted)} vs {batch_size}"
            )
        return adapted

    @staticmethod
    def _ensure_view_tensor(view, key, expected_shape_suffix, sample_idx, view_idx):
        value = view.get(key, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f'sample={sample_idx} view={view_idx} missing tensor key "{key}", '
                f"got type={type(value)}"
            )
        if tuple(value.shape) != tuple(expected_shape_suffix):
            raise ValueError(
                f'sample={sample_idx} view={view_idx} key="{key}" shape mismatch: '
                f"expected={tuple(expected_shape_suffix)}, got={tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f'sample={sample_idx} view={view_idx} key="{key}" contains non-finite values'
            )
        return value

    def _validate_preprocessed_views_for_geometry(self, views_before, views_after, sample_idx):
        if len(views_before) != len(views_after):
            raise ValueError(
                f"sample={sample_idx}: processed view count mismatch: "
                f"{len(views_before)} vs {len(views_after)}"
            )

        for view_idx, (before_view, after_view) in enumerate(zip(views_before, views_after)):
            expects_ray = ("intrinsics" in before_view) or ("ray_directions" in before_view)
            expects_depth = "depth_z" in before_view
            expects_pose = "camera_poses" in before_view

            image = after_view.get("img", None)
            if not isinstance(image, torch.Tensor) or image.dim() != 4 or image.shape[0] != 1:
                raise ValueError(
                    f"sample={sample_idx} view={view_idx}: processed image must be [1,3,H,W], "
                    f'got type={type(image)} shape={getattr(image, "shape", None)}'
                )
            height, width = int(image.shape[-2]), int(image.shape[-1])

            if expects_ray or expects_depth:
                self._ensure_view_tensor(
                    after_view,
                    key="ray_directions_cam",
                    expected_shape_suffix=(1, height, width, 3),
                    sample_idx=sample_idx,
                    view_idx=view_idx,
                )

            if expects_depth:
                self._ensure_view_tensor(
                    after_view,
                    key="depth_along_ray",
                    expected_shape_suffix=(1, height, width, 1),
                    sample_idx=sample_idx,
                    view_idx=view_idx,
                )

            if expects_pose:
                self._ensure_view_tensor(
                    after_view,
                    key="camera_pose_quats",
                    expected_shape_suffix=(1, 4),
                    sample_idx=sample_idx,
                    view_idx=view_idx,
                )
                self._ensure_view_tensor(
                    after_view,
                    key="camera_pose_trans",
                    expected_shape_suffix=(1, 3),
                    sample_idx=sample_idx,
                    view_idx=view_idx,
                )

    def _normalize_is_metric_scale(self, processed_views, sample_idx):
        for view_idx, view in enumerate(processed_views):
            if "is_metric_scale" not in view:
                continue
            value = view["is_metric_scale"]
            image = view["img"]
            target_device = image.device
            if isinstance(value, torch.Tensor):
                metric = value.to(device=target_device, dtype=torch.bool)
            elif isinstance(value, np.ndarray):
                metric = torch.as_tensor(value, device=target_device, dtype=torch.bool)
            elif isinstance(value, (list, tuple)):
                metric = torch.as_tensor(value, device=target_device, dtype=torch.bool)
            elif isinstance(value, (bool, np.bool_)):
                metric = torch.tensor([bool(value)], device=target_device, dtype=torch.bool)
            else:
                raise TypeError(
                    f"sample={sample_idx} view={view_idx}: unsupported is_metric_scale type {type(value)}"
                )

            metric = metric.reshape(-1)
            if metric.numel() != 1:
                raise ValueError(
                    f"sample={sample_idx} view={view_idx}: is_metric_scale must contain one value, "
                    f"got shape={tuple(metric.shape)}"
                )
            view["is_metric_scale"] = metric

    @staticmethod
    def _move_processed_views_to_device(processed_views, target_device):
        ignore_keys = {"instance", "idx", "true_shape", "data_norm_type", "filename"}

        def _move(value):
            if isinstance(value, dict):
                return {k: _move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_move(v) for v in value)
            if hasattr(value, "to"):
                return value.to(target_device, non_blocking=True)
            return value

        moved = []
        for view in processed_views:
            out = {}
            for key, value in view.items():
                out[key] = value if key in ignore_keys else _move(value)
            moved.append(out)
        return moved

    def forward(self, img, points=None, img_metas=None, view_extra=None):
        if not isinstance(img, torch.Tensor) or img.dim() != 5:
            raise ValueError(
                f"img must be Tensor[B, TN, C, H, W], got type={type(img)} "
                f'shape={getattr(img, "shape", None)}'
            )

        batch_size = int(img.shape[0])
        if img_metas is None:
            img_metas = [{} for _ in range(batch_size)]
        if not isinstance(img_metas, list) or len(img_metas) != batch_size:
            raise ValueError(
                f"img_metas must be list of length B={batch_size}, got {type(img_metas)}"
            )

        batch_views = self.input_adapter(
            img=img,
            points=points,
            img_metas=img_metas,
            view_extra=view_extra,
        )

        processed_batch = []
        for sample_idx, sample_views in enumerate(batch_views):
            sample_views, preprocess_cfg = self._prepare_views_for_preprocess(sample_views)
            processed_views = self._preprocess_inputs(sample_views, **preprocess_cfg)
            views_before_inference = [dict(view) for view in processed_views]
            self._normalize_is_metric_scale(processed_views, sample_idx=sample_idx)
            self._validate_preprocessed_views_for_geometry(
                views_before_inference,
                processed_views,
                sample_idx=sample_idx,
            )
            processed_views = self._move_processed_views_to_device(
                processed_views, target_device=img.device
            )
            processed_batch.append(processed_views)
        return processed_batch
