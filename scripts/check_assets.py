#!/usr/bin/env python3
"""Check AdaOcc release assets without modifying data.

The checker validates the public reproduction layout and reports exact missing
paths. It supports two depth modes:

* --online-depth: DepthAnything checkpoint is required, precomputed PNGs are not.
* --precomputed-depth: pkl-referenced depth PNGs are required and decoded.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

DEFAULT_SPLITS = [
    "train_occscannet_mini.pkl",
    "val_occscannet_mini.pkl",
    "test_occscannet_mini.pkl",
]
REQUIRED_LABEL_KEYS = [
    "semantics",
    "mask_lidar",
    "mask_camera",
    "raw_semantics",
    "voxel_origin",
    "voxel_size",
]


def install_numpy_pickle_compat_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def infos_from_pkl(path: Path) -> List[dict]:
    obj = load_pickle(path)
    if isinstance(obj, dict):
        infos = obj.get("infos", obj.get("data_list", []))
    else:
        infos = obj
    if not isinstance(infos, list):
        raise TypeError(f"{path} does not contain a list of infos")
    return infos


def add_missing(missing: List[dict], kind: str, path: Path, detail: str = "") -> None:
    missing.append({"kind": kind, "path": str(path), "detail": detail})


def read_float_depth_png(path: Path):
    """Lightweight AdaOcc depth PNG validation.

    Training uses cv2.imread in the data loader. The release asset checker uses
    PIL here to avoid stressing OpenCV in environments where repeated PNG decode
    can crash the interpreter. This checks the file is a readable 4-channel PNG
    whose raw bytes can be viewed as little-endian float32 depth values.
    """
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(f"Pillow is required to verify depth PNGs: {exc}") from exc
    try:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        return None
    depth = rgba.view("<f4").squeeze()
    if depth.ndim != 2:
        return None
    return depth


def check_label(path: Path) -> Tuple[bool, str]:
    try:
        with np.load(path) as label:
            missing = [key for key in REQUIRED_LABEL_KEYS if key not in label.files]
            if missing:
                return False, f"missing_keys={missing}"
            if tuple(label["raw_semantics"].shape) != (60, 60, 36):
                return False, f"bad_raw_semantics_shape={label['raw_semantics'].shape}"
            if label["voxel_origin"].reshape(-1).shape[0] != 3:
                return False, f"bad_voxel_origin_shape={label['voxel_origin'].shape}"
            if label["voxel_size"].reshape(-1).shape[0] != 3:
                return False, f"bad_voxel_size_shape={label['voxel_size'].shape}"
    except Exception as exc:
        return False, f"label_load_error={exc}"
    return True, "ok"


def check_manifest(
    data_root: Path,
    split_paths: Iterable[Path],
    *,
    require_precomputed_depth: bool,
    verify_depth_png: bool,
    max_missing_report: int,
    max_depth_checks: int,
) -> Tuple[Dict[str, int], List[dict], Dict[str, dict]]:
    counts = {"splits": 0, "infos": 0, "labels_checked": 0, "depth_paths_checked": 0, "depth_decoded": 0}
    missing: List[dict] = []
    samples: Dict[str, dict] = {}
    for split_path in split_paths:
        if not split_path.exists():
            add_missing(missing, "pkl", split_path)
            continue
        counts["splits"] += 1
        try:
            infos = infos_from_pkl(split_path)
        except Exception as exc:
            add_missing(missing, "pkl_load", split_path, str(exc))
            continue
        for info in infos:
            counts["infos"] += 1
            token = str(info.get("token") or info.get("sample_token") or "")
            if not token:
                add_missing(missing, "token", split_path, "record lacks token/sample_token")
                continue
            scene = str(info.get("scene_name") or token.split("/")[0])
            cam = (info.get("cams") or {}).get("CAM_FRONT", {})
            lidar_path = resolve_path(data_root, info.get("lidar_path", ""))
            image_path = resolve_path(data_root, cam.get("data_path", ""))
            label_path = data_root / "gts_camvisbits" / token / "labels.npz"
            samples.setdefault("first", {
                "token": token,
                "lidar": str(lidar_path),
                "image": str(image_path),
                "label": str(label_path),
                "scene": scene,
            })
            if not lidar_path.exists():
                add_missing(missing, "lidar", lidar_path, token)
            if not image_path.exists():
                add_missing(missing, "image", image_path, token)
            if not label_path.exists():
                add_missing(missing, "label", label_path, token)
            else:
                counts["labels_checked"] += 1
                ok, reason = check_label(label_path)
                if not ok:
                    add_missing(missing, "label_keys", label_path, reason)
            depth_rel = cam.get("depth_path")
            if require_precomputed_depth:
                depth_path = resolve_path(data_root, depth_rel or "")
                samples.setdefault("first", {}).setdefault("depth", str(depth_path))
                if not depth_path.exists():
                    add_missing(missing, "depth", depth_path, token)
                else:
                    counts["depth_paths_checked"] += 1
                    should_decode = verify_depth_png and (max_depth_checks <= 0 or counts["depth_decoded"] < max_depth_checks)
                    if should_decode:
                        try:
                            depth = read_float_depth_png(depth_path)
                        except Exception as exc:
                            add_missing(missing, "depth_decode", depth_path, str(exc))
                        else:
                            counts["depth_decoded"] += 1
                            if depth is None or not np.isfinite(depth).any():
                                add_missing(missing, "depth_decode", depth_path, "not AdaOcc float32 RGBA PNG")
            if len(missing) > max_missing_report > 0:
                # Continue counts are less useful than responsive diagnostics once
                # a large manifest is clearly missing assets.
                return counts, missing, samples
    return counts, missing, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/OccScanNet")
    parser.add_argument("--pretrain-root", default="pretrain")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--online-depth", action="store_true", help="Require online DepthAnything checkpoint")
    parser.add_argument("--precomputed-depth", action="store_true", help="Require pkl-referenced precomputed depth PNGs")
    parser.add_argument("--verify-depth-png", action="store_true", help="Decode precomputed depth PNGs as float32 RGBA")
    parser.add_argument("--max-missing-report", type=int, default=50)
    parser.add_argument("--max-depth-checks", type=int, default=16, help="Maximum existing depth PNGs to decode when --verify-depth-png is set; 0 means decode all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    pretrain_root = Path(args.pretrain_root).expanduser().resolve()
    missing: List[dict] = []

    if not data_root.exists():
        add_missing(missing, "data_root", data_root)
    else:
        for dirname in ("gathered_data", "posed_images", "gts_camvisbits"):
            path = data_root / dirname
            if not path.exists():
                add_missing(missing, "data_dir", path)

    if not (pretrain_root / "fusion_pretrain_model.pth").exists():
        add_missing(missing, "pretrain", pretrain_root / "fusion_pretrain_model.pth")
    if args.online_depth and not (pretrain_root / "depth_anything" / "finetune_scannet_depthanythingv2.pth").exists():
        add_missing(missing, "depth_anything_ckpt", pretrain_root / "depth_anything" / "finetune_scannet_depthanythingv2.pth")

    local_radio = pretrain_root / "radio" / "C-RADIOv3-B"
    if not local_radio.exists():
        add_missing(missing, "radio", local_radio)

    split_paths = [resolve_path(data_root, split) for split in args.splits]
    counts, manifest_missing, samples = check_manifest(
        data_root,
        split_paths,
        require_precomputed_depth=bool(args.precomputed_depth),
        verify_depth_png=bool(args.verify_depth_png),
        max_missing_report=max(int(args.max_missing_report), 0),
        max_depth_checks=max(int(args.max_depth_checks), 0),
    ) if data_root.exists() else ({"splits": 0, "infos": 0, "labels_checked": 0, "depth_paths_checked": 0, "depth_decoded": 0}, [], {})
    missing.extend(manifest_missing)

    if args.max_missing_report > 0 and len(missing) > args.max_missing_report:
        missing = missing[: args.max_missing_report]

    payload = {
        "ok": len(missing) == 0,
        "data_root": str(data_root),
        "pretrain_root": str(pretrain_root),
        "mode": {
            "online_depth": bool(args.online_depth),
            "precomputed_depth": bool(args.precomputed_depth),
        },
        "counts": counts,
        "samples": samples,
        "missing": missing,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ok={payload['ok']} data_root={data_root} pretrain_root={pretrain_root}")
        print(f"counts={counts}")
        if missing:
            print("missing:")
            for item in missing:
                print(f"- {item['kind']}: {item['path']} {item.get('detail','')}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
