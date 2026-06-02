#!/usr/bin/env python3
"""Generate AdaOcc OccScanNet-mini ``gts_camvisbits`` labels from gathered_data.

The public mini annotation PKLs are only indexes.  OccScanNet semantic labels
live in ``gathered_data/<scene>/<frame>.pkl`` as ``target_1_4``.  AdaOcc's
OccScanNet config expects per-sample NPZ files at

    gts_camvisbits/{token}/labels.npz

where ``token`` is already ``<scene>/<frame>``.

This script derives the required NPZ keys without changing model code:

* ``raw_semantics``: OccScanNet raw 0/1..11/255 labels at 0.08 m, stored in
  the native ``target_1_4`` orientation.  AdaOcc's raw branch indexes this as
  [y, x, z] and converts indices back to xyz when needed.
* ``semantics``: dense AdaOcc ego/occ grid labels [x, y, z] with classes
  0..10 and ``empty_label`` for empty/unknown.
* ``mask_camera``/``mask_lidar``/``mask_camera_bits``: known-voxel masks
  projected into the dense AdaOcc grid.
* ``voxel_origin``/``voxel_size``: raw OccScanNet grid geometry.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_SPLITS = [
    "train_occscannet_mini.pkl",
    "val_occscannet_mini.pkl",
    "test_occscannet_mini.pkl",
]
DEFAULT_PC_RANGE = [-3.20, -4.80, -5.60, 7.20, 4.80, 5.60]
DEFAULT_VOXEL_SIZE = [0.08, 0.08, 0.08]
DEFAULT_EMPTY_LABEL = 11


def install_numpy_pickle_compat_aliases() -> None:
    """Allow pickles produced by newer NumPy versions to load in older NumPy."""

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_pickle(path: Path):
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def annotation_infos(data_root: Path, split_names: Iterable[str]) -> Dict[str, dict]:
    """Collect unique mini sample infos keyed by token."""

    records: Dict[str, dict] = {}
    for split_name in split_names:
        split_path = Path(split_name).expanduser()
        if not split_path.is_absolute():
            split_path = data_root / split_path
        obj = load_pickle(split_path)
        infos = obj.get("infos", obj.get("data_list", obj)) if isinstance(obj, dict) else obj
        for info in infos:
            token = info.get("token") or info.get("sample_token")
            if not token:
                raise KeyError(f"annotation info in {split_path} lacks token/sample_token")
            records.setdefault(str(token), info)
    return records


def dense_shape_from_range(pc_range: np.ndarray, voxel_size: np.ndarray) -> Tuple[int, int, int]:
    shape_f = (pc_range[3:] - pc_range[:3]) / voxel_size
    shape = np.rint(shape_f).astype(np.int64)
    if not np.allclose(shape_f, shape, atol=1e-5):
        raise ValueError(f"pc_range is not divisible by voxel_size: {shape_f}")
    return tuple(int(v) for v in shape)


def world_to_ego(points_world: np.ndarray, ego2global_rotation, ego2global_translation) -> np.ndarray:
    """Transform row-vector world points into AdaOcc ego/occ coordinates."""

    rot_ego_to_global = np.asarray(ego2global_rotation, dtype=np.float32).reshape(3, 3)
    trans_ego_to_global = np.asarray(ego2global_translation, dtype=np.float32).reshape(1, 3)
    # Column convention: p_global = R @ p_ego + t.
    # Use an explicit solve instead of assuming R is exactly orthonormal; this
    # matches the reference OccScanNet-mini label projection on boundary voxels.
    return np.linalg.solve(
        rot_ego_to_global,
        (points_world.astype(np.float32) - trans_ego_to_global).T,
    ).T.astype(np.float32, copy=False)


def derive_label_arrays(
    info: dict,
    gathered: dict,
    *,
    pc_range: np.ndarray,
    dense_voxel_size: np.ndarray,
    dense_shape: Tuple[int, int, int],
    empty_label: int,
) -> dict:
    """Derive one AdaOcc labels.npz payload from one gathered_data pickle."""

    target_xyz = np.asarray(gathered["target_1_4"], dtype=np.uint8)
    if target_xyz.shape != (60, 60, 36):
        raise ValueError(f"target_1_4 must be (60, 60, 36), got {target_xyz.shape}")

    # Keep OccScanNet's native target_1_4 axis order.  The AdaOcc raw branch
    # treats raw_semantics indices as [y, x, z] and performs coords[:, [1, 0, 2]]
    # internally before mapping voxels into xyz/world space.  Transposing here
    # would swap x/y twice and place raw labels in the wrong OccScanNet cells.
    raw_semantics = np.ascontiguousarray(target_xyz)
    raw_voxel_origin = np.asarray(gathered["voxel_origin"], dtype=np.float32).reshape(3)
    raw_voxel_size = dense_voxel_size.astype(np.float32).reshape(3)

    semantics = np.full(dense_shape, int(empty_label), dtype=np.uint8)
    known_mask = np.zeros(dense_shape, dtype=np.bool_)

    known_yxz = np.argwhere(raw_semantics != 255)
    if known_yxz.size:
        known_xyz = known_yxz[:, [1, 0, 2]].astype(np.float32)
        known_world = raw_voxel_origin[None, :] + known_xyz * raw_voxel_size[None, :]
        known_ego = world_to_ego(
            known_world,
            info["ego2global_rotation"],
            info["ego2global_translation"],
        )
        dense_idx = np.floor((known_ego - pc_range[:3][None, :]) / dense_voxel_size[None, :]).astype(np.int64)
        inside = np.all((dense_idx >= 0) & (dense_idx < np.asarray(dense_shape)[None, :]), axis=1)
        if np.any(inside):
            idx_inside = dense_idx[inside]
            lin_inside = np.ravel_multi_index(idx_inside.T, dense_shape)
            known_mask.reshape(-1)[lin_inside] = True

            labels_inside = raw_semantics[tuple(known_yxz[inside].T)]
            occ = (labels_inside >= 1) & (labels_inside <= 11)
            if np.any(occ):
                # Multiple raw voxels can project into the same dense AdaOcc
                # voxel.  The reference labels resolve those rare collisions by
                # keeping the smallest semantic id.
                np.minimum.at(
                    semantics.reshape(-1),
                    lin_inside[occ],
                    (labels_inside[occ] - 1).astype(np.uint8),
                )

    mask_camera = known_mask.astype(np.uint8)
    mask_lidar = known_mask.astype(np.uint8)
    mask_camera_bits = mask_camera.astype(np.uint8)  # CAM_FRONT is bit 0.

    return {
        "semantics": np.ascontiguousarray(semantics),
        "mask_lidar": np.ascontiguousarray(mask_lidar),
        "mask_camera": np.ascontiguousarray(mask_camera),
        "mask_camera_bits": np.ascontiguousarray(mask_camera_bits),
        "camera_names": np.asarray(["CAM_FRONT"]),
        "raw_semantics": raw_semantics,
        "voxel_origin": np.ascontiguousarray(raw_voxel_origin),
        "voxel_size": np.ascontiguousarray(raw_voxel_size),
    }


def output_path(occ_root: Path, token: str) -> Path:
    return occ_root / token / "labels.npz"


def verify_one(path: Path, dense_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path) as label:
            required = ["semantics", "mask_camera", "mask_lidar", "raw_semantics", "voxel_origin", "voxel_size"]
            missing = [key for key in required if key not in label.files]
            if missing:
                return False, f"missing_keys={missing}"
            if tuple(label["semantics"].shape) != tuple(dense_shape):
                return False, f"bad_semantics_shape={label['semantics'].shape}"
            if tuple(label["mask_camera"].shape) != tuple(dense_shape):
                return False, f"bad_mask_camera_shape={label['mask_camera'].shape}"
            if tuple(label["mask_lidar"].shape) != tuple(dense_shape):
                return False, f"bad_mask_lidar_shape={label['mask_lidar'].shape}"
            if tuple(label["raw_semantics"].shape) != (60, 60, 36):
                return False, f"bad_raw_shape={label['raw_semantics'].shape}"
            if label["voxel_origin"].reshape(-1).shape[0] != 3:
                return False, f"bad_origin_shape={label['voxel_origin'].shape}"
            if label["voxel_size"].reshape(-1).shape[0] != 3:
                return False, f"bad_voxel_size_shape={label['voxel_size'].shape}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"load_error={exc}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/OccScanNet")
    parser.add_argument("--occ-root", default=None, help="Default: <data-root>/gts_camvisbits")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--pc-range", nargs=6, type=float, default=DEFAULT_PC_RANGE)
    parser.add_argument("--voxel-size", nargs=3, type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--empty-label", type=int, default=DEFAULT_EMPTY_LABEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    occ_root = Path(args.occ_root).expanduser().resolve() if args.occ_root else data_root / "gts_camvisbits"
    pc_range = np.asarray(args.pc_range, dtype=np.float32)
    voxel_size = np.asarray(args.voxel_size, dtype=np.float32)
    dense_shape = dense_shape_from_range(pc_range, voxel_size)

    records = annotation_infos(data_root, args.splits)
    tokens: List[str] = sorted(records.keys())
    if args.limit > 0:
        tokens = tokens[: args.limit]

    print(
        f"records={len(records)} selected={len(tokens)} data_root={data_root} "
        f"occ_root={occ_root} dense_shape={dense_shape}",
        flush=True,
    )

    if args.verify_only:
        bad = []
        for i, token in enumerate(tokens, 1):
            ok, reason = verify_one(output_path(occ_root, token), dense_shape)
            if not ok:
                bad.append((token, reason))
            if i == 1 or i % 500 == 0 or i == len(tokens):
                print(f"verify {i}/{len(tokens)} bad={len(bad)}", flush=True)
        if bad:
            print("first_bad", bad[:20], flush=True)
            return 1
        print(f"verify ok={len(tokens)}", flush=True)
        return 0

    generated = 0
    skipped = 0
    failed = 0
    for i, token in enumerate(tokens, 1):
        info = records[token]
        out = output_path(occ_root, token)
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        lidar_path = data_root / info["lidar_path"]
        try:
            gathered = load_pickle(lidar_path)
            arrays = derive_label_arrays(
                info,
                gathered,
                pc_range=pc_range,
                dense_voxel_size=voxel_size,
                dense_shape=dense_shape,
                empty_label=args.empty_label,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **arrays)
            generated += 1
        except Exception as exc:
            failed += 1
            print(f"[error] token={token} lidar_path={lidar_path}: {exc}", flush=True)
        if i == 1 or i % 250 == 0 or i == len(tokens):
            print(f"progress {i}/{len(tokens)} generated={generated} skipped={skipped} failed={failed}", flush=True)

    print(f"finished selected={len(tokens)} generated={generated} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
