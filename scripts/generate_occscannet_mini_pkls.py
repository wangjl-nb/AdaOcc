#!/usr/bin/env python3
"""Generate AdaOcc OccScanNet-mini annotation PKLs from OccScanNet prepared data.

Input layout expected under --data-root:
  gathered_data/<scene>/<frame>.pkl
  posed_images/<scene>/<frame>.jpg
  train_subscenes.txt
  val_subscenes.txt

The gathered_data pickle stores camera pose/intrinsics and raw occupancy labels.
This script builds the lightweight AdaOcc index PKLs consumed by
``configs/adaocc/radio_occscannet_mini.py``. It does not generate labels.npz;
use ``generate_occscannet_mini_gts_camvisbits.py`` for that.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

DEFAULT_TRAIN = "train_occscannet_mini.pkl"
DEFAULT_VAL = "val_occscannet_mini.pkl"
DEFAULT_TEST = "test_occscannet_mini.pkl"
DEFAULT_TRAIN_MINI_COUNT = 4639
DEFAULT_VAL_MINI_COUNT = 2007
LIDAR2EGO_R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
LIDAR2EGO_T = np.zeros(3, dtype=np.float64)
SENSOR2LIDAR_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
SENSOR2LIDAR_T = np.zeros(3, dtype=np.float64)


def install_numpy_pickle_compat_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_pickle(path: Path):
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def read_split(path: Path) -> List[str]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        records.append(line)
    return records


def rel_from_gathered_path(path_text: str) -> tuple[str, str, str]:
    rel = Path(path_text)
    parts = rel.parts
    if len(parts) < 3 or parts[-3] != "gathered_data":
        raise ValueError(f"split entry must look like gathered_data/<scene>/<frame>.pkl, got {path_text}")
    scene = parts[-2]
    frame = Path(parts[-1]).stem
    token = f"{scene}/{frame}"
    return scene, frame, token


def make_info(data_root: Path, gathered_rel: str, depth_dir: str) -> dict:
    scene, frame, token = rel_from_gathered_path(gathered_rel)
    gathered_path = data_root / gathered_rel
    gathered = load_pickle(gathered_path)
    cam_pose = np.asarray(gathered["cam_pose"], dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(gathered["intrinsic"], dtype=np.float64)
    if intrinsic.shape == (3, 3):
        intrinsic4 = np.eye(4, dtype=np.float64)
        intrinsic4[:3, :3] = intrinsic
    else:
        intrinsic4 = intrinsic.reshape(4, 4)
    sensor2global_r = cam_pose[:3, :3]
    sensor2global_t = cam_pose[:3, 3]
    # The historical AdaOcc mini PKLs use a lidar/ego convention where
    # sensor2global = ego2global @ lidar2ego @ sensor2lidar in column form.
    ego2global_r = sensor2global_r @ SENSOR2LIDAR_R.T @ LIDAR2EGO_R.T
    ego2global_t = sensor2global_t.copy()
    info = {
        "token": token,
        "scene_name": scene,
        "scene_token": scene,
        "timestamp": None,
        "lidar_path": f"gathered_data/{scene}/{frame}.pkl",
        "lidar2ego_rotation": LIDAR2EGO_R.copy(),
        "lidar2ego_translation": LIDAR2EGO_T.copy(),
        "ego2global_rotation": ego2global_r,
        "ego2global_translation": ego2global_t,
        "ego2occ": np.eye(4, dtype=np.float64),
        "cam_sweeps": {"prev": [], "next": []},
        "cams": {
            "CAM_FRONT": {
                "data_path": f"posed_images/{scene}/{frame}.jpg",
                "cam_intrinsic": intrinsic4,
                "sensor2global_rotation": sensor2global_r,
                "sensor2global_translation": sensor2global_t,
                "sensor2lidar_rotation": SENSOR2LIDAR_R.copy(),
                "sensor2lidar_translation": SENSOR2LIDAR_T.copy(),
                "timestamp": None,
                "depth_path": f"{depth_dir}/{scene}/{frame}.png",
            }
        },
    }
    return info


def build_infos(data_root: Path, entries: Iterable[str], depth_dir: str, limit: int = 0) -> List[dict]:
    infos = []
    for i, rel in enumerate(entries):
        if limit > 0 and i >= limit:
            break
        infos.append(make_info(data_root, rel, depth_dir))
    return infos


def write_pkl(path: Path, infos: List[dict], dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"version": "v1.0-trainval", "dataset": dataset, "parity_mode": "native_occscannet"},
        "infos": infos,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/OccScanNet")
    parser.add_argument("--train-split", default="train_subscenes.txt")
    parser.add_argument("--val-split", default="val_subscenes.txt")
    parser.add_argument("--train-output", default=DEFAULT_TRAIN)
    parser.add_argument("--val-output", default=DEFAULT_VAL)
    parser.add_argument("--test-output", default=DEFAULT_TEST)
    parser.add_argument("--depth-dir", default="depth_splatssc_stage1_ftdav2_vitb_20m_full")
    parser.add_argument("--dataset", default="occscannet")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_MINI_COUNT, help="0 means all train split entries")
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_MINI_COUNT, help="0 means all val split entries")
    parser.add_argument("--limit", type=int, default=0, help="Debug override applied to both splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    train_split = Path(args.train_split)
    val_split = Path(args.val_split)
    if not train_split.is_absolute():
        train_split = data_root / train_split
    if not val_split.is_absolute():
        val_split = data_root / val_split
    train_entries = read_split(train_split)
    val_entries = read_split(val_split)
    train_limit = args.limit if args.limit > 0 else args.train_count
    val_limit = args.limit if args.limit > 0 else args.val_count
    train_infos = build_infos(data_root, train_entries, args.depth_dir, limit=train_limit)
    val_infos = build_infos(data_root, val_entries, args.depth_dir, limit=val_limit)
    outputs = [
        (args.train_output, train_infos),
        (args.val_output, val_infos),
        (args.test_output, val_infos),
    ]
    for name, infos in outputs:
        out = Path(name)
        if not out.is_absolute():
            out = data_root / out
        if out.exists() and not args.overwrite:
            print(f"[skip] exists: {out}", flush=True)
            continue
        write_pkl(out, infos, args.dataset)
        print(f"[ok] wrote {out} infos={len(infos)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
