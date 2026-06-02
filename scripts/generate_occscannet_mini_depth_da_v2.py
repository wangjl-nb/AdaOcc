#!/usr/bin/env python3
"""Generate AdaOcc mini depth PNGs with Depth-Anything-V2 metric indoor weights.

The AdaOcc loader reads depth PNGs by viewing RGBA bytes as little-endian
float32. This script writes that exact format: a HxWx4 uint8 PNG whose bytes are
`depth.astype('<f4').view(uint8)`.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def install_depth_anything_module(repo_root: Path) -> None:
    """Expose the vendored Depth-Anything-V2 metric-depth package.

    The public release ships the lightweight Python model code needed by the
    online/precomputed depth paths. Users still provide the checkpoint weights.
    """
    pkg = repo_root / "Depth_Anything_V2" / "metric_depth" / "depth_anything_v2"
    dpt = pkg / "dpt.py"
    if not dpt.exists():
        raise FileNotFoundError(
            f"Depth-Anything-V2 metric-depth code missing: {dpt}. "
            "Restore Depth_Anything_V2/metric_depth before generating depth."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

def load_model(repo_root: Path, weight_path: Path, device: torch.device):
    install_depth_anything_module(repo_root)
    from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    }
    model = DepthAnythingV2(**{**model_configs["vitb"], "max_depth": 20.0})
    state = torch.load(str(weight_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print("[warn] first missing:", missing[:10], flush=True)
        if unexpected:
            print("[warn] first unexpected:", unexpected[:10], flush=True)
    model.to(device).eval()
    return model


def collect_records(data_root: Path, pkl_names):
    records = {}
    for pkl_name in pkl_names:
        p = Path(pkl_name)
        if not p.is_absolute():
            p = data_root / p
        with p.open("rb") as f:
            data = pickle.load(f)
        infos = data.get("infos") if isinstance(data, dict) else data
        for info in infos:
            cam = info["cams"]["CAM_FRONT"]
            token = info.get("token") or f"{info['scene_name']}/{Path(cam['data_path']).stem}"
            image_rel = cam["data_path"]
            depth_rel = cam["depth_path"]
            records[token] = (image_rel, depth_rel)
    return records


def write_float_depth_png(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype="<f4")
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    rgba = depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgba):
        raise IOError(f"cv2.imwrite failed: {path}")


def read_float_depth_png(path: Path):
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return None
    try:
        arr = rgba.view("<f4").squeeze()
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--data-root", default="data/OccScanNet")
    ap.add_argument("--weights", default="pretrain/depth_anything/finetune_scannet_depthanythingv2.pth")
    ap.add_argument("--splits", nargs="+", default=[
        "train_occscannet_mini.pkl",
        "val_occscannet_mini.pkl",
        "test_occscannet_mini.pkl",
    ])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-size", type=int, default=518, help="Depth-Anything-V2 transform size")
    ap.add_argument("--output-height", type=int, default=0, help="0 means original RGB image height")
    ap.add_argument("--output-width", type=int, default=0, help="0 means original RGB image width")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    weight_path = Path(args.weights)
    if not weight_path.is_absolute():
        weight_path = repo_root / weight_path
    if not weight_path.exists():
        raise FileNotFoundError(f"DepthAnything weights missing: {weight_path}")

    records = collect_records(data_root, args.splits)
    tokens = sorted(records.keys())
    if args.limit > 0:
        tokens = tokens[:args.limit]
    print(f"records={len(records)} selected={len(tokens)} data_root={data_root}", flush=True)

    if args.verify_only:
        missing = []
        bad = []
        for token in tokens:
            _, depth_rel = records[token]
            p = data_root / depth_rel
            if not p.exists():
                missing.append(str(p))
                continue
            d = read_float_depth_png(p)
            if d is None or not np.isfinite(d).any():
                bad.append(str(p))
        print(f"verify missing={len(missing)} bad={len(bad)} ok={len(tokens)-len(missing)-len(bad)}", flush=True)
        if missing:
            print("first_missing", missing[:10], flush=True)
        if bad:
            print("first_bad", bad[:10], flush=True)
        return 1 if missing or bad else 0

    device = torch.device(args.device)
    model = load_model(repo_root, weight_path, device)
    done = 0
    skipped = 0
    failed = 0
    with torch.inference_mode():
        for i, token in enumerate(tokens, 1):
            image_rel, depth_rel = records[token]
            image_path = data_root / image_rel
            depth_path = data_root / depth_rel
            if depth_path.exists() and not args.overwrite:
                skipped += 1
                continue
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw is None:
                print(f"[error] missing image token={token} path={image_path}", flush=True)
                failed += 1
                continue
            try:
                image_tensor, (orig_h, orig_w) = model.image2tensor(raw, input_size=args.input_size)
                image_tensor = image_tensor.to(device)
                target_h = args.output_height or orig_h
                target_w = args.output_width or orig_w
                depth = model.infer_image(image_tensor, target_h, target_w, output_feature=False)
                if isinstance(depth, torch.Tensor):
                    depth = depth.detach().cpu().numpy()
                depth = np.asarray(depth, dtype=np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth[depth < 0] = 0
                write_float_depth_png(depth_path, depth)
                done += 1
            except Exception as e:
                print(f"[error] failed token={token}: {e}", flush=True)
                failed += 1
                continue
            if i == 1 or i % 100 == 0 or i == len(tokens):
                print(f"progress {i}/{len(tokens)} generated={done} skipped={skipped} failed={failed}", flush=True)
    print(f"finished selected={len(tokens)} generated={done} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
