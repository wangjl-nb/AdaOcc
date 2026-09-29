#!/usr/bin/env python3
"""Check that a released AdaOcc checkpoint loads into a public config.

The released full-split checkpoint was trained before the public code cleanup,
so this script builds the model described by --config and compares it against
the checkpoint state dict. It reports missing, unexpected, and shape-mismatched
parameters, and exits non-zero when the checkpoint cannot be loaded directly.

Example:

  python scripts/check_checkpoint.py \
    --config configs/occscannet/radio_occscannet_full.py \
    --checkpoint checkpoints/adaocc_radio_occscannet_full_epoch100.pth

Building the model loads the local RADIO weights under
`pretrain/radio/C-RADIOv3-B/` (and the Depth-Anything checkpoint when the
config enables online depth).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_model(config_path: Path):
    import torch  # noqa: F401  (imported here so --help works without torch)
    from mmengine.config import Config
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    importlib.import_module("models")
    importlib.import_module("loaders")

    cfg = Config.fromfile(str(config_path))
    return MODELS.build(cfg.model)


def load_state_dict(checkpoint_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not an MMEngine checkpoint (missing 'state_dict')"
        )
    meta = checkpoint.get("meta", {}) or {}
    return checkpoint["state_dict"], meta


def compare(model_state, ckpt_state):
    missing = sorted(key for key in model_state if key not in ckpt_state)
    unexpected = sorted(key for key in ckpt_state if key not in model_state)
    shape_mismatch: List[Tuple[str, tuple, tuple]] = []
    for key in model_state:
        if key in ckpt_state and tuple(model_state[key].shape) != tuple(ckpt_state[key].shape):
            shape_mismatch.append(
                (key, tuple(ckpt_state[key].shape), tuple(model_state[key].shape))
            )
    return missing, unexpected, shape_mismatch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Public AdaOcc config the checkpoint should load into")
    parser.add_argument("--checkpoint", required=True, help="Released or user-provided AdaOcc checkpoint")
    parser.add_argument("--strict-unexpected", action="store_true", help="Fail when the checkpoint has extra parameters")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"config not found: {config_path}")
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    model = build_model(config_path)
    ckpt_state, meta = load_state_dict(checkpoint_path)
    model_state = model.state_dict()
    missing, unexpected, shape_mismatch = compare(model_state, ckpt_state)

    ok = not missing and not shape_mismatch and (not args.strict_unexpected or not unexpected)
    report = {
        "ok": ok,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "epoch": meta.get("epoch"),
        "model_tensors": len(model_state),
        "checkpoint_tensors": len(ckpt_state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": [
            {"key": key, "checkpoint": list(ckpt), "model": list(model_shape)}
            for key, ckpt, model_shape in shape_mismatch
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"config: {config_path}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"epoch: {meta.get('epoch')}")
        print(f"model tensors: {len(model_state)}  checkpoint tensors: {len(ckpt_state)}")
        print(f"missing: {len(missing)}  unexpected: {len(unexpected)}  shape mismatch: {len(shape_mismatch)}")
        for key in missing[:10]:
            print(f"  [missing] {key}")
        for key in unexpected[:10]:
            print(f"  [unexpected] {key}")
        for key, ckpt, model_shape in shape_mismatch[:10]:
            print(f"  [shape] {key}: checkpoint {ckpt} vs model {model_shape}")
        print("OK: checkpoint loads into this config" if ok else "FAILED: checkpoint is not compatible")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
