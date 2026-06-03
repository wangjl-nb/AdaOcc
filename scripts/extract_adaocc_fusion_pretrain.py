#!/usr/bin/env python3
"""Extract the AdaOcc public-baseline subset from an OPUS fusion pretrain checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


DEFAULT_PREFIXES = ("pts_middle_encoder.",)


def get_state_dict(checkpoint: Mapping):
    """Return a checkpoint state dict from common PyTorch/MMEngine layouts."""
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def filter_state_dict(state_dict: Mapping, prefixes: Sequence[str] = DEFAULT_PREFIXES):
    """Keep only keys whose names start with one of ``prefixes``."""
    prefixes = tuple(prefixes)
    return OrderedDict((k, v) for k, v in state_dict.items() if k.startswith(prefixes))


def convert_spconv_kernels_for_load_hook(state_dict: Mapping):
    """Store 5D sparse-conv kernels in the legacy layout expected by spconv hooks.

    Current model state_dict entries are [out, kx, ky, kz, in], but the
    spconv load hook used by this stack expects checkpoint entries in
    [kx, ky, kz, in, out] and permutes them while loading.
    """
    converted = OrderedDict()
    converted_keys = []
    for key, value in state_dict.items():
        if key.startswith("pts_middle_encoder.") and key.endswith(".weight") and getattr(value, "ndim", None) == 5:
            converted[key] = value.permute(1, 2, 3, 4, 0).contiguous()
            converted_keys.append(key)
        else:
            converted[key] = value
    return converted, converted_keys


def state_dict_nbytes(state_dict: Mapping) -> int:
    total = 0
    for value in state_dict.values():
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            total += int(value.numel()) * int(value.element_size())
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_output_checkpoint(checkpoint: MutableMapping, selected_state: Mapping, source: Path, converted_keys=()):
    return {
        "state_dict": selected_state,
        "meta": {
            "source": str(source),
            "kept_prefixes": list(DEFAULT_PREFIXES),
            "spconv_kernel_layout": "5D kernels stored as legacy [kx, ky, kz, in, out] for the current spconv load hook",
            "converted_spconv_kernel_keys": list(converted_keys),
            "note": "AdaOcc public-baseline middle-encoder subset extracted from OPUS fusion_pretrain_model.pth.",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the AdaOcc public-baseline middle-encoder weights from OPUS "
            "fusion_pretrain_model.pth. The output remains loadable through "
            "MMEngine cfg.load_from because checkpoint loading is non-strict."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input OPUS fusion_pretrain_model.pth")
    parser.add_argument("--output", required=True, type=Path, help="Output slim AdaOcc checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the subset summary without writing")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite")

    import torch

    checkpoint = torch.load(args.input, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state_dict = get_state_dict(checkpoint)
    selected = filter_state_dict(state_dict)
    selected, converted_keys = convert_spconv_kernels_for_load_hook(selected)
    if not selected:
        raise RuntimeError(f"No keys matched prefixes: {DEFAULT_PREFIXES}")

    print(f"input: {args.input}")
    print(f"input_sha256: {sha256_file(args.input)}")
    print(f"input_keys: {len(state_dict)}")
    print(f"selected_keys: {len(selected)}")
    print(f"selected_prefixes: {', '.join(DEFAULT_PREFIXES)}")
    print(f"selected_tensor_size_mb: {state_dict_nbytes(selected) / 1024 / 1024:.2f}")
    print(f"converted_spconv_kernels: {len(converted_keys)}")

    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_output_checkpoint(checkpoint, selected, args.input, converted_keys), args.output)
    print(f"output: {args.output}")
    print(f"output_size_mb: {args.output.stat().st_size / 1024 / 1024:.2f}")
    print(f"output_sha256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
