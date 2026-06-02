#!/usr/bin/env bash
# Copy this file to configs/local_paths.sh and edit paths for your machine.
# This file is sourced by dist_train.sh, dist_val.sh, and scripts/link_local_assets.sh.
# Keep configs/local_paths.sh private; it is ignored by git.

# Repository-local defaults. You usually only need to edit the *_SRC entries
# below when creating symlinks to external datasets/checkpoints.
export ADAOCC_REPO_ROOT="${ADAOCC_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ADAOCC_DATA_ROOT="${ADAOCC_DATA_ROOT:-$ADAOCC_REPO_ROOT/data/OccScanNet}"
export ADAOCC_PRETRAIN="${ADAOCC_PRETRAIN:-$ADAOCC_REPO_ROOT/pretrain/fusion_pretrain_model.pth}"
export ADAOCC_DEPTH_ANYTHING_CKPT="${ADAOCC_DEPTH_ANYTHING_CKPT:-$ADAOCC_REPO_ROOT/pretrain/depth_anything/finetune_scannet_depthanythingv2.pth}"
export ADAOCC_OUTPUT_ROOT="${ADAOCC_OUTPUT_ROOT:-$ADAOCC_REPO_ROOT/outputs}"
export HF_HOME="${HF_HOME:-$ADAOCC_REPO_ROOT/pretrain/huggingface}"

# RADIO can be a local snapshot path or the Hugging Face model id. Set
# ADAOCC_HF_LOCAL_FILES_ONLY=1 after the snapshot/cache exists.
export ADAOCC_RADIO_MODEL="${ADAOCC_RADIO_MODEL:-nvidia/C-RADIOv3-B}"
export ADAOCC_HF_LOCAL_FILES_ONLY="${ADAOCC_HF_LOCAL_FILES_ONLY:-0}"

# Default public reproduction path: online frozen DepthAnythingV2.
# Set ADAOCC_ONLINE_DEPTH=0 only if you generated/provided precomputed depth PNGs.
export ADAOCC_ONLINE_DEPTH="${ADAOCC_ONLINE_DEPTH:-1}"
export ADAOCC_PRECOMPUTED_DEPTH_FOR_ONLINE="${ADAOCC_PRECOMPUTED_DEPTH_FOR_ONLINE:-0}"

# The single-level RADIO baseline can use the PyTorch fallback if the custom
# MSMV CUDA extension is unavailable.
export ADAOCC_DISABLE_MSMV_CUDA="${ADAOCC_DISABLE_MSMV_CUDA:-1}"
export ADAOCC_DEBUG_FINITE="${ADAOCC_DEBUG_FINITE:-1}"

# Optional link sources used by scripts/link_local_assets.sh. Edit these paths
# and run `scripts/link_local_assets.sh` to create repo-local symlinks.
export ADAOCC_LINK_DATA_SRC="${ADAOCC_LINK_DATA_SRC:-}"
export ADAOCC_LINK_PRETRAIN_SRC="${ADAOCC_LINK_PRETRAIN_SRC:-}"
export ADAOCC_LINK_DEPTH_ANYTHING_CKPT_SRC="${ADAOCC_LINK_DEPTH_ANYTHING_CKPT_SRC:-}"
# Use either a local snapshot directory C-RADIOv3-B/ or a HF cache directory
# models--nvidia--C-RADIOv3-B/. Leave empty to let Hugging Face download.
export ADAOCC_LINK_RADIO_CACHE_SRC="${ADAOCC_LINK_RADIO_CACHE_SRC:-}"
