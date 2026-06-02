#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CONFIG="${ADAOCC_LOCAL_CONFIG:-$repo_root/configs/local_paths.sh}"
if [[ -f "$LOCAL_CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_CONFIG"
elif [[ -n "${ADAOCC_LOCAL_CONFIG:-}" ]]; then
  echo "[error] ADAOCC_LOCAL_CONFIG points to a missing file: $LOCAL_CONFIG" >&2
  exit 1
fi

data_src="${ADAOCC_LINK_DATA_SRC:-}"
pretrain_src="${ADAOCC_LINK_PRETRAIN_SRC:-}"
radio_src="${ADAOCC_LINK_RADIO_CACHE_SRC:-}"
depth_ckpt_src="${ADAOCC_LINK_DEPTH_ANYTHING_CKPT_SRC:-}"

usage() {
  cat <<'USAGE'
Usage:
  cp configs/local_paths.example.sh configs/local_paths.sh
  # Edit ADAOCC_LINK_*_SRC in configs/local_paths.sh, then run:
  scripts/link_local_assets.sh

Override config file path with ADAOCC_LOCAL_CONFIG=/path/to/local_paths.sh.
Creates/updates symlinks under ignored data/ and pretrain/ paths. It never
removes source data or checkpoints.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "$repo_root/data" "$repo_root/pretrain/depth_anything" "$repo_root/pretrain/radio" "$repo_root/pretrain/huggingface/hub"

link_if_set() {
  local src="$1" dst="$2" label="$3"
  if [[ -z "$src" ]]; then
    echo "[skip] $label source env is unset"
    return 0
  fi
  if [[ ! -e "$src" ]]; then
    echo "[error] $label source missing: $src" >&2
    return 1
  fi
  ln -sfn "$src" "$dst"
  echo "[ok] $label -> $dst"
}

link_if_set "$data_src" "$repo_root/data/OccScanNet" "OccScanNet data root"
link_if_set "$pretrain_src" "$repo_root/pretrain/fusion_pretrain_model.pth" "OPUS fusion pretrain"
link_if_set "$depth_ckpt_src" "$repo_root/pretrain/depth_anything/finetune_scannet_depthanythingv2.pth" "DepthAnything fine-tuned checkpoint"
if [[ -n "$radio_src" ]]; then
  if [[ "$radio_src" == *models--nvidia--C-RADIOv3-B* ]]; then
    link_if_set "$radio_src" "$repo_root/pretrain/huggingface/hub/models--nvidia--C-RADIOv3-B" "RADIO HF cache"
  else
    link_if_set "$radio_src" "$repo_root/pretrain/radio/C-RADIOv3-B" "RADIO local snapshot"
  fi
else
  echo "[skip] RADIO source env is unset; config can download/use nvidia/C-RADIOv3-B if allowed"
fi
