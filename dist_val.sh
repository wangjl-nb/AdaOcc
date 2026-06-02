#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOCAL_CONFIG="${ADAOCC_LOCAL_CONFIG:-$ROOT/configs/local_paths.sh}"
if [[ -f "$LOCAL_CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_CONFIG"
elif [[ -n "${ADAOCC_LOCAL_CONFIG:-}" ]]; then
  echo "[error] ADAOCC_LOCAL_CONFIG points to a missing file: $LOCAL_CONFIG" >&2
  exit 1
fi

GPUS=${1:?usage: dist_val.sh GPUS CONFIG WEIGHT [MASTER_PORT]}
CONFIG=${2:-configs/adaocc/radio_occscannet_mini.py}
WEIGHT=${3:?usage: dist_val.sh GPUS CONFIG WEIGHT [MASTER_PORT]}
MASTER_PORT=${MASTER_PORT:-${4:-29500}}
PYTHON=${PYTHON:-python}
export ADAOCC_REPO_ROOT="${ADAOCC_REPO_ROOT:-$ROOT}"
export ADAOCC_OUTPUT_ROOT="${ADAOCC_OUTPUT_ROOT:-$ROOT/outputs}"
export HF_HOME="${HF_HOME:-$ROOT/pretrain/huggingface}"

"$PYTHON" -m torch.distributed.run \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${GPUS}" \
  "$ROOT/val.py" \
  --config "${CONFIG}" \
  --weights "${WEIGHT}"
