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

GPUS=${1:-8}
CONFIG=${2:-configs/adaocc/radio_occscannet_mini.py}
shift $(( $# >= 2 ? 2 : $# ))

PYTHON=${PYTHON:-python}
export ADAOCC_REPO_ROOT="${ADAOCC_REPO_ROOT:-$ROOT}"
export ADAOCC_OUTPUT_ROOT="${ADAOCC_OUTPUT_ROOT:-$ROOT/outputs}"
export HF_HOME="${HF_HOME:-$ROOT/pretrain/huggingface}"
mkdir -p "$ADAOCC_OUTPUT_ROOT"

ADAOCC_RUN_LABEL=${ADAOCC_RUN_LABEL:-adaocc_radio_mini}
if [[ $# -ge 2 && "${1:-}" == "--run-label" ]]; then
  ADAOCC_RUN_LABEL=$2
  shift 2
fi

ADAOCC_DISABLE_MSMV_CUDA=${ADAOCC_DISABLE_MSMV_CUDA:-1}
ADAOCC_RUN_TIMESTAMP=${ADAOCC_RUN_TIMESTAMP:-$(date +%Y-%m-%d/%H-%M-%S)}
PYTHONWARNINGS="ignore:torch.utils.checkpoint:UserWarning,ignore:The torch.cuda.*DtypeTensor constructors are no longer recommended.:UserWarning" \
ADAOCC_DEBUG_FINITE=${ADAOCC_DEBUG_FINITE:-1} \
ADAOCC_DISABLE_MSMV_CUDA=$ADAOCC_DISABLE_MSMV_CUDA \
ADAOCC_RUN_TIMESTAMP=$ADAOCC_RUN_TIMESTAMP \
ADAOCC_RUN_LABEL=$ADAOCC_RUN_LABEL \
"$PYTHON" -m torch.distributed.run --nproc_per_node "$GPUS" "$ROOT/train.py" --config "$CONFIG" "$@"
