#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPUS=${1:-8}
CONFIG=${2:-configs/occscannet/radio_occscannet_mini.py}
WEIGHT=${3:?usage: dist_val.sh GPUS CONFIG WEIGHT [MASTER_PORT]}
MASTER_PORT=${MASTER_PORT:-${4:-29500}}
python -m torch.distributed.run \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${GPUS}" \
  "$ROOT/val.py" \
  --config "${CONFIG}" \
  --weights "${WEIGHT}"
