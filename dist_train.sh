#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPUS=${1:-8}
CONFIG=${2:-configs/adaocc/radio_occscannet_mini.py}
shift $(( $# >= 2 ? 2 : $# ))

python -m torch.distributed.run \
  --nproc_per_node "$GPUS" \
  "$ROOT/train.py" \
  --config "$CONFIG" \
  "$@"
