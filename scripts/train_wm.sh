#!/usr/bin/env bash
set -euo pipefail

# World Model Stage 2 training
# Usage: bash scripts/train_wm.sh

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

python tools/train.py \
  projects/configs/sparsedrive_small_stage2_wm_optimized_v2.py \
  --work-dir work_dirs/sparsedrive_stage2_wm
