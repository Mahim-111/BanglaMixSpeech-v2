#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/config/defaults.env"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  DATASET_ROOT="$1"
  shift
else
  DATASET_ROOT="${BANGLAMIX_DATASET:-$PROJECT_ROOT/data/BanglaMix}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"
if [[ -n "$CHECKPOINT_ROOT" && "$CHECKPOINT_ROOT" != /* ]]; then
  CHECKPOINT_ROOT="$PROJECT_ROOT/$CHECKPOINT_ROOT"
fi
PYTHON="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Activate the shared Conda environment, then run this script again." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/repos/speechbrain:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export ASTEROID_CACHE="${ASTEROID_CACHE:-$PROJECT_ROOT/cache/asteroid}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/tmp}"
mkdir -p "$OUTPUT_ROOT" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/tmp" "$PROJECT_ROOT/cache"

if [[ ! -d "$DATASET_ROOT/Mix_Speech" ]]; then
  echo "BanglaMix dataset not found at: $DATASET_ROOT" >&2
  echo "Pass it as the first argument:" >&2
  echo "  ./run_zero_shot.sh /absolute/path/to/BanglaMix" >&2
  exit 2
fi

"$PYTHON" "$PROJECT_ROOT/scripts/preflight.py" --dataset "$DATASET_ROOT" --device "$DEVICE"

COMMAND=(
  "$PYTHON" "$PROJECT_ROOT/run_suite.py"
  --dataset "$DATASET_ROOT"
  --models "$MODELS"
  --speakers "$SPEAKERS"
  --split "$SPLIT"
  --variation "$VARIATION"
  --metrics "$METRICS"
  --evaluation-sr "$EVALUATION_SR"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --output "$OUTPUT_ROOT"
)

if [[ -n "$MAX_ITEMS" ]]; then
  COMMAND+=(--max-items "$MAX_ITEMS")
fi
if [[ -n "$CHECKPOINT_ROOT" ]]; then
  COMMAND+=(--checkpoint-root "$CHECKPOINT_ROOT")
fi
if [[ -n "$RUN_NAME" ]]; then
  COMMAND+=(--run-name "$RUN_NAME")
fi

COMMAND+=("$@")
exec "${COMMAND[@]}"
