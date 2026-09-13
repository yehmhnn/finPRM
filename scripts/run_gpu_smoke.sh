#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$FINPRM_ROOT"
require_cuda
require_directory "$FINPRM_MODEL_DIR"
require_file "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl"

smoke_dir="$FINPRM_RUN_DIR/gpu-smoke"
mkdir -p "$smoke_dir"
capture_environment "$smoke_dir/environment"
nvidia-smi | tee "$smoke_dir/nvidia-smi.txt"

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$smoke_dir/base-score" \
  --model "$FINPRM_MODEL_DIR" \
  --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 \
  --max-examples 8

"$FINPRM_PYTHON" scripts/train_lora_prm.py \
  --train-data "$FINPRM_DATA_DIR/processed/train-primary/examples.jsonl" \
  --output "$smoke_dir/lora" \
  --model "$FINPRM_MODEL_DIR" \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length 512 \
  --max-examples 8 \
  --epochs 1 \
  --gradient-accumulation 4 \
  --seed "$FINPRM_SEED"

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$smoke_dir/adapter-score" \
  --model "$FINPRM_MODEL_DIR" \
  --adapter "$smoke_dir/lora/adapter" \
  --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 \
  --max-examples 8

printf '%s\n' "GPU smoke test complete: $smoke_dir"
