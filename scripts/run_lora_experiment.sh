#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$FINPRM_ROOT"
require_cuda
require_directory "$FINPRM_MODEL_DIR"
for split in train dev test; do
  require_file "$FINPRM_DATA_DIR/processed/$split-primary/examples.jsonl"
done

run_dir="$FINPRM_RUN_DIR/qwen-prm-lora-seed-$FINPRM_SEED"
mkdir -p "$run_dir"
capture_environment "$run_dir/environment"
nvidia-smi | tee "$run_dir/nvidia-smi-before.txt"

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$run_dir/base-dev" \
  --model "$FINPRM_MODEL_DIR" --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 --select-threshold

base_threshold=$("$FINPRM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$run_dir/base-dev/metrics.json")
"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/test-primary/examples.jsonl" \
  --output "$run_dir/base-test" \
  --model "$FINPRM_MODEL_DIR" --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 --threshold "$base_threshold"

"$FINPRM_PYTHON" scripts/train_lora_prm.py \
  --train-data "$FINPRM_DATA_DIR/processed/train-primary/examples.jsonl" \
  --output "$run_dir/lora" \
  --model "$FINPRM_MODEL_DIR" \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" --max-length "$FINPRM_MAX_LENGTH" \
  --epochs 1 --gradient-accumulation 16 --learning-rate 5e-5 \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --seed "$FINPRM_SEED"

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$run_dir/lora-dev" \
  --model "$FINPRM_MODEL_DIR" --adapter "$run_dir/lora/adapter" --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 --select-threshold

lora_threshold=$("$FINPRM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$run_dir/lora-dev/metrics.json")
"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/test-primary/examples.jsonl" \
  --output "$run_dir/lora-test" \
  --model "$FINPRM_MODEL_DIR" --adapter "$run_dir/lora/adapter" --quantization 4bit \
  --evidence-mode "$FINPRM_EVIDENCE_MODE" --max-length "$FINPRM_MAX_LENGTH" \
  --batch-size 1 --threshold "$lora_threshold"

nvidia-smi | tee "$run_dir/nvidia-smi-after.txt"
printf '%s\n' "Base/LoRA experiment complete: $run_dir"
