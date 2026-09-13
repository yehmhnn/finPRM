#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
source "$FINPRM_ROOT/configs/retrieval.env"
cd "$FINPRM_ROOT"

mode=${1:-$FINPRM_RETRIEVAL_PRIMARY_MODE}
if [[ " $FINPRM_RETRIEVAL_MODES " != *" $mode "* ]]; then
  printf 'Unsupported retrieval mode: %s\n' "$mode" >&2
  exit 2
fi

run_dir="$FINPRM_RUN_DIR/retrieval-$mode-k$FINPRM_RETRIEVAL_K"
adapter="$FINPRM_RUN_DIR/qwen-prm-lora-seed-$FINPRM_SEED/lora/adapter"
mkdir -p "$run_dir"/{base-dev,base-test,lora-dev,lora-test}
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-status.txt"; exit "$status"' EXIT

require_cuda
require_directory "$FINPRM_MODEL_DIR"
require_directory "$adapter"
for split in dev test; do
  require_file "$FINPRM_DATA_DIR/processed/$split-primary/examples.jsonl"
  require_file "$FINPRM_RETRIEVAL_SANITY_DIR/$mode-k$FINPRM_RETRIEVAL_K-$split-plan.json"
done

sha256sum "$adapter/adapter_model.safetensors" > "$run_dir/lora-adapter-sha256-before.txt"

# Assert that the exact chat-templated prompts fit without target truncation.
"$FINPRM_PYTHON" - \
  "$FINPRM_DATA_DIR" "$FINPRM_MODEL_DIR" "$FINPRM_RETRIEVAL_SANITY_DIR" \
  "$run_dir" "$mode" "$FINPRM_RETRIEVAL_K" "$FINPRM_MAX_LENGTH" \
  "$FINPRM_EVIDENCE_MODE" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from finprm.models.qwen_prm import MODEL_REVISION, encode_record, load_records
from scripts.score_prm import load_retrieval_plan

data_dir, model_dir, plan_dir, run_dir = map(Path, sys.argv[1:5])
mode = sys.argv[5]
k = int(sys.argv[6])
max_length = int(sys.argv[7])
evidence_mode = sys.argv[8]
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    local_files_only=True,
)
result = {}
for split in ("dev", "test"):
    data = data_dir / f"processed/{split}-primary/examples.jsonl"
    plan = plan_dir / f"{mode}-k{k}-{split}-plan.json"
    _, demonstrations = load_retrieval_plan(plan, data)
    encoded = [
        encode_record(
            tokenizer,
            record,
            max_length,
            evidence_mode,
            demonstrations=demonstrations[record["metadata"]["stable_id"]],
        )
        for record in load_records(data)
    ]
    assert all(item.original_tokens <= max_length for item in encoded)
    assert all(item.truncated_tokens == 0 for item in encoded)
    assert all(len(item.input_ids) == item.original_tokens for item in encoded)
    result[split] = {
        "examples": len(encoded),
        "max_actual_tokens": max(item.original_tokens for item in encoded),
        "truncated_examples": 0,
    }
(run_dir / "prompt-assertion.json").write_text(
    json.dumps({"status": "PASS", **result}, indent=2) + "\n",
    encoding="utf-8",
)
print({"strict_prompt_assertion": "PASS", **result})
PY

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$run_dir/base-dev" --model "$FINPRM_MODEL_DIR" \
  --quantization 4bit --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" --batch-size 1 --select-threshold \
  --retrieval-plan "$FINPRM_RETRIEVAL_SANITY_DIR/$mode-k$FINPRM_RETRIEVAL_K-dev-plan.json"

base_threshold=$("$FINPRM_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
  "$run_dir/base-dev/metrics.json")
printf '%s\n' "$base_threshold" > "$run_dir/frozen-base-dev-threshold.txt"
"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/test-primary/examples.jsonl" \
  --output "$run_dir/base-test" --model "$FINPRM_MODEL_DIR" \
  --quantization 4bit --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" --batch-size 1 --threshold "$base_threshold" \
  --retrieval-plan "$FINPRM_RETRIEVAL_SANITY_DIR/$mode-k$FINPRM_RETRIEVAL_K-test-plan.json"

"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --output "$run_dir/lora-dev" --model "$FINPRM_MODEL_DIR" --adapter "$adapter" \
  --quantization 4bit --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" --batch-size 1 --select-threshold \
  --retrieval-plan "$FINPRM_RETRIEVAL_SANITY_DIR/$mode-k$FINPRM_RETRIEVAL_K-dev-plan.json"

lora_threshold=$("$FINPRM_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
  "$run_dir/lora-dev/metrics.json")
printf '%s\n' "$lora_threshold" > "$run_dir/frozen-lora-dev-threshold.txt"
"$FINPRM_PYTHON" scripts/score_prm.py \
  --data "$FINPRM_DATA_DIR/processed/test-primary/examples.jsonl" \
  --output "$run_dir/lora-test" --model "$FINPRM_MODEL_DIR" --adapter "$adapter" \
  --quantization 4bit --evidence-mode "$FINPRM_EVIDENCE_MODE" \
  --max-length "$FINPRM_MAX_LENGTH" --batch-size 1 --threshold "$lora_threshold" \
  --retrieval-plan "$FINPRM_RETRIEVAL_SANITY_DIR/$mode-k$FINPRM_RETRIEVAL_K-test-plan.json"

sha256sum "$adapter/adapter_model.safetensors" > "$run_dir/lora-adapter-sha256-after.txt"
diff -u "$run_dir/lora-adapter-sha256-before.txt" "$run_dir/lora-adapter-sha256-after.txt"
touch "$run_dir/COMPLETE"
