#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$FINPRM_ROOT"

mkdir -p "$FINPRM_DATA_DIR/raw/finqa" "$FINPRM_DATA_DIR/processed" "$(dirname "$FINPRM_MODEL_DIR")"
"$FINPRM_PYTHON" scripts/download_finqa.py \
  --output "$FINPRM_DATA_DIR/raw/finqa" \
  --transport archive

for split in train dev test; do
  "$FINPRM_PYTHON" scripts/validate_finqa.py \
    "$FINPRM_DATA_DIR/raw/finqa/$split.json" \
    --output "$FINPRM_DATA_DIR/processed/$split-validation.json"
  "$FINPRM_PYTHON" scripts/build_process_data.py \
    "$FINPRM_DATA_DIR/raw/finqa/$split.json" \
    --split "$split" \
    --output "$FINPRM_DATA_DIR/processed/$split-full" \
    --negatives-per-positive 2 \
    --audit-per-group 50 \
    --seed "$FINPRM_SEED"
  "$FINPRM_PYTHON" scripts/build_evaluation_set.py \
    "$FINPRM_DATA_DIR/processed/$split-full/examples.jsonl" \
    --output "$FINPRM_DATA_DIR/processed/$split-primary" \
    --seed "$FINPRM_SEED"
done

"$FINPRM_ROOT/.venv/bin/modelscope" download \
  --model "$FINPRM_MODEL_ID" \
  --revision "$FINPRM_MODELSCOPE_REVISION" \
  --local_dir "$FINPRM_MODEL_DIR"

for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  require_file "$FINPRM_MODEL_DIR/$shard"
done
"$FINPRM_PYTHON" scripts/verify_checkpoint.py "$FINPRM_MODEL_DIR"

"$FINPRM_PYTHON" - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

model_dir = os.environ["FINPRM_MODEL_DIR"]
revision = os.environ["FINPRM_MODEL_REVISION"]
config = AutoConfig.from_pretrained(model_dir, revision=revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=revision, trust_remote_code=True)
ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
if ids != [151651]:
    raise SystemExit(f"Unexpected <extra_0> tokenization: {ids}")
if config.architectures != ["Qwen2ForProcessRewardModel"]:
    raise SystemExit(f"Unexpected model architecture: {config.architectures}")
print({"architecture": config.architectures[0], "step_token_id": ids[0]})
PY

printf '%s\n' 'Data, evaluation manifests, and model checkpoint are ready.'
