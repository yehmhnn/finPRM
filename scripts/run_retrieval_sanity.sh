#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
source "$FINPRM_ROOT/configs/retrieval.env"
cd "$FINPRM_ROOT"

require_directory "$FINPRM_MODEL_DIR"
require_directory "$FINPRM_EMBEDDING_DIR"
for split in train dev test; do
  require_file "$FINPRM_DATA_DIR/processed/$split-primary/examples.jsonl"
done

"$FINPRM_PYTHON" scripts/build_retrieval_index.py \
  --train-data "$FINPRM_DATA_DIR/processed/train-primary/examples.jsonl" \
  --output "$FINPRM_RETRIEVAL_INDEX" \
  --embedding-model "$FINPRM_EMBEDDING_DIR" \
  --batch-size "$FINPRM_RETRIEVAL_BATCH_SIZE"

"$FINPRM_PYTHON" scripts/retrieval_sanity.py \
  --index "$FINPRM_RETRIEVAL_INDEX" \
  --embedding-model "$FINPRM_EMBEDDING_DIR" \
  --qwen-model "$FINPRM_MODEL_DIR" \
  --dev-data "$FINPRM_DATA_DIR/processed/dev-primary/examples.jsonl" \
  --test-data "$FINPRM_DATA_DIR/processed/test-primary/examples.jsonl" \
  --output "$FINPRM_RETRIEVAL_SANITY_DIR/report.json" \
  --k-values $FINPRM_RETRIEVAL_K_VALUES \
  --max-length "$FINPRM_MAX_LENGTH" \
  --sample-count 3

printf '%s\n' "Retrieval sanity check complete; no PRM inference was run."
