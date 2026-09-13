#!/usr/bin/env bash
set -euo pipefail

FINPRM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -a
source "$FINPRM_ROOT/configs/experiment.env"
set +a

export HF_HOME=${HF_HOME:-/root/autodl-tmp/huggingface}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="$FINPRM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
FINPRM_PYTHON=${FINPRM_PYTHON:-$FINPRM_ROOT/.venv/bin/python}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_directory() {
  if [[ ! -d "$1" ]]; then
    printf 'Required directory is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_cuda() {
  "$FINPRM_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; start the GPU instance first")
if torch.cuda.get_device_capability(0)[0] < 8:
    raise SystemExit("BF16 QLoRA requires an Ampere-or-newer GPU")
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
})
PY
}

capture_environment() {
  local output_dir=$1
  mkdir -p "$output_dir"
  git -C "$FINPRM_ROOT" rev-parse HEAD > "$output_dir/code-commit.txt"
  git -C "$FINPRM_ROOT" status --short > "$output_dir/code-status.txt"
  git -C "$FINPRM_ROOT" diff --binary > "$output_dir/code.diff"
  "$FINPRM_PYTHON" -m pip freeze > "$output_dir/pip-freeze.txt"
  uname -a > "$output_dir/uname.txt"
}
