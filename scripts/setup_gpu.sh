#!/usr/bin/env bash
set -euo pipefail

FINPRM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$FINPRM_ROOT"

python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12, found {sys.version}")
PY

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --requirement requirements-gpu.txt

.venv/bin/python - <<'PY'
import torch, transformers, peft, bitsandbytes, accelerate
if not torch.__version__.startswith("2.8."):
    raise SystemExit(f"Expected image-provided PyTorch 2.8.x, found {torch.__version__}")
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "bitsandbytes": bitsandbytes.__version__,
    "accelerate": accelerate.__version__,
})
PY

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q
printf '%s\n' 'Environment setup complete.'
