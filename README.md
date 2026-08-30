# FinPRM-Adapt

Research project on parameter-efficient adaptation of process reward models (PRMs) to structured financial numerical reasoning.

The proposed study constructs binary step-verification examples from FinQA's executable gold programs and compares four conditions:

1. an unadapted base verifier;
2. retrieval-only adaptation;
3. LoRA-based supervised fine-tuning;
4. retrieval combined with LoRA.

## Repository contents

- `latex/finprm_proposal.tex` - editable LaTeX manuscript
- `latex/references.bib` - bibliography
- `output/pdf/finprm_proposal.pdf` - compiled proposal
- `PRM project.pdf` - original project concept

The first implementation milestone provides a strict FinQA loader and a
deterministic executor for the dataset's program language. Model training is
added only after this hardware-independent foundation is validated.

## Local setup

Use Python 3.11 when creating a fresh environment. The current foundation also
runs on Python 3.9 so it can be tested with the macOS system Python.

```sh
uv sync --extra dev --python 3.11
uv run python scripts/download_finqa.py
uv run python scripts/validate_finqa.py data/raw/finqa/train.json --limit 100
uv run pytest
```

The dataset is downloaded from the official FinQA repository at a pinned
revision. Files under `data/` are intentionally ignored by Git and are not
uploaded to this repository.

## Laptop and GPU configurations

- `configs/local.yaml` uses `hf-internal-testing/tiny-random-bert`. This is a
  deliberately tiny model with random weights. Its predictions are meaningless;
  it only verifies tokenization, batching, training, saving, and reloading.
- `configs/gpu_smoke.yaml` runs a small subset with the real model on CUDA.
- `configs/gpu_full.yaml` is used only after the CUDA smoke test passes.

The real model checkpoint remains `TO_BE_FROZEN` until tokenizer compatibility,
license, local inference, and GPU memory requirements have been checked.

## Build a process-supervision pilot

Each gold FinQA operation becomes a positive next-step example. The builder also
creates conservative negatives through one operator substitution or operand
reversal at a time. It rejects candidates that fail execution, preserve the gold
step value, or preserve the complete program's final answer.

```sh
uv run python scripts/build_process_data.py \
  data/raw/finqa/train.json \
  --split train \
  --output data/processed/pilot-train-100 \
  --limit 100 \
  --negatives-per-positive 2 \
  --seed 42
```

The ignored output directory contains:

- `examples.jsonl`: complete model inputs, targets, and audit metadata;
- `summary.json`: counts, rejection reasons, and a deterministic checksum;
- `audit_sample.csv`: compact rows for manual label review; and
- `audit_sample.jsonl`: the same audit examples with complete context.

Generated data remains outside Git. The builder and its validation rules are
committed so the same records can be reproduced on another machine.

## Compile the proposal

From the `latex` directory:

```sh
latexmk -pdf -outdir=../output/pdf finprm_proposal.tex
```

## Planned implementation structure

```text
src/
  data/        FinQA parsing and process-example construction
  retrieval/   question- and step-level retrieval
  models/      base verifier and LoRA training
  evaluation/  intrinsic PRM and Best-of-N evaluation
  app/         laptop-local reasoning checker
configs/       reproducible experiment configurations
tests/         executor and data-validation tests
```

Large datasets, model checkpoints, and experiment logs are intentionally excluded from version control.
