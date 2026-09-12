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

## Run the local classifier smoke test

The tiny random BERT checkpoint is not an experimental baseline. It verifies the
complete software path using an unaudited pilot: stable text serialization,
question-grouped splitting, tokenization, binary training, evaluation, model
saving, and prediction-equivalent reloading.

```sh
uv sync --extra dev --extra ml --python 3.11
uv run python scripts/train_smoke.py --config configs/local.yaml
```

Outputs are written under ignored `runs/local-smoke/`. Accuracy from this run is
not a research result because the checkpoint is random and the pilot labels have
not completed human audit.

On a CUDA machine, first rebuild the ignored pilot data and run the identical
pipeline with:

```sh
uv run python scripts/train_smoke.py --config configs/cuda_pipeline_smoke.yaml
```

This must report `"device": "cuda"` and a near-zero reload difference before a
real checkpoint or LoRA training is attempted.

## Build a process-supervision pilot

Each gold FinQA operation becomes a positive next-step example. The builder also
creates conservative negatives through one operator substitution or operand
reversal at a time. Finance-specific rules additionally replace a number with a
different number from the same local evidence unit and remove explicit percent
or scale conversion. A diagnostic rule creates dangling intermediate references.
The builder rejects candidates that preserve the gold step value or the complete
program's final answer. Invalid references are retained only when the validator
confirms that the candidate points to an unavailable step.

The implemented corruption labels are:

- `entity_context_swap`: substitute a number found beside the original number
  in the same question, supporting sentence, report sentence, or table row;
- `unit_scale_mismatch`: remove an explicit percent sign or scale multiplier;
- `dangling_reference`: point to a current, not-yet-produced intermediate value;
- `operator_substitution`: retain operands but change the arithmetic operator;
- `operand_reversal`: reverse arguments for an order-sensitive operator.

Arithmetic-result corruption is intentionally excluded from the structured core:
an operation such as `subtract(4500, 1200)` does not contain a claimed result for
us to corrupt. That error family becomes applicable only if an optional
natural-language trace says, for example, that the result is 3100.

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
