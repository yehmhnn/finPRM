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

The implementation includes deterministic FinQA process-example construction,
native Qwen PRM scoring, QLoRA adaptation, and Train-only demonstration
retrieval.

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

## Laptop integration test

- `configs/local.yaml` uses `hf-internal-testing/tiny-random-bert`. This is a
  deliberately tiny model with random weights. Its predictions are meaningless;
  it only verifies tokenization, batching, training, saving, and reloading.
The tiny random BERT checkpoint is not an experimental baseline. It verifies the
complete software path using an unaudited pilot: stable text serialization,
question-grouped splitting, tokenization, binary training, evaluation, model
saving, and prediction-equivalent reloading.

Long inputs are truncated from the evidence side only. The smoke trainer checks
that the complete question, correct prefix, candidate, and task fit within
`max_length`; it fails instead of silently truncating those protected fields and
records length statistics in `metrics.json`.

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
  --output data/processed/pilot-hard-v2 \
  --limit 100 \
  --negatives-per-positive 2 \
  --audit-per-group 50 \
  --seed 42
```

The ignored output directory contains:

- `examples.jsonl`: complete model inputs, targets, and audit metadata;
- `summary.json`: counts, rejection reasons, and a deterministic checksum;
- `audit_sample.csv`: deterministic, stratified rows for manual review; and
- `audit_sample.jsonl`: the same audit examples with complete context.

The audit includes positives and up to 50 examples from every retained
corruption type, distributed across step positions and program lengths. An
exported audit is not considered reviewed until a person fills in
`human_valid` and `review_notes`.

Generated data remains outside Git. The builder and its validation rules are
committed so the same records can be reproduced on another machine.

For development and test, create the primary label-balanced evaluation set
after building the full diagnostic data:

```sh
uv run python scripts/build_evaluation_set.py \
  data/processed/dev-full/examples.jsonl \
  --output data/processed/dev-primary \
  --seed 42
```

This keeps one positive and one accepted negative per eligible source step.
The deterministic sampler assigns scarce corruption families first and records
the resulting per-type counts. Keep the full diagnostic set for error-type
analysis; do not tune the sampler from model performance on test data.

## Compile the proposal

From the `latex` directory:

```sh
latexmk -pdf -outdir=../output/pdf finprm_proposal.tex
```

## Implementation structure

```text
src/
  finprm/
    data/       FinQA parsing and process-example construction
    models/     serialization and native Qwen PRM utilities
    retrieval/  Train-only dense demonstration retrieval
scripts/        data, evaluation, training, and operational entry points
configs/        frozen experiment defaults and local smoke configuration
tests/          data, serialization, sampling, and retrieval tests
```

Large datasets, model checkpoints, and experiment logs are intentionally excluded from version control.

## Prepared GPU workflow

The frozen primary checkpoint is Qwen/Qwen2.5-Math-PRM-7B at revision
`0610740060112df12585d00a1c5f4624d2f59051`. It is loaded through its native
`Qwen2ForProcessRewardModel` implementation, and only the final `<extra_0>`
position for the candidate operation is used as `p_correct`. Base and LoRA
evaluation both use the same NF4 quantized load so a precision change is not
mistaken for an adapter gain.

On the CUDA 12.8 / PyTorch 2.8 / Python 3.12 image, run:

```sh
cd /root/autodl-tmp/finPRM
./scripts/setup_gpu.sh
./scripts/prepare_assets.sh
./scripts/run_gpu_smoke.sh
./scripts/run_lora_experiment.sh
```

`setup_gpu.sh` reuses the image-provided PyTorch and installs a pinned PRM/QLoRA
compatibility set. `prepare_assets.sh` verifies source checksums, validates and
builds all official splits, creates label-balanced primary manifests, downloads
the frozen checkpoint from Qwen's official ModelScope mirror at immutable
revision `5699834a93b4707388291a5d6be57a30dfcf310e`, verifies every large
weight shard by SHA256, and checks its architecture and reward-token ID. The GPU
smoke test runs eight base examples, eight QLoRA training examples, and an
adapter reload/score. Only after it succeeds should the full experiment run.

The experiment selects thresholds independently for base and LoRA using the
same dev macro-F1 rule, then applies each frozen threshold once to test. It
saves per-example probabilities, metrics, adapters, package versions, GPU
information, Git status, and the exact uncommitted code diff under `runs/`.

## Retrieval workflow

Retrieval uses the frozen `BAAI/bge-small-en-v1.5` checkpoint, a Train-only
index, and two supported keys: `question` and `joint` (question, prefix, and
candidate). Labels are added to demonstrations only after similarity search.

Build the index and prompt-length plans without running PRM inference:

```sh
./scripts/run_retrieval_sanity.sh
```

After reviewing those plans, run the frozen primary joint condition or the
question-only ablation. These commands reuse the existing LoRA adapter and do
not train either model:

```sh
./scripts/run_retrieval_experiment.sh
./scripts/run_retrieval_experiment.sh question
```
