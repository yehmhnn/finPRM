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

Implementation code and experiment configurations will be added as the project develops.

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
