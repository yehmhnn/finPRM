#!/usr/bin/env python3
"""Export lightweight release tables from completed experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from finprm.evaluation import binary_metrics


@dataclass(frozen=True)
class Condition:
    name: str
    model: str
    retrieval_key: str
    run_directory: str
    prefix: str


CONDITIONS = (
    Condition("Base", "Base", "none", "qwen-prm-base-seed-42", "base"),
    Condition(
        "Base + question-only retrieval",
        "Base",
        "question",
        "retrieval-question-k2",
        "base",
    ),
    Condition(
        "Base + joint retrieval", "Base", "joint", "retrieval-primary", "base"
    ),
    Condition("LoRA", "LoRA", "none", "qwen-prm-lora-seed-42", "lora"),
    Condition(
        "LoRA + question-only retrieval",
        "LoRA",
        "question",
        "retrieval-question-k2",
        "lora",
    ),
    Condition(
        "LoRA + joint retrieval", "LoRA", "joint", "retrieval-primary", "lora"
    ),
)

MAIN_FIELDS = (
    "condition",
    "model",
    "retrieval_key",
    "retrieval_k",
    "dev_threshold",
    "dev_macro_f1",
    "test_samples",
    "test_accuracy",
    "test_macro_f1",
    "test_auroc",
    "test_auprc_correct",
    "test_auprc_error",
    "test_brier",
    "test_ece_10_bin",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "test_runtime_seconds",
    "peak_gpu_allocated_gib",
)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artifact_paths(runs_dir: Path, condition: Condition) -> tuple[Path, Path, Path]:
    root = runs_dir / condition.run_directory
    return (
        root / f"{condition.prefix}-dev/metrics.json",
        root / f"{condition.prefix}-test/metrics.json",
        root / f"{condition.prefix}-test/predictions.jsonl",
    )


def main_metrics(runs_dir: Path) -> tuple[List[Dict[str, Any]], List[Path]]:
    rows = []
    sources = []
    for condition in CONDITIONS:
        dev_path, test_path, _ = artifact_paths(runs_dir, condition)
        dev = load_json(dev_path)
        test = load_json(test_path)
        if test["threshold"] != dev["threshold"]:
            raise ValueError(f"Test threshold was not frozen from Dev for {condition.name}")
        tn, fp = test["confusion_matrix"][0]
        fn, tp = test["confusion_matrix"][1]
        rows.append(
            {
                "condition": condition.name,
                "model": condition.model,
                "retrieval_key": condition.retrieval_key,
                "retrieval_k": test.get("retrieval_k", 0),
                "dev_threshold": dev["threshold"],
                "dev_macro_f1": dev["macro_f1"],
                "test_samples": test["examples"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_auroc": test["auroc_correct"],
                "test_auprc_correct": test["auprc_correct"],
                "test_auprc_error": test["auprc_error"],
                "test_brier": test["brier"],
                "test_ece_10_bin": test["ece_10_bin"],
                "test_tn": tn,
                "test_fp": fp,
                "test_fn": fn,
                "test_tp": tp,
                "test_runtime_seconds": test["elapsed_seconds"],
                "peak_gpu_allocated_gib": test.get("peak_gpu_allocated_gib"),
            }
        )
        sources.extend((dev_path, test_path))
    return rows, sources


def position_group(step_index: int) -> str:
    if step_index == 0:
        return "Step 1"
    if step_index == 1:
        return "Step 2"
    return "Step 3+"


def analysis_rows(
    runs_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    step_rows = []
    corruption_rows = []
    sources = []
    reference_records = None
    for condition in (CONDITIONS[0], CONDITIONS[3]):
        dev_path, _, predictions_path = artifact_paths(runs_dir, condition)
        threshold = load_json(dev_path)["threshold"]
        records = load_jsonl(predictions_path)
        identity = [
            (
                row["stable_id"],
                row["label"],
                row["step_index"],
                row["corruption_type"],
            )
            for row in records
        ]
        if reference_records is not None and identity != reference_records:
            raise ValueError("Base and LoRA Test predictions are not aligned")
        reference_records = identity

        grouped_steps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped_steps[position_group(record["step_index"])].append(record)
        for position in ("Step 1", "Step 2", "Step 3+"):
            group = grouped_steps[position]
            labels = [row["label"] for row in group]
            scores = [row["p_correct"] for row in group]
            metrics = binary_metrics(labels, scores, threshold)
            tn, fp = metrics["confusion_matrix"][0]
            fn, tp = metrics["confusion_matrix"][1]
            step_rows.append(
                {
                    "model": condition.model,
                    "position": position,
                    "samples": len(group),
                    "positives": sum(labels),
                    "negatives": len(labels) - sum(labels),
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

        grouped_corruptions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["label"] == 0:
                grouped_corruptions[record["corruption_type"]].append(record)
        for corruption_type in sorted(grouped_corruptions):
            group = grouped_corruptions[corruption_type]
            false_positives = sum(row["p_correct"] >= threshold for row in group)
            corruption_rows.append(
                {
                    "model": condition.model,
                    "corruption_type": corruption_type,
                    "negative_samples": len(group),
                    "threshold": threshold,
                    "accuracy": 1.0 - false_positives / len(group),
                    "false_positives": false_positives,
                }
            )
        sources.extend((dev_path, predictions_path))
    return step_rows, corruption_rows, sources


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=project_root / "runs")
    parser.add_argument("--output", type=Path, default=project_root / "results")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    main_rows, main_sources = main_metrics(args.runs)
    step_rows, corruption_rows, analysis_sources = analysis_rows(args.runs)
    write_csv(args.output / "main_metrics.csv", MAIN_FIELDS, main_rows)
    write_csv(
        args.output / "step_position_metrics.csv",
        tuple(step_rows[0]),
        step_rows,
    )
    write_csv(
        args.output / "corruption_type_metrics.csv",
        tuple(corruption_rows[0]),
        corruption_rows,
    )

    source_paths = sorted(set(main_sources + analysis_sources))
    manifest = {
        "schema_version": "finprm-release-results-v1",
        "generator": "scripts/export_results.py",
        "source_policy": "Existing completed artifacts only; no model inference",
        "sources": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in source_paths
        },
        "result_files": {
            "main_metrics.csv": len(main_rows),
            "step_position_metrics.csv": len(step_rows),
            "corruption_type_metrics.csv": len(corruption_rows),
        },
        "analysis_definitions": {
            "step_position": "Zero-based step_index grouped as Step 1, Step 2, and Step 3+; both labels included",
            "corruption_type": "Negative Test samples only; accuracy is the correctly rejected fraction",
        },
        "figure_files": {
            "overview.png": {
                "sha256": file_sha256(project_root / "assets/figures/overview.png"),
                "source": "User-provided finalized method overview",
            },
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
