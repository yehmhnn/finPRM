#!/usr/bin/env python3
"""Build JSONL process-verification examples and an audit summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from finprm.data import build_split, load_split


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-size", type=int, default=100)
    args = parser.parse_args()

    sources = list(load_split(args.source, args.limit))
    result = build_split(
        sources,
        args.split,
        max_negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as stream:
        for example in result.examples:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    label_counts = Counter(example.target.label for example in result.examples)
    corruption_counts = Counter(
        example.metadata.corruption_type
        for example in result.examples
        if example.metadata.corruption_type is not None
    )
    negative_examples = [
        example for example in result.examples if example.target.label == 0
    ]
    audit_path = args.output / "audit_sample.jsonl"
    with audit_path.open("w", encoding="utf-8") as stream:
        for example in negative_examples[: args.audit_size]:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    audit_csv_path = args.output / "audit_sample.csv"
    with audit_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "stable_id",
                "finqa_id",
                "question",
                "prefix",
                "gold_operation",
                "candidate",
                "corruption_type",
            ),
        )
        writer.writeheader()
        for example in negative_examples[: args.audit_size]:
            writer.writerow(
                {
                    "stable_id": example.metadata.stable_id,
                    "finqa_id": example.metadata.finqa_id,
                    "question": example.input.question,
                    "prefix": " ; ".join(example.input.prefix),
                    "gold_operation": example.metadata.gold_operation,
                    "candidate": example.input.candidate,
                    "corruption_type": example.metadata.corruption_type,
                }
            )

    stable_ids = [example.metadata.stable_id for example in result.examples]
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("stable ID collision detected")

    summary = {
        "source": str(args.source),
        "split": args.split,
        "source_examples": len(sources),
        "process_examples": len(result.examples),
        "positive_examples": label_counts[1],
        "negative_examples": label_counts[0],
        "unique_stable_ids": len(set(stable_ids)),
        "corruption_counts": dict(sorted(corruption_counts.items())),
        "rejections": dict(sorted(result.rejections.items())),
        "seed": args.seed,
        "negatives_per_positive": args.negatives_per_positive,
        "examples_sha256": file_sha256(examples_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
