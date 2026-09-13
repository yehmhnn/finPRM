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


def _audit_rank(example, seed: int) -> str:
    payload = f"{seed}|audit|{example.metadata.stable_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _position_bucket(example) -> str:
    if example.metadata.step_index == 0:
        return "first"
    if example.metadata.step_index + 1 == example.metadata.program_length:
        return "last"
    return "middle"


def stratified_audit_sample(examples, per_group: int, seed: int):
    """Sample deterministically across error type, step position, and length."""
    groups = {}
    for example in examples:
        key = example.metadata.corruption_type or "positive"
        groups.setdefault(key, {}).setdefault(
            (_position_bucket(example), example.metadata.program_length), []
        ).append(example)

    selected = []
    for group_name in sorted(groups):
        strata = groups[group_name]
        for values in strata.values():
            values.sort(key=lambda item: _audit_rank(item, seed))
        stratum_keys = sorted(strata)
        group_count = 0
        while group_count < per_group:
            added = False
            for key in stratum_keys:
                if strata[key]:
                    selected.append(strata[key].pop(0))
                    group_count += 1
                    added = True
                    if group_count == per_group:
                        break
            if not added:
                break
    return selected


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
    parser.add_argument(
        "--audit-per-group",
        type=int,
        default=50,
        help="examples per corruption type plus positives; samples are stratified",
    )
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
    if args.audit_per_group < 0:
        parser.error("--audit-per-group must be non-negative")
    audit_examples = stratified_audit_sample(
        result.examples, args.audit_per_group, args.seed
    )
    audit_path = args.output / "audit_sample.jsonl"
    with audit_path.open("w", encoding="utf-8") as stream:
        for example in audit_examples:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    audit_csv_path = args.output / "audit_sample.csv"
    with audit_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "stable_id",
                "finqa_id",
                "label",
                "question",
                "prefix",
                "step_index",
                "program_length",
                "gold_operation",
                "candidate",
                "corruption_type",
                "original_argument",
                "replacement_argument",
                "replacement_source",
                "human_valid",
                "review_notes",
            ),
        )
        writer.writeheader()
        for example in audit_examples:
            writer.writerow(
                {
                    "stable_id": example.metadata.stable_id,
                    "finqa_id": example.metadata.finqa_id,
                    "label": example.target.label,
                    "question": example.input.question,
                    "prefix": " ; ".join(example.input.prefix),
                    "step_index": example.metadata.step_index,
                    "program_length": example.metadata.program_length,
                    "gold_operation": example.metadata.gold_operation,
                    "candidate": example.input.candidate,
                    "corruption_type": example.metadata.corruption_type,
                    "original_argument": (example.metadata.corruption_details or {}).get("original", ""),
                    "replacement_argument": (example.metadata.corruption_details or {}).get("replacement", ""),
                    "replacement_source": (example.metadata.corruption_details or {}).get("replacement_source", ""),
                    "human_valid": "",
                    "review_notes": "",
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
        "audit_examples": len(audit_examples),
        "audit_counts": dict(
            sorted(
                Counter(
                    item.metadata.corruption_type or "positive"
                    for item in audit_examples
                ).items()
            )
        ),
        "audit_sha256": file_sha256(audit_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
