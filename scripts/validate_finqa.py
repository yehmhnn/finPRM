#!/usr/bin/env python3
"""Validate FinQA schemas and execute gold programs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from finprm.data import execute_program, execution_values_equal, load_split


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    total = 0
    valid = 0
    answer_matches = 0
    operation_counts: Counter[str] = Counter()
    total_steps = 0
    errors: Counter[str] = Counter()
    mismatches = []
    for example in load_split(args.split, args.limit):
        total += 1
        if example.program is None:
            errors["missing_program"] += 1
            continue
        result = execute_program(example.program, example.table)
        if result.valid:
            valid += 1
            total_steps += len(result.steps)
            operation_counts.update(step.operation.operator for step in result.steps)
            if execution_values_equal(result.value, example.execution_answer):
                answer_matches += 1
            else:
                mismatches.append(
                    {
                        "finqa_id": example.example_id,
                        "executor_value": result.value,
                        "expected_value": example.execution_answer,
                    }
                )
        else:
            errors[result.error_type or "unknown"] += 1

    print(f"examples={total}")
    print(f"executable={valid}")
    print(f"invalid={total - valid}")
    print(f"gold_answer_matches={answer_matches}")
    print(f"gold_answer_mismatches={valid - answer_matches}")
    print(f"gold_program_steps={total_steps}")
    print("operators=" + ",".join(f"{name}:{count}" for name, count in operation_counts.most_common()))
    for name, count in errors.most_common():
        print(f"{name}={count}")
    for mismatch in mismatches[:5]:
        print(
            "mismatch="
            f"{mismatch['finqa_id']}: executor={mismatch['executor_value']!r}, "
            f"gold={mismatch['expected_value']!r}"
        )

    if args.output is not None:
        report = {
            "source": str(args.split),
            "source_sha256": file_sha256(args.split),
            "limit": args.limit,
            "examples": total,
            "executable": valid,
            "invalid": total - valid,
            "gold_answer_matches": answer_matches,
            "gold_answer_mismatches": valid - answer_matches,
            "gold_program_steps": total_steps,
            "operators": dict(sorted(operation_counts.items())),
            "errors": dict(sorted(errors.items())),
            "mismatches": mismatches,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
