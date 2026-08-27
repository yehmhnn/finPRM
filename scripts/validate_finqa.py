#!/usr/bin/env python3
"""Validate FinQA schemas and execute gold programs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from finprm.data import execute_program, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("--limit", type=int)
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
            if result.value == example.execution_answer:
                answer_matches += 1
            elif len(mismatches) < 5:
                mismatches.append(
                    (example.example_id, result.value, example.execution_answer)
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
    for example_id, actual, expected in mismatches:
        print(f"mismatch={example_id}: executor={actual!r}, gold={expected!r}")


if __name__ == "__main__":
    main()
