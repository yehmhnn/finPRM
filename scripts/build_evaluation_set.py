#!/usr/bin/env python3
"""Create a deterministic one-positive/one-negative evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from finprm.data import select_primary_evaluation_pairs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no examples found")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.source)
    selected, summary = select_primary_evaluation_pairs(records, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary.update(
        {
            "source": str(args.source),
            "source_sha256": file_sha256(args.source),
            "examples_sha256": file_sha256(examples_path),
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
