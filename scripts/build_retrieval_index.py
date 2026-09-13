#!/usr/bin/env python3
"""Build the frozen Train-only dense demonstration index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finprm.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    manifest = build_index(
        args.train_data,
        args.output,
        args.embedding_model,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
