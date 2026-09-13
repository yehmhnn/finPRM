#!/usr/bin/env python3
"""Freeze Train demonstration IDs for one evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finprm.retrieval import DEFAULT_K, DemonstrationRetriever
from finprm.retrieval.index import load_jsonl, save_retrieval_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("question", "joint"), required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    targets = load_jsonl(args.target_data)
    retriever = DemonstrationRetriever(args.index, args.embedding_model, args.mode)
    matches = retriever.retrieve_many(targets, k=args.k, batch_size=args.batch_size)
    plan = save_retrieval_plan(
        args.output,
        args.target_data,
        args.mode,
        args.k,
        retriever.manifest,
        targets,
        matches,
    )
    print(json.dumps({key: value for key, value in plan.items() if key != "rows"}, indent=2))
    print({"targets": len(plan["rows"])})


if __name__ == "__main__":
    main()
