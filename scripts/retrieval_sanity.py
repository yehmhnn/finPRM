#!/usr/bin/env python3
"""Create retrieval examples and verify split and Qwen prompt constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from finprm.models.qwen_prm import MODEL_REVISION, _messages, _visible_parts, encode_record
from finprm.retrieval import (
    RETRIEVAL_MODES,
    DemonstrationRetriever,
    format_demonstrations,
)
from finprm.retrieval.index import file_sha256, load_jsonl, save_retrieval_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()

    dev = load_jsonl(args.dev_data)
    test = load_jsonl(args.test_data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model, revision=MODEL_REVISION, trust_remote_code=True,
        local_files_only=True,
    )
    report = {
        "k_values": args.k_values,
        "evidence_mode": "gold",
        "max_length": args.max_length,
        "targets": {
            "dev": {"path": str(args.dev_data.resolve()), "sha256": file_sha256(args.dev_data)},
            "test": {"path": str(args.test_data.resolve()), "sha256": file_sha256(args.test_data)},
        },
        "modes": {},
    }

    def raw_token_count(target, matches):
        question, prefix, candidate, narrative, table_text = _visible_parts(target, "gold")
        demo_text = format_demonstrations([match["record"] for match in matches])
        rendered = tokenizer.apply_chat_template(
            _messages(
                question, prefix, candidate, narrative, table_text, demo_text
            ),
            tokenize=False,
            add_generation_prompt=False,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    def percentile(values, quantile):
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(quantile * len(ordered) + 0.999999) - 1))
        return ordered[index]

    for mode in RETRIEVAL_MODES:
        retriever = DemonstrationRetriever(args.index, args.embedding_model, mode)
        targets = [*dev, *test]
        max_k = max(args.k_values)
        retrieved = retriever.retrieve_many(targets, k=max_k)
        shown = []
        for target, matches in zip(dev[: args.sample_count], retrieved[: args.sample_count]):
            shown.append(
                {
                    "target": target,
                    "demonstrations": [
                        {
                            "similarity": match["similarity"],
                            "record": match["record"],
                        }
                        for match in matches
                    ],
                }
            )
        report["modes"][mode] = {
            "target_examples_checked": len(targets),
            "statistics": {},
            "examples": shown,
        }
        for k in args.k_values:
            report["modes"][mode]["statistics"][str(k)] = {}
            for split_name, split_records, split_matches, split_path in (
                ("dev", dev, retrieved[: len(dev)], args.dev_data),
                ("test", test, retrieved[len(dev) :], args.test_data),
            ):
                limited_matches = [matches[:k] for matches in split_matches]
                save_retrieval_plan(
                    args.output.parent / f"{mode}-k{k}-{split_name}-plan.json",
                    split_path,
                    mode,
                    k,
                    retriever.manifest,
                    split_records,
                    limited_matches,
                )
                lengths = [
                    raw_token_count(target, matches)
                    for target, matches in zip(split_records, limited_matches)
                ]
                over_limit = [
                    (target, matches)
                    for target, matches, length in zip(
                        split_records, limited_matches, lengths
                    )
                    if length > args.max_length
                ]
                unencodable = 0
                for target, matches in over_limit:
                    try:
                        encode_record(
                            tokenizer,
                            target,
                            args.max_length,
                            "gold",
                            demonstrations=[match["record"] for match in matches],
                        )
                    except ValueError:
                        unencodable += 1
                report["modes"][mode]["statistics"][str(k)][split_name] = {
                    "examples": len(lengths),
                    "mean": sum(lengths) / len(lengths),
                    "p95": percentile(lengths, 0.95),
                    "p99": percentile(lengths, 0.99),
                    "max": max(lengths),
                    "over_2048_count": len(over_limit),
                    "over_2048_percent": 100.0 * len(over_limit) / len(lengths),
                    "unencodable_after_table_truncation": unencodable,
                }

    index_manifest = json.loads((args.index / "manifest.json").read_text())
    train_records = load_jsonl(Path(index_manifest["source_path"]))
    train_ids = {record["metadata"]["finqa_id"] for record in train_records}
    dev_ids = {record["metadata"]["finqa_id"] for record in dev}
    test_ids = {record["metadata"]["finqa_id"] for record in test}
    report["leakage"] = {
        "train_dev_overlap": len(train_ids & dev_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "dev_test_overlap": len(dev_ids & test_ids),
        "pass": not (train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids),
    }
    report["index_manifest"] = index_manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "modes"}, indent=2))
    for mode, values in report["modes"].items():
        print(mode, values["statistics"])


if __name__ == "__main__":
    main()
