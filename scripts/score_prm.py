#!/usr/bin/env python3
"""Score FinPRM examples with the native Qwen process-reward head."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from finprm.evaluation import binary_metrics, select_threshold
from finprm.models.qwen_prm import (
    MODEL_ID,
    MODEL_REVISION,
    NATIVE_SERIALIZER_VERSION,
    encode_record,
    load_qwen_prm,
    load_records,
    pad_encoded_examples,
)
from finprm.retrieval.index import file_sha256, load_jsonl


def load_retrieval_plan(path, data_path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if file_sha256(data_path) != plan["target_sha256"]:
        raise ValueError("retrieval plan target checksum does not match --data")
    source_path = Path(plan["index_source_path"])
    if file_sha256(source_path) != plan["index_source_sha256"]:
        raise ValueError("retrieval plan Train source checksum mismatch")
    source = {
        record["metadata"]["stable_id"]: record
        for record in load_jsonl(source_path)
    }
    by_target = {}
    train_finqa_ids = {record["metadata"]["finqa_id"] for record in source.values()}
    for row in plan["rows"]:
        if row["target_finqa_id"] in train_finqa_ids:
            raise ValueError("target FinQA ID appears in retrieval Train source")
        by_target[row["target_stable_id"]] = [
            source[item["stable_id"]] for item in row["demonstrations"]
        ]
    return plan, by_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter")
    parser.add_argument("--quantization", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--evidence-mode", choices=("gold", "full"), default="gold")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--retrieval-plan", type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native PRM scoring")
    records = load_records(args.data, args.max_examples)
    retrieval_plan = None
    demonstrations = {}
    if args.retrieval_plan:
        if args.evidence_mode != "gold":
            raise ValueError("retrieval experiment is frozen to evidence_mode=gold")
        retrieval_plan, demonstrations = load_retrieval_plan(args.retrieval_plan, args.data)
        missing = [
            record["metadata"]["stable_id"]
            for record in records
            if record["metadata"]["stable_id"] not in demonstrations
        ]
        if missing:
            raise ValueError(f"retrieval plan is missing {len(missing)} requested targets")
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_qwen_prm(args.model, args.quantization, args.adapter)
    model.eval()
    encoded = [
        encode_record(
            tokenizer,
            record,
            args.max_length,
            args.evidence_mode,
            demonstrations=demonstrations.get(record["metadata"]["stable_id"]),
        )
        for record in records
    ]
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(encoded), args.batch_size):
            items = encoded[offset : offset + args.batch_size]
            batch = pad_encoded_examples(tokenizer, items)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            logits = model(**batch).logits
            for row, item in enumerate(items):
                reward_logits = logits[row, item.reward_position].float()
                probability = torch.softmax(reward_logits, dim=-1)[1]
                score = float(probability.cpu())
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(f"invalid score for {item.stable_id}: {score}")
                predictions.append(
                    {
                        "stable_id": item.stable_id,
                        "finqa_id": item.finqa_id,
                        "label": item.label,
                        "p_correct": score,
                        "step_index": item.metadata.get("step_index"),
                        "program_length": item.metadata.get("program_length"),
                        "corruption_type": item.metadata.get("corruption_type"),
                        "original_tokens": item.original_tokens,
                        "truncated_tokens": item.truncated_tokens,
                    }
                )

    labels = [item["label"] for item in predictions]
    scores = [item["p_correct"] for item in predictions]
    score_distributions = {}
    for label in (0, 1):
        values = [score for item_label, score in zip(labels, scores) if item_label == label]
        score_distributions[str(label)] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    threshold = select_threshold(labels, scores) if args.select_threshold else args.threshold
    metrics = binary_metrics(labels, scores, threshold)
    metrics.update(
        {
            "examples": len(predictions),
            "elapsed_seconds": time.perf_counter() - started,
            "model": args.model,
            "model_revision": MODEL_REVISION,
            "adapter": args.adapter,
            "quantization": args.quantization,
            "evidence_mode": args.evidence_mode,
            "max_length": args.max_length,
            "serializer_version": NATIVE_SERIALIZER_VERSION,
            "truncated_examples": sum(item["truncated_tokens"] > 0 for item in predictions),
            "max_original_tokens": max(item["original_tokens"] for item in predictions),
            "retrieval_plan": str(args.retrieval_plan) if args.retrieval_plan else None,
            "retrieval_mode": retrieval_plan["mode"] if retrieval_plan else None,
            "retrieval_k": retrieval_plan["k"] if retrieval_plan else 0,
            "p_correct_distribution": score_distributions,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
