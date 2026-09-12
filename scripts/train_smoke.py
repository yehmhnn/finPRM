#!/usr/bin/env python3
"""Train, save, reload, and evaluate a tiny binary FinPRM classifier."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml

from finprm.models.serialization import (
    SERIALIZER_VERSION,
    grouped_train_eval_split,
    load_process_jsonl,
)


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def macro_f1(predictions, labels) -> float:
    scores = []
    for label in (0, 1):
        true_positive = sum(p == label and y == label for p, y in zip(predictions, labels))
        false_positive = sum(p == label and y != label for p, y in zip(predictions, labels))
        false_negative = sum(p != label and y == label for p, y in zip(predictions, labels))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate(torch, model, loader, device: str):
    model.eval()
    predictions, labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, target = (item.to(device) for item in batch)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target,
            )
            losses.append(output.loss.item())
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(target.cpu().tolist())
    accuracy = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": accuracy,
        "macro_f1": macro_f1(predictions, labels),
        "examples": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed = int(config["runtime"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(torch, config["runtime"]["device"])
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    data_config = config["data"]
    examples = load_process_jsonl(
        data_config["input"], data_config.get("evidence_mode", "gold")
    )
    max_examples = data_config.get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    train_examples, eval_examples = grouped_train_eval_split(
        examples, float(data_config.get("eval_fraction", 0.2)), seed
    )

    model_config = config["model"]
    checkpoint = model_config["checkpoint"]
    revision = model_config.get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        revision=revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    max_length = int(model_config["max_length"])

    def tensor_dataset(items):
        encoded = tokenizer(
            [item.text for item in items],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([item.label for item in items], dtype=torch.long)
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)

    training_config = config["training"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        tensor_dataset(train_examples),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    eval_loader = DataLoader(
        tensor_dataset(eval_examples),
        batch_size=int(training_config["batch_size"]),
    )
    optimizer = AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 5e-5)))
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    started = time.perf_counter()
    model.train()
    step_count = 0
    for _ in range(int(training_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (input_ids, attention_mask, labels) in enumerate(train_loader):
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            (output.loss / accumulation_steps).backward()
            is_update = (batch_index + 1) % accumulation_steps == 0
            is_last = batch_index + 1 == len(train_loader)
            if is_update or is_last:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

    metrics = evaluate(torch, model, eval_loader, device)
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    model.eval()
    sample = next(iter(eval_loader))
    with torch.no_grad():
        before = model(
            input_ids=sample[0].to(device), attention_mask=sample[1].to(device)
        ).logits.cpu()
    reloaded = AutoModelForSequenceClassification.from_pretrained(output_dir / "model").to(device)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(
            input_ids=sample[0].to(device), attention_mask=sample[1].to(device)
        ).logits.cpu()
    reload_max_difference = (before - after).abs().max().item()
    if reload_max_difference > 1e-5:
        raise RuntimeError(f"saved/reloaded logits differ by {reload_max_difference}")

    metrics.update(
        {
            "pipeline_smoke_test": True,
            "device": device,
            "checkpoint": checkpoint,
            "revision": revision,
            "serializer_version": SERIALIZER_VERSION,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "training_steps": step_count,
            "gradient_accumulation_steps": accumulation_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "reload_max_logit_difference": reload_max_difference,
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
