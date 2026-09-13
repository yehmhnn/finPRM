#!/usr/bin/env python3
"""QLoRA adaptation of the native Qwen process-reward head."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from finprm.models.qwen_prm import (
    MODEL_ID,
    MODEL_REVISION,
    NATIVE_SERIALIZER_VERSION,
    encode_record,
    load_qwen_prm,
    load_records,
    pad_encoded_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--evidence-mode", choices=("gold", "full"), default="gold")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.gradient_accumulation < 1:
        parser.error("epochs and gradient accumulation must be positive")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    records = load_records(args.train_data, args.max_examples)
    tokenizer, model = load_qwen_prm(args.model, "4bit")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_modules = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0 or trainable >= total:
        raise RuntimeError(f"invalid trainable parameter count: {trainable}/{total}")

    class ProcessDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            return encode_record(
                tokenizer, records[index], args.max_length, args.evidence_mode
            )

    def collate(items):
        batch = pad_encoded_examples(tokenizer, items)
        labels = torch.full_like(batch["input_ids"], -100)
        for row, item in enumerate(items):
            labels[row, item.reward_position] = item.label
        batch["labels"] = labels
        return batch

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ProcessDataset(),
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    updates_per_epoch = (len(loader) + args.gradient_accumulation - 1) // args.gradient_accumulation
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer_steps = 0
    loss_sum = 0.0
    loss_history = []
    epoch_losses = []
    window_loss_sum = 0.0
    window_loss_count = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch_index, batch in enumerate(loader):
            window_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model(**batch)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            loss_value = float(output.loss.detach().cpu())
            loss_sum += loss_value
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            window_loss_sum += loss_value
            window_loss_count += 1
            (output.loss / window_size).backward()
            is_update = (batch_index + 1) % args.gradient_accumulation == 0
            is_last = batch_index + 1 == len(loader)
            if is_update or is_last:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                loss_history.append(
                    {
                        "epoch": epoch + 1,
                        "optimizer_step": optimizer_steps,
                        "mean_microbatch_loss": window_loss_sum / window_loss_count,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
                window_loss_sum = 0.0
                window_loss_count = 0
        epoch_losses.append(epoch_loss_sum / epoch_loss_count)
        checkpoint = args.output / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)

    final_dir = args.output / "adapter"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    torch.cuda.synchronize()
    with (args.output / "training_loss.jsonl").open("w", encoding="utf-8") as stream:
        for item in loss_history:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "serializer_version": NATIVE_SERIALIZER_VERSION,
        "train_data": str(args.train_data),
        "train_examples": len(records),
        "epochs": args.epochs,
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.gradient_accumulation,
        "optimizer_steps": optimizer_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": warmup_steps,
        "scheduler": "linear",
        "warmup_ratio": args.warmup_ratio,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "optimizer_weight_decay": 0.01,
        "quantization": "nf4-double-quant-bf16-compute",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": list(target_modules),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "mean_microbatch_loss": loss_sum / (len(loader) * args.epochs),
        "initial_optimizer_step_loss": loss_history[0]["mean_microbatch_loss"],
        "final_optimizer_step_loss": loss_history[-1]["mean_microbatch_loss"],
        "epoch_mean_losses": epoch_losses,
        "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
        "seed": args.seed,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hostname": os.uname().nodename,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
