"""Native input and scoring utilities for Qwen2.5-Math-PRM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
MODEL_REVISION = "0610740060112df12585d00a1c5f4624d2f59051"
STEP_SEPARATOR = "<extra_0>"
NATIVE_SERIALIZER_VERSION = "qwen-math-prm-finqa-v1"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class EncodedPRMExample:
    input_ids: List[int]
    attention_mask: List[int]
    reward_position: int
    label: int
    stable_id: str
    finqa_id: str
    metadata: Mapping[str, Any]
    original_tokens: int
    truncated_tokens: int


def load_records(path: Path, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
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
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise ValueError(f"{path}: no examples found")
    return records


def _require(mapping: Mapping[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        raise ValueError(f"{key} must be {kind.__name__}")
    return value


def _visible_parts(record: Mapping[str, Any], evidence_mode: str):
    process_input = _require(record, "input", dict)
    question = _require(process_input, "question", str)
    prefix = _require(process_input, "prefix", list)
    candidate = _require(process_input, "candidate", str)
    table = _require(process_input, "table", list)
    supporting = _require(process_input, "supporting_facts", list)
    if evidence_mode == "gold":
        narrative = supporting
    elif evidence_mode == "full":
        narrative = [
            *_require(process_input, "pre_text", list),
            *_require(process_input, "post_text", list),
        ]
    else:
        raise ValueError("evidence_mode must be 'gold' or 'full'")
    narrative_text = "\n".join(str(item) for item in narrative) or "<NONE>"
    table_text = "\n".join(" | ".join(str(cell) for cell in row) for row in table)
    return question, [str(item) for item in prefix], candidate, narrative_text, table_text


def _messages(
    question: str,
    prefix: Sequence[str],
    candidate: str,
    narrative: str,
    table_text: str,
    demonstrations: Optional[str] = None,
) -> List[Dict[str, str]]:
    target_query = (
        "Financial evidence text:\n"
        f"{narrative}\n\nFinancial table:\n{table_text}\n\n"
        f"Question:\n{question}"
    )
    query = (
        f"{demonstrations}\n\nTarget sample:\n{target_query}"
        if demonstrations
        else target_query
    )
    steps = [*prefix, candidate]
    response = "\n\n".join(
        f"Step {index}: {step}{STEP_SEPARATOR}"
        for index, step in enumerate(steps, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]


def step_separator_id(tokenizer) -> int:
    token_ids = tokenizer.encode(STEP_SEPARATOR, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{STEP_SEPARATOR} must map to one token, got {token_ids}")
    return int(token_ids[0])


def encode_record(
    tokenizer,
    record: Mapping[str, Any],
    max_length: int,
    evidence_mode: str,
    demonstrations: Optional[Sequence[Mapping[str, Any]]] = None,
):
    question, prefix, candidate, narrative, table_text = _visible_parts(record, evidence_mode)
    demonstration_text = None
    if demonstrations:
        from finprm.retrieval import format_demonstrations

        demonstration_text = format_demonstrations(demonstrations)

    def render(table: str):
        text = tokenizer.apply_chat_template(
            _messages(
                question,
                prefix,
                candidate,
                narrative,
                table,
                demonstration_text,
            ),
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    full_ids = render(table_text)
    original_tokens = len(full_ids)
    if original_tokens > max_length:
        table_ids = tokenizer.encode(table_text, add_special_tokens=False)
        low, high = 0, len(table_ids)
        while low < high:
            middle = (low + high + 1) // 2
            candidate_table = tokenizer.decode(table_ids[:middle], skip_special_tokens=False)
            if len(render(candidate_table)) <= max_length:
                low = middle
            else:
                high = middle - 1
        table_text = tokenizer.decode(table_ids[:low], skip_special_tokens=False)
        full_ids = render(table_text)
    if len(full_ids) > max_length:
        raise ValueError(
            "question, gold evidence text, prefix, and candidate exceed "
            f"max_length={max_length} even after table truncation"
        )

    separator = step_separator_id(tokenizer)
    positions = [index for index, token_id in enumerate(full_ids) if token_id == separator]
    expected_markers = len(prefix) + 1
    if len(positions) != expected_markers:
        raise ValueError(
            f"expected {expected_markers} reward markers, found {len(positions)}"
        )

    target = _require(record, "target", dict)
    label = _require(target, "label", int)
    if label not in {0, 1}:
        raise ValueError("target.label must be 0 or 1")
    metadata = _require(record, "metadata", dict)
    stable_id = _require(metadata, "stable_id", str)
    finqa_id = _require(metadata, "finqa_id", str)
    return EncodedPRMExample(
        input_ids=[int(item) for item in full_ids],
        attention_mask=[1] * len(full_ids),
        reward_position=positions[-1],
        label=label,
        stable_id=stable_id,
        finqa_id=finqa_id,
        metadata=metadata,
        original_tokens=original_tokens,
        truncated_tokens=original_tokens - len(full_ids),
    )


def pad_encoded_examples(tokenizer, examples: Sequence[EncodedPRMExample]):
    if not examples:
        raise ValueError("cannot collate an empty batch")
    features = [
        {"input_ids": item.input_ids, "attention_mask": item.attention_mask}
        for item in examples
    ]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    return batch


def load_qwen_prm(model_path: str, quantization: str, adapter_path: Optional[str] = None):
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=MODEL_REVISION, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    kwargs: Dict[str, Any] = {
        "revision": MODEL_REVISION,
        "trust_remote_code": True,
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization != "bf16":
        raise ValueError("quantization must be '4bit' or 'bf16'")
    model = AutoModel.from_pretrained(model_path, **kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model
