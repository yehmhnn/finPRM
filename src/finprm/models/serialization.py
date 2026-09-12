"""Stable text serialization for binary FinPRM classifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

SERIALIZER_VERSION = "finprm-serializer-v1"


@dataclass(frozen=True)
class SerializedExample:
    stable_id: str
    finqa_id: str
    text: str
    label: int


def _require(mapping: Mapping[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"{key} must be {expected_type.__name__}")
    return value


def _table_text(table: Sequence[Sequence[str]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def serialize_input(process_input: Mapping[str, Any], evidence_mode: str = "gold") -> str:
    """Convert model-visible fields into a fixed, label-free prompt."""
    if evidence_mode not in {"gold", "full"}:
        raise ValueError("evidence_mode must be 'gold' or 'full'")
    question = _require(process_input, "question", str)
    candidate = _require(process_input, "candidate", str)
    table = _require(process_input, "table", list)
    prefix = _require(process_input, "prefix", list)
    supporting = _require(process_input, "supporting_facts", list)

    if evidence_mode == "gold":
        narrative = supporting
    else:
        pre_text = _require(process_input, "pre_text", list)
        post_text = _require(process_input, "post_text", list)
        narrative = [*pre_text, *post_text]

    sections = [
        "[EVIDENCE TEXT]",
        "\n".join(str(item) for item in narrative) or "<NONE>",
        "[EVIDENCE TABLE]",
        _table_text(table),
        "[QUESTION]",
        question,
        "[CORRECT PREFIX]",
        "\n".join(str(item) for item in prefix) or "<START>",
        "[CANDIDATE NEXT OPERATION]",
        candidate,
        "[TASK]",
        "Classify the candidate as CORRECT or INCORRECT.",
    ]
    return "\n".join(sections)


def load_process_jsonl(
    path: Union[Path, str], evidence_mode: str = "gold"
) -> List[SerializedExample]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                process_input = _require(record, "input", dict)
                target = _require(record, "target", dict)
                metadata = _require(record, "metadata", dict)
                label = _require(target, "label", int)
                if label not in {0, 1}:
                    raise ValueError("label must be 0 or 1")
                examples.append(
                    SerializedExample(
                        stable_id=_require(metadata, "stable_id", str),
                        finqa_id=_require(metadata, "finqa_id", str),
                        text=serialize_input(process_input, evidence_mode),
                        label=label,
                    )
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path}: no process examples found")
    return examples


def grouped_train_eval_split(
    examples: Sequence[SerializedExample], eval_fraction: float, seed: int
) -> Tuple[List[SerializedExample], List[SerializedExample]]:
    """Split by FinQA ID so steps from one question cannot leak across sets."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    train, evaluation = [], []
    boundary = int(eval_fraction * 10_000)
    for example in examples:
        digest = hashlib.sha256(f"{seed}|{example.finqa_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (evaluation if bucket < boundary else train).append(example)
    if not train or not evaluation:
        raise ValueError("grouped split produced an empty partition; use more source questions")
    return train, evaluation

