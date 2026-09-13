"""Deterministic construction of a label-balanced primary evaluation set."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Tuple

EVALUATION_SAMPLER_VERSION = "finprm-eval-pairs-v1"


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("record metadata must be an object")
    return metadata


def _label(record: Mapping[str, Any]) -> int:
    target = record.get("target")
    if not isinstance(target, dict) or target.get("label") not in {0, 1}:
        raise ValueError("record target.label must be 0 or 1")
    return int(target["label"])


def _key(record: Mapping[str, Any]) -> Tuple[str, int]:
    metadata = _metadata(record)
    finqa_id = metadata.get("finqa_id")
    step_index = metadata.get("step_index")
    if not isinstance(finqa_id, str) or not isinstance(step_index, int):
        raise ValueError("metadata must contain string finqa_id and integer step_index")
    return finqa_id, step_index


def _rank(seed: int, *parts: object) -> str:
    payload = "|".join((str(seed), EVALUATION_SAMPLER_VERSION, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_primary_evaluation_pairs(
    records: Iterable[Mapping[str, Any]], seed: int
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Keep one positive and one deterministic negative for every eligible step.

    Steps with fewer available corruption families are assigned first.  For a
    flexible step, the least-used available corruption family is selected.  The
    method preserves label balance while reducing domination by common negative
    generators without fabricating examples for unsupported error types.
    """
    positives: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    negatives: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    input_count = 0
    for record in records:
        input_count += 1
        key = _key(record)
        if _label(record) == 1:
            if key in positives:
                raise ValueError(f"multiple positive examples for {key}")
            positives[key] = record
        else:
            corruption_type = _metadata(record).get("corruption_type")
            if not isinstance(corruption_type, str) or not corruption_type:
                raise ValueError(f"negative example for {key} lacks corruption_type")
            negatives.setdefault(key, []).append(record)

    eligible = [key for key in positives if negatives.get(key)]
    eligible.sort(
        key=lambda key: (
            len({_metadata(item)["corruption_type"] for item in negatives[key]}),
            _rank(seed, *key),
        )
    )

    type_counts: Counter[str] = Counter()
    chosen: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for key in eligible:
        by_type: Dict[str, List[Mapping[str, Any]]] = {}
        for item in negatives[key]:
            by_type.setdefault(str(_metadata(item)["corruption_type"]), []).append(item)
        minimum = min(type_counts[name] for name in by_type)
        candidates = [name for name in by_type if type_counts[name] == minimum]
        chosen_type = min(candidates, key=lambda name: _rank(seed, *key, name))
        chosen_item = min(
            by_type[chosen_type],
            key=lambda item: _rank(seed, _metadata(item).get("stable_id", "")),
        )
        chosen[key] = chosen_item
        type_counts[chosen_type] += 1

    ordered_keys = sorted(eligible, key=lambda key: _rank(seed, "output", *key))
    selected: List[Mapping[str, Any]] = []
    for key in ordered_keys:
        selected.extend((positives[key], chosen[key]))

    summary = {
        "sampler_version": EVALUATION_SAMPLER_VERSION,
        "seed": seed,
        "input_examples": input_count,
        "source_steps": len(positives),
        "paired_steps": len(ordered_keys),
        "excluded_steps_without_negative": len(positives) - len(ordered_keys),
        "output_examples": len(selected),
        "positive_examples": len(ordered_keys),
        "negative_examples": len(ordered_keys),
        "negative_type_counts": dict(sorted(type_counts.items())),
    }
    return selected, summary
