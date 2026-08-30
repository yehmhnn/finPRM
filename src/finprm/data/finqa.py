"""Minimal, strict loader for the official FinQA JSON splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union


class FinQASchemaError(ValueError):
    """Raised when a source record does not match the expected FinQA schema."""


@dataclass(frozen=True)
class FinQAExample:
    example_id: str
    question: str
    table: Tuple[Tuple[str, ...], ...]
    pre_text: Tuple[str, ...]
    post_text: Tuple[str, ...]
    supporting_facts: Tuple[str, ...]
    program: Optional[str]
    execution_answer: Any


def _strings(value: Any, field: str, example_id: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FinQASchemaError(f"{example_id}: {field} must be a list of strings")
    return tuple(value)


def _parse_record(record: Any, index: int) -> FinQAExample:
    if not isinstance(record, dict):
        raise FinQASchemaError(f"record {index}: expected an object")

    example_id = record.get("id")
    if not isinstance(example_id, str) or not example_id:
        raise FinQASchemaError(f"record {index}: missing non-empty id")

    qa = record.get("qa")
    if not isinstance(qa, dict):
        raise FinQASchemaError(f"{example_id}: qa must be an object")
    question = qa.get("question")
    if not isinstance(question, str) or not question:
        raise FinQASchemaError(f"{example_id}: qa.question must be non-empty")

    raw_table = record.get("table")
    if not isinstance(raw_table, list) or not raw_table:
        raise FinQASchemaError(f"{example_id}: table must be a non-empty list")
    rows = []
    width = None
    for row_index, row in enumerate(raw_table):
        if not isinstance(row, list) or not row or not all(isinstance(cell, str) for cell in row):
            raise FinQASchemaError(f"{example_id}: invalid table row {row_index}")
        width = len(row) if width is None else width
        if len(row) != width:
            raise FinQASchemaError(f"{example_id}: table rows have inconsistent widths")
        rows.append(tuple(row))

    program = qa.get("program")
    if program is not None and not isinstance(program, str):
        raise FinQASchemaError(f"{example_id}: qa.program must be text or null")
    raw_supporting = qa.get("gold_inds", {})
    if raw_supporting is None:
        raw_supporting = {}
    if not isinstance(raw_supporting, dict) or not all(
        isinstance(value, str) for value in raw_supporting.values()
    ):
        raise FinQASchemaError(f"{example_id}: qa.gold_inds must map IDs to text")

    return FinQAExample(
        example_id=example_id,
        question=question,
        table=tuple(rows),
        pre_text=_strings(record.get("pre_text", []), "pre_text", example_id),
        post_text=_strings(record.get("post_text", []), "post_text", example_id),
        supporting_facts=tuple(raw_supporting.values()),
        program=program,
        execution_answer=qa.get("exe_ans"),
    )


def load_split(path: Union[Path, str], limit: Optional[int] = None) -> Iterator[FinQAExample]:
    """Load and validate a FinQA split, yielding at most ``limit`` examples."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise FinQASchemaError(f"{source}: top-level JSON value must be a list")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    stop = len(records) if limit is None else min(limit, len(records))
    for index in range(stop):
        yield _parse_record(records[index], index)
