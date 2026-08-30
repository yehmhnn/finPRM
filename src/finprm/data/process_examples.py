"""Construct auditable binary process-verification examples from FinQA."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .finqa import FinQAExample
from .program import Operation, Scalar, execute_program, parse_program

VALIDATOR_VERSION = "finprm-validator-v1"

NUMERIC_OPERATOR_ALTERNATIVES: Mapping[str, Tuple[str, ...]] = {
    "add": ("subtract", "multiply", "divide"),
    "subtract": ("add", "multiply", "divide"),
    "multiply": ("add", "subtract", "divide"),
    "divide": ("add", "subtract", "multiply"),
    "table_max": ("table_min", "table_sum", "table_average"),
    "table_min": ("table_max", "table_sum", "table_average"),
    "table_sum": ("table_max", "table_min", "table_average"),
    "table_average": ("table_max", "table_min", "table_sum"),
}
ORDER_SENSITIVE_OPERATORS = frozenset({"subtract", "divide", "exp", "greater"})


@dataclass(frozen=True)
class ProcessInput:
    question: str
    pre_text: Tuple[str, ...]
    post_text: Tuple[str, ...]
    table: Tuple[Tuple[str, ...], ...]
    supporting_facts: Tuple[str, ...]
    prefix: Tuple[str, ...]
    candidate: str


@dataclass(frozen=True)
class ProcessTarget:
    label: int


@dataclass(frozen=True)
class ProcessMetadata:
    stable_id: str
    finqa_id: str
    split: str
    step_index: int
    gold_operation: str
    corruption_type: Optional[str]
    validator_version: str
    random_seed: int


@dataclass(frozen=True)
class ProcessExample:
    input: ProcessInput
    target: ProcessTarget
    metadata: ProcessMetadata

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMutation:
    operation: Operation
    corruption_type: str


@dataclass
class BuildResult:
    examples: List[ProcessExample]
    rejections: Counter[str]


def _stable_id(
    finqa_id: str, split: str, step_index: int, label: int, candidate: Operation
) -> str:
    payload = "|".join(
        [finqa_id, split, str(step_index), str(label), str(candidate)]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{split}-{digest}"


def _same_value(left: Scalar, right: Scalar) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _make_example(
    source: FinQAExample,
    split: str,
    operations: Sequence[Operation],
    step_index: int,
    candidate: Operation,
    label: int,
    corruption_type: Optional[str],
    seed: int,
) -> ProcessExample:
    gold = operations[step_index]
    prefix = tuple(str(operation) for operation in operations[:step_index])
    return ProcessExample(
        input=ProcessInput(
            question=source.question,
            pre_text=source.pre_text,
            post_text=source.post_text,
            table=source.table,
            supporting_facts=source.supporting_facts,
            prefix=prefix,
            candidate=str(candidate),
        ),
        target=ProcessTarget(label=label),
        metadata=ProcessMetadata(
            stable_id=_stable_id(
                source.example_id, split, step_index, label, candidate
            ),
            finqa_id=source.example_id,
            split=split,
            step_index=step_index,
            gold_operation=str(gold),
            corruption_type=corruption_type,
            validator_version=VALIDATOR_VERSION,
            random_seed=seed,
        ),
    )


def propose_mutations(operation: Operation) -> Tuple[CandidateMutation, ...]:
    """Create single-change candidates without deciding whether they are valid negatives."""
    mutations = []
    for replacement in NUMERIC_OPERATOR_ALTERNATIVES.get(operation.operator, ()):
        mutations.append(
            CandidateMutation(
                Operation(replacement, operation.arguments), "operator_substitution"
            )
        )
    left, right = operation.arguments
    if operation.operator in ORDER_SENSITIVE_OPERATORS and left != right:
        mutations.append(
            CandidateMutation(
                Operation(operation.operator, (right, left)), "operand_reversal"
            )
        )
    return tuple(mutations)


def _mutation_rank(seed: int, source_id: str, step: int, mutation: CandidateMutation) -> str:
    payload = f"{seed}|{source_id}|{step}|{mutation.corruption_type}|{mutation.operation}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_process_examples(
    source: FinQAExample,
    split: str,
    max_negatives_per_positive: int = 2,
    seed: int = 42,
) -> BuildResult:
    """Create positives and conservative, execution-validated negatives."""
    if source.program is None:
        return BuildResult([], Counter({"missing_gold_program": 1}))
    if max_negatives_per_positive < 0:
        raise ValueError("max_negatives_per_positive must be non-negative")

    operations = parse_program(source.program)
    gold_trace = execute_program(operations, source.table)
    if not gold_trace.valid:
        return BuildResult([], Counter({"invalid_gold_program": 1}))

    examples: List[ProcessExample] = []
    rejections: Counter[str] = Counter()
    for step_index, gold_operation in enumerate(operations):
        examples.append(
            _make_example(
                source,
                split,
                operations,
                step_index,
                gold_operation,
                1,
                None,
                seed,
            )
        )

        if max_negatives_per_positive == 0:
            continue

        accepted = []
        proposals = sorted(
            propose_mutations(gold_operation),
            key=lambda mutation: _mutation_rank(
                seed, source.example_id, step_index, mutation
            ),
        )
        for mutation in proposals:
            mutated_program = list(operations)
            mutated_program[step_index] = mutation.operation
            candidate_trace = execute_program(mutated_program, source.table)
            if not candidate_trace.valid:
                rejections[f"candidate_{candidate_trace.error_type}"] += 1
                continue
            gold_step_value = gold_trace.steps[step_index].value
            candidate_step_value = candidate_trace.steps[step_index].value
            if _same_value(gold_step_value, candidate_step_value):
                rejections["same_step_value"] += 1
                continue
            if _same_value(gold_trace.value, candidate_trace.value):
                rejections["same_final_answer"] += 1
                continue
            accepted.append(mutation)
            if len(accepted) == max_negatives_per_positive:
                break

        if not proposals:
            rejections["no_supported_mutation"] += 1
        elif len(accepted) < max_negatives_per_positive:
            rejections["insufficient_unambiguous_mutations"] += (
                max_negatives_per_positive - len(accepted)
            )
        for mutation in accepted:
            examples.append(
                _make_example(
                    source,
                    split,
                    operations,
                    step_index,
                    mutation.operation,
                    0,
                    mutation.corruption_type,
                    seed,
                )
            )
    return BuildResult(examples, rejections)


def build_split(
    sources: Iterable[FinQAExample],
    split: str,
    max_negatives_per_positive: int = 2,
    seed: int = 42,
) -> BuildResult:
    combined = BuildResult([], Counter())
    for source in sources:
        result = build_process_examples(
            source, split, max_negatives_per_positive, seed
        )
        combined.examples.extend(result.examples)
        combined.rejections.update(result.rejections)
    return combined
