"""Construct auditable binary process-verification examples from FinQA."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .finqa import FinQAExample
from .program import Operation, Scalar, execute_program, parse_number, parse_program

VALIDATOR_VERSION = "finprm-validator-v2"

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
SCALE_CONSTANTS = frozenset({"const_100", "const_1000", "const_10000", "const_100000", "const_1000000", "const_1000000000"})
NUMBER_PATTERN = re.compile(r"(?<![#\w])[-+]?(?:\d[\d,]*\.?\d*|\.\d+)%?")
MAX_CONTEXT_REPLACEMENTS_PER_ARGUMENT = 8


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
    corruption_details: Optional[Dict[str, str]]
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
    details: Tuple[Tuple[str, str], ...] = ()


@dataclass
class BuildResult:
    examples: List[ProcessExample]
    rejections: Counter[str]


def _stable_id(
    finqa_id: str,
    split: str,
    step_index: int,
    label: int,
    candidate: Operation,
    corruption_type: Optional[str],
) -> str:
    payload = "|".join(
        [
            finqa_id,
            split,
            str(step_index),
            str(label),
            str(candidate),
            corruption_type or "positive",
        ]
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
    corruption_details: Optional[Dict[str, str]] = None,
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
                source.example_id,
                split,
                step_index,
                label,
                candidate,
                corruption_type,
            ),
            finqa_id=source.example_id,
            split=split,
            step_index=step_index,
            gold_operation=str(gold),
            corruption_type=corruption_type,
            corruption_details=corruption_details,
            validator_version=VALIDATOR_VERSION,
            random_seed=seed,
        ),
    )


def _numbers_in_text(text: str) -> Tuple[str, ...]:
    found = []
    for match in NUMBER_PATTERN.finditer(text):
        token = match.group(0).replace(",", "")
        try:
            parse_number(token)
        except ValueError:
            continue
        if token not in found:
            found.append(token)
    return tuple(found)


def _different_numeric_value(left: str, right: str) -> bool:
    try:
        return not math.isclose(
            parse_number(left), parse_number(right), rel_tol=1e-9, abs_tol=1e-12
        )
    except ValueError:
        return left != right


def _context_replacements(
    source: FinQAExample, original: str
) -> Tuple[Tuple[str, str], ...]:
    """Find other numbers beside ``original`` in the same local evidence unit."""
    units = [("question", source.question)]
    units.extend(
        (f"supporting_fact:{index}", text)
        for index, text in enumerate(source.supporting_facts)
    )
    units.extend(
        (f"table_row:{index}", " | ".join(row))
        for index, row in enumerate(source.table)
    )
    units.extend(
        (f"pre_text:{index}", text) for index, text in enumerate(source.pre_text)
    )
    units.extend(
        (f"post_text:{index}", text) for index, text in enumerate(source.post_text)
    )

    candidates = set()
    for location, text in units:
        numbers = _numbers_in_text(text)
        if not any(not _different_numeric_value(original, value) for value in numbers):
            continue
        for value in numbers:
            if _different_numeric_value(original, value):
                candidates.add((value, location))
    return tuple(sorted(candidates))


def propose_mutations(
    operation: Operation,
    source: Optional[FinQAExample] = None,
    step_index: Optional[int] = None,
    seed: int = 42,
) -> Tuple[CandidateMutation, ...]:
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

    # Explicit FinQA scale markers permit a mechanically attributable unit error.
    for argument_index, argument in enumerate(operation.arguments):
        replacement = None
        if argument.endswith("%"):
            replacement = argument[:-1]
        elif argument in SCALE_CONSTANTS:
            replacement = "const_1"
        if replacement and _different_numeric_value(argument, replacement):
            arguments = list(operation.arguments)
            arguments[argument_index] = replacement
            mutations.append(
                CandidateMutation(
                    Operation(operation.operator, tuple(arguments)),
                    "unit_scale_mismatch",
                    (("argument_index", str(argument_index)), ("original", argument), ("replacement", replacement)),
                )
            )

    if source is not None:
        for argument_index, argument in enumerate(operation.arguments):
            if argument.startswith("#") or argument.startswith("const_"):
                continue
            replacements = list(_context_replacements(source, argument))
            replacements.sort(
                key=lambda item: hashlib.sha256(
                    f"{seed}|{source.example_id}|{step_index}|{argument_index}|{item[0]}|{item[1]}".encode("utf-8")
                ).hexdigest()
            )
            for replacement, location in replacements[:MAX_CONTEXT_REPLACEMENTS_PER_ARGUMENT]:
                arguments = list(operation.arguments)
                arguments[argument_index] = replacement
                mutations.append(
                    CandidateMutation(
                        Operation(operation.operator, tuple(arguments)),
                        "entity_context_swap",
                        (("argument_index", str(argument_index)), ("original", argument), ("replacement", replacement), ("replacement_source", location)),
                    )
                )

    if step_index is not None:
        for argument_index, argument in enumerate(operation.arguments):
            if argument.startswith("#"):
                replacement = f"#{step_index}"
                if replacement != argument:
                    arguments = list(operation.arguments)
                    arguments[argument_index] = replacement
                    mutations.append(
                        CandidateMutation(
                            Operation(operation.operator, tuple(arguments)),
                            "dangling_reference",
                            (("argument_index", str(argument_index)), ("original", argument), ("replacement", replacement)),
                        )
                    )

    # A candidate can sometimes be proposed by two rules. Keep the more explicit
    # first attribution so labels and per-type evaluation remain unambiguous.
    unique = {}
    for mutation in mutations:
        unique.setdefault(str(mutation.operation), mutation)
    return tuple(unique.values())


def _mutation_rank(seed: int, source_id: str, step: int, mutation: CandidateMutation) -> str:
    payload = f"{seed}|{source_id}|{step}|{mutation.corruption_type}|{mutation.operation}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced_mutation_order(
    mutations: Sequence[CandidateMutation], seed: int, source_id: str, step: int
) -> Tuple[CandidateMutation, ...]:
    """Put the best candidate from each category before additional candidates."""
    groups: Dict[str, List[CandidateMutation]] = {}
    for mutation in mutations:
        groups.setdefault(mutation.corruption_type, []).append(mutation)
    for group in groups.values():
        group.sort(key=lambda item: _mutation_rank(seed, source_id, step, item))
    categories = sorted(
        groups,
        key=lambda name: hashlib.sha256(
            f"{seed}|{source_id}|{step}|category|{name}".encode("utf-8")
        ).hexdigest(),
    )
    first = [groups[name].pop(0) for name in categories]
    remainder = sorted(
        (item for group in groups.values() for item in group),
        key=lambda item: _mutation_rank(seed, source_id, step, item),
    )
    return tuple(first + remainder)


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
        proposals = _balanced_mutation_order(
            propose_mutations(gold_operation, source, step_index, seed),
            seed,
            source.example_id,
            step_index,
        )
        for mutation in proposals:
            mutated_program = list(operations)
            mutated_program[step_index] = mutation.operation
            candidate_trace = execute_program(mutated_program, source.table)
            if not candidate_trace.valid:
                if (
                    mutation.corruption_type == "dangling_reference"
                    and candidate_trace.error_type == "invalid_reference"
                    and candidate_trace.error_step == step_index
                ):
                    accepted.append(mutation)
                    if len(accepted) == max_negatives_per_positive:
                        break
                    continue
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
                    dict(mutation.details),
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
