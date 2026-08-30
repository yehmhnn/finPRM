"""Parser and deterministic executor for FinQA's program language."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

Scalar = Union[float, str]

ARITHMETIC_OPERATORS = frozenset(
    {"add", "subtract", "multiply", "divide", "exp", "greater"}
)
TABLE_OPERATORS = frozenset(
    {"table_max", "table_min", "table_sum", "table_average"}
)
SUPPORTED_OPERATORS = ARITHMETIC_OPERATORS | TABLE_OPERATORS


class ProgramParseError(ValueError):
    """Raised when a FinQA program cannot be parsed unambiguously."""


@dataclass(frozen=True)
class Operation:
    operator: str
    arguments: Tuple[str, str]

    def __str__(self) -> str:
        return f"{self.operator}({self.arguments[0]}, {self.arguments[1]})"


@dataclass(frozen=True)
class ExecutionStep:
    index: int
    operation: Operation
    value: Scalar


@dataclass(frozen=True)
class ExecutionResult:
    valid: bool
    value: Optional[Scalar]
    steps: Tuple[ExecutionStep, ...]
    error_type: Optional[str] = None
    error_step: Optional[int] = None
    message: Optional[str] = None


_OPERATION = re.compile(r"\s*([a-z_]+)\(([^()]*)\)\s*(?:,\s*|$)")
_REFERENCE = re.compile(r"#(\d+)$")


def parse_program(program: str) -> Tuple[Operation, ...]:
    """Parse ``op(arg1, arg2), ...`` into typed operations."""
    if not isinstance(program, str) or not program.strip():
        raise ProgramParseError("program must be non-empty text")
    text = program.strip()
    if text.endswith("EOF"):
        text = text[:-3].rstrip(" ,")

    operations = []
    position = 0
    while position < len(text):
        match = _OPERATION.match(text, position)
        if match is None:
            raise ProgramParseError(f"invalid syntax near position {position}: {text[position:position + 30]!r}")
        operator, raw_arguments = match.groups()
        if operator not in SUPPORTED_OPERATORS:
            raise ProgramParseError(f"unsupported operator: {operator}")
        arguments = tuple(part.strip() for part in raw_arguments.split(","))
        if len(arguments) != 2 or not all(arguments):
            raise ProgramParseError(f"{operator} requires exactly two non-empty arguments")
        operations.append(Operation(operator, (arguments[0], arguments[1])))
        position = match.end()
    return tuple(operations)


def format_program(operations: Sequence[Operation]) -> str:
    """Serialize operations using FinQA's canonical sequential syntax."""
    return ", ".join(str(operation) for operation in operations)


def parse_number(text: str) -> float:
    """Convert FinQA numeric literals, percentages, and ``const_*`` tokens."""
    value = text.strip().replace(",", "").replace("$", "")
    if value.startswith("const_"):
        value = value[len("const_") :]
        if value == "m1":
            value = "-1"
    percentage = value.endswith("%")
    if percentage:
        value = value[:-1].strip()
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"not a FinQA number: {text!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"number must be finite: {text!r}")
    return number / 100.0 if percentage else number


def _table_rows(table: Sequence[Sequence[str]]) -> Mapping[str, Sequence[str]]:
    return {row[0].strip(): row[1:] for row in table if row}


def _parse_table_cell(cell: str) -> float:
    # This matches the official evaluator: remove currency markers and ignore
    # parenthesized annotations following the primary number.
    primary = cell.replace("$", "").strip().split("(", 1)[0].strip()
    return parse_number(primary)


def _resolve_numeric(argument: str, results: Sequence[Scalar]) -> float:
    reference = _REFERENCE.fullmatch(argument)
    if reference:
        index = int(reference.group(1))
        if index >= len(results):
            raise LookupError(f"reference {argument} does not name an earlier step")
        value = results[index]
        if not isinstance(value, float):
            raise TypeError(f"reference {argument} does not contain a number")
        return value
    return parse_number(argument)


def _invalid(
    steps: Sequence[ExecutionStep], error_type: str, step: int, message: str
) -> ExecutionResult:
    return ExecutionResult(False, None, tuple(steps), error_type, step, message)


def execute_program(
    program: Union[str, Sequence[Operation]], table: Sequence[Sequence[str]]
) -> ExecutionResult:
    """Execute a FinQA program and return a trace or a typed validation error."""
    try:
        operations = parse_program(program) if isinstance(program, str) else tuple(program)
    except ProgramParseError as error:
        return _invalid((), "parse_error", 0, str(error))

    values: list[Scalar] = []
    steps: list[ExecutionStep] = []
    rows = _table_rows(table)

    for index, operation in enumerate(operations):
        op = operation.operator
        left, right = operation.arguments
        try:
            if op in TABLE_OPERATORS:
                if left not in rows:
                    return _invalid(steps, "missing_table_row", index, f"unknown table row: {left!r}")
                try:
                    numbers = [_parse_table_cell(cell) for cell in rows[left]]
                except ValueError as error:
                    return _invalid(steps, "non_numeric_table_row", index, str(error))
                if not numbers:
                    return _invalid(steps, "empty_table_row", index, f"table row {left!r} has no values")
                if op == "table_max":
                    value: Scalar = max(numbers)
                elif op == "table_min":
                    value = min(numbers)
                elif op == "table_sum":
                    value = sum(numbers)
                else:
                    value = sum(numbers) / len(numbers)
            else:
                try:
                    arg1 = _resolve_numeric(left, values)
                    arg2 = _resolve_numeric(right, values)
                except LookupError as error:
                    return _invalid(steps, "invalid_reference", index, str(error))
                except TypeError as error:
                    return _invalid(steps, "type_mismatch", index, str(error))
                except ValueError as error:
                    return _invalid(steps, "invalid_number", index, str(error))

                if op == "add":
                    value = arg1 + arg2
                elif op == "subtract":
                    value = arg1 - arg2
                elif op == "multiply":
                    value = arg1 * arg2
                elif op == "divide":
                    if arg2 == 0:
                        return _invalid(steps, "division_by_zero", index, "division by zero")
                    value = arg1 / arg2
                elif op == "exp":
                    value = arg1**arg2
                elif op == "greater":
                    value = "yes" if arg1 > arg2 else "no"
                else:
                    return _invalid(steps, "unsupported_operator", index, op)
        except (ArithmeticError, OverflowError) as error:
            return _invalid(steps, "arithmetic_error", index, str(error))

        if isinstance(value, float) and not math.isfinite(value):
            return _invalid(steps, "non_finite_result", index, f"{op} produced {value}")
        values.append(value)
        steps.append(ExecutionStep(index, operation, value))

    if not steps:
        return _invalid((), "empty_program", 0, "program has no operations")
    final = values[-1]
    if isinstance(final, float):
        final = round(final, 5)
    return ExecutionResult(True, final, tuple(steps))
