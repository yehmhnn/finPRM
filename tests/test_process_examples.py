import unittest

from finprm.data.finqa import FinQAExample
from finprm.data.process_examples import (
    build_process_examples,
    propose_mutations,
)
from finprm.data.program import Operation


def source(program="subtract(1200, 1000), divide(#0, 1000)"):
    return FinQAExample(
        example_id="report/page-1",
        question="What was the percentage change?",
        table=(("", "2022", "2021"), ("revenue", "1200", "1000")),
        pre_text=("Revenue increased.",),
        post_text=(),
        supporting_facts=("Revenue was 1200 and 1000.",),
        program=program,
        execution_answer=0.2,
    )


class ProcessExampleTests(unittest.TestCase):
    def test_positive_records_have_correct_prefixes(self):
        result = build_process_examples(source(), "train", 0, seed=42)
        self.assertEqual(2, len(result.examples))
        first, second = result.examples
        self.assertEqual(1, first.target.label)
        self.assertEqual((), first.input.prefix)
        self.assertEqual(("subtract(1200, 1000)",), second.input.prefix)
        self.assertEqual("divide(#0, 1000)", second.input.candidate)
        self.assertIsNone(second.metadata.corruption_type)

    def test_operator_and_reversal_proposals_change_one_component(self):
        operation = Operation("subtract", ("1200", "1000"))
        mutations = propose_mutations(operation)
        self.assertIn("operand_reversal", {item.corruption_type for item in mutations})
        self.assertIn("operator_substitution", {item.corruption_type for item in mutations})
        for mutation in mutations:
            if mutation.corruption_type == "operator_substitution":
                self.assertEqual(operation.arguments, mutation.operation.arguments)
            else:
                self.assertEqual(operation.operator, mutation.operation.operator)

    def test_negatives_change_step_and_final_values(self):
        result = build_process_examples(source(), "train", 2, seed=42)
        negatives = [item for item in result.examples if item.target.label == 0]
        self.assertGreaterEqual(len(negatives), 2)
        self.assertTrue(all(item.input.candidate != item.metadata.gold_operation for item in negatives))
        self.assertTrue(all(item.metadata.corruption_type for item in negatives))

    def test_generation_is_deterministic(self):
        first = build_process_examples(source(), "train", 2, seed=7)
        second = build_process_examples(source(), "train", 2, seed=7)
        self.assertEqual(
            [item.to_dict() for item in first.examples],
            [item.to_dict() for item in second.examples],
        )
        stable_ids = [item.metadata.stable_id for item in first.examples]
        self.assertEqual(len(stable_ids), len(set(stable_ids)))

    def test_commutative_equal_arguments_are_not_reversed(self):
        operation = Operation("subtract", ("5", "5"))
        reversals = [
            item for item in propose_mutations(operation)
            if item.corruption_type == "operand_reversal"
        ]
        self.assertEqual([], reversals)

    def test_context_swap_uses_a_local_evidence_unit(self):
        mutations = propose_mutations(
            Operation("subtract", ("1200", "1000")), source(), 0, seed=42
        )
        swaps = [item for item in mutations if item.corruption_type == "entity_context_swap"]
        self.assertTrue(swaps)
        for swap in swaps:
            details = dict(swap.details)
            self.assertIn("replacement_source", details)
            self.assertTrue(
                details["replacement_source"].startswith(("supporting_fact:", "table_row:"))
            )

    def test_scale_mismatch_removes_explicit_multiplier(self):
        mutations = propose_mutations(Operation("multiply", ("#0", "const_100")))
        scale_errors = [
            item for item in mutations if item.corruption_type == "unit_scale_mismatch"
        ]
        self.assertEqual("multiply(#0, const_1)", str(scale_errors[0].operation))

    def test_dangling_reference_points_to_current_unavailable_step(self):
        mutations = propose_mutations(Operation("divide", ("#0", "1000")), step_index=1)
        dangling = [
            item for item in mutations if item.corruption_type == "dangling_reference"
        ]
        self.assertEqual("divide(#1, 1000)", str(dangling[0].operation))

    def test_rejects_gold_program_that_disagrees_with_execution_answer(self):
        bad = source()
        bad = FinQAExample(**{**bad.__dict__, "execution_answer": 0.25})
        result = build_process_examples(bad, "train", 2, seed=42)
        self.assertEqual([], result.examples)
        self.assertEqual(1, result.rejections["gold_answer_mismatch"])

    def test_metadata_records_program_length_and_validator_version(self):
        result = build_process_examples(source(), "train", 0, seed=42)
        self.assertTrue(all(item.metadata.program_length == 2 for item in result.examples))
        self.assertTrue(
            all(item.metadata.validator_version == "finprm-validator-v3" for item in result.examples)
        )


if __name__ == "__main__":
    unittest.main()
