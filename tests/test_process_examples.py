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


if __name__ == "__main__":
    unittest.main()
