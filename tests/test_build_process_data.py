import importlib.util
import unittest
from pathlib import Path

from finprm.data.finqa import FinQAExample
from finprm.data.process_examples import build_process_examples


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_process_data.py"
SPEC = importlib.util.spec_from_file_location("build_process_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def source(example_id: str, program: str):
    return FinQAExample(
        example_id=example_id,
        question="What was the change?",
        table=(("", "2022", "2021"), ("revenue", "1200", "1000")),
        pre_text=(),
        post_text=(),
        supporting_facts=("Revenue was 1200 and 1000.",),
        program=program,
        execution_answer=200.0,
    )


class AuditSamplingTests(unittest.TestCase):
    def test_sample_includes_positives_and_is_deterministic(self):
        examples = []
        for index in range(6):
            result = build_process_examples(
                source(f"report-{index}", "subtract(1200, 1000)"),
                "train",
                2,
                seed=42,
            )
            examples.extend(result.examples)
        first = MODULE.stratified_audit_sample(examples, per_group=2, seed=42)
        second = MODULE.stratified_audit_sample(examples, per_group=2, seed=42)
        self.assertEqual(
            [item.metadata.stable_id for item in first],
            [item.metadata.stable_id for item in second],
        )
        self.assertIn(1, {item.target.label for item in first})
        self.assertIn(0, {item.target.label for item in first})
        counts = {}
        for item in first:
            group = item.metadata.corruption_type or "positive"
            counts[group] = counts.get(group, 0) + 1
        self.assertTrue(all(count <= 2 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
