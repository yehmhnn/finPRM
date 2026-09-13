import unittest

from finprm.data.evaluation_sampling import select_primary_evaluation_pairs


def record(finqa_id, step, label, corruption_type=None, variant=0):
    return {
        "input": {},
        "target": {"label": label},
        "metadata": {
            "stable_id": f"{finqa_id}-{step}-{label}-{corruption_type}-{variant}",
            "finqa_id": finqa_id,
            "step_index": step,
            "corruption_type": corruption_type,
        },
    }


class EvaluationSamplingTests(unittest.TestCase):
    def test_selects_one_positive_and_negative_per_step(self):
        records = []
        for step in range(8):
            records.append(record("question", step, 1))
            records.append(record("question", step, 0, "operator_substitution"))
            records.append(record("question", step, 0, "operand_reversal"))
        selected, summary = select_primary_evaluation_pairs(records, seed=42)
        self.assertEqual(16, len(selected))
        self.assertEqual(8, summary["positive_examples"])
        self.assertEqual(8, summary["negative_examples"])
        counts = summary["negative_type_counts"]
        self.assertLessEqual(abs(counts["operator_substitution"] - counts["operand_reversal"]), 1)

    def test_is_deterministic_and_reports_unpaired_steps(self):
        records = [
            record("q1", 0, 1),
            record("q1", 0, 0, "operator_substitution"),
            record("q2", 0, 1),
        ]
        first, first_summary = select_primary_evaluation_pairs(records, seed=7)
        second, second_summary = select_primary_evaluation_pairs(records, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(1, first_summary["excluded_steps_without_negative"])

    def test_rejects_duplicate_positives(self):
        records = [record("q", 0, 1), record("q", 0, 1)]
        with self.assertRaisesRegex(ValueError, "multiple positive"):
            select_primary_evaluation_pairs(records, seed=42)


if __name__ == "__main__":
    unittest.main()
