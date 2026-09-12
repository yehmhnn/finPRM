import json
import tempfile
import unittest
from pathlib import Path

from finprm.models.serialization import (
    SerializedExample,
    grouped_train_eval_split,
    load_process_jsonl,
    serialize_input,
)


def process_input():
    return {
        "question": "What was the change?",
        "pre_text": ["Full context before."],
        "post_text": ["Full context after."],
        "supporting_facts": ["Revenue was 1200 and 1000."],
        "table": [["", "2022", "2021"], ["revenue", "1200", "1000"]],
        "prefix": ["subtract(1200, 1000)"],
        "candidate": "divide(#0, 1000)",
    }


class SerializationTests(unittest.TestCase):
    def test_serialization_contains_input_but_not_label_metadata(self):
        text = serialize_input(process_input(), "gold")
        self.assertIn("[CANDIDATE NEXT OPERATION]", text)
        self.assertIn("Revenue was 1200 and 1000.", text)
        self.assertNotIn("corruption_type", text)
        self.assertNotIn("label", text.lower())

    def test_full_evidence_mode_uses_report_text(self):
        text = serialize_input(process_input(), "full")
        self.assertIn("Full context before.", text)
        self.assertNotIn("Revenue was 1200 and 1000.", text)

    def test_loader_reads_nested_process_record(self):
        record = {
            "input": process_input(),
            "target": {"label": 1},
            "metadata": {"stable_id": "train-1", "finqa_id": "report-1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = load_process_jsonl(path)
        self.assertEqual(1, len(loaded))
        self.assertEqual(1, loaded[0].label)

    def test_grouped_split_has_no_source_question_leakage(self):
        examples = [
            SerializedExample(f"id-{group}-{step}", f"question-{group}", "text", step % 2)
            for group in range(30)
            for step in range(2)
        ]
        train, evaluation = grouped_train_eval_split(examples, 0.2, seed=42)
        train_ids = {item.finqa_id for item in train}
        evaluation_ids = {item.finqa_id for item in evaluation}
        self.assertFalse(train_ids & evaluation_ids)
        self.assertEqual(len(examples), len(train) + len(evaluation))


if __name__ == "__main__":
    unittest.main()

