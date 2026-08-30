import json
import tempfile
import unittest
from pathlib import Path

from finprm.data.finqa import FinQASchemaError, load_split


class FinQALoaderTests(unittest.TestCase):
    def write_split(self, records):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "split.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def test_loads_valid_record(self):
        path = self.write_split(
            [
                {
                    "id": "report-1",
                    "pre_text": ["before"],
                    "post_text": ["after"],
                    "table": [["", "2022"], ["revenue", "100"]],
                    "qa": {
                        "question": "What is revenue?",
                        "program": "add(100, 0)",
                        "exe_ans": 100,
                    },
                }
            ]
        )
        example = next(load_split(path))
        self.assertEqual("report-1", example.example_id)
        self.assertEqual("add(100, 0)", example.program)
        self.assertEqual((), example.supporting_facts)

    def test_rejects_inconsistent_table_width(self):
        path = self.write_split(
            [
                {
                    "id": "bad-table",
                    "table": [["", "2022"], ["revenue"]],
                    "qa": {"question": "Question?"},
                }
            ]
        )
        with self.assertRaises(FinQASchemaError):
            list(load_split(path))


if __name__ == "__main__":
    unittest.main()
