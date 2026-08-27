import unittest

from finprm.data.program import execute_program, parse_number, parse_program


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.table = (
            ("", "2022", "2021"),
            ("revenue", "$ 1,200", "$ 1,000"),
            ("margin", "20%", "15%"),
            ("not numeric", "n/a", "unknown"),
        )

    def test_parses_multiple_operations(self):
        operations = parse_program("subtract(1200, 1000), divide(#0, 1000)")
        self.assertEqual(2, len(operations))
        self.assertEqual("divide", operations[1].operator)
        self.assertEqual(("#0", "1000"), operations[1].arguments)

    def test_numbers_and_constants(self):
        self.assertAlmostEqual(0.236, parse_number("23.6%"))
        self.assertEqual(-1.0, parse_number("const_m1"))
        self.assertEqual(1000000.0, parse_number("const_1000000"))

    def test_executes_real_finqa_style_program(self):
        result = execute_program("divide(100, 100), divide(3.8, #0)", self.table)
        self.assertTrue(result.valid)
        self.assertEqual(3.8, result.value)
        self.assertEqual(2, len(result.steps))

    def test_executes_table_operations(self):
        result = execute_program("table_sum(revenue, none)", self.table)
        self.assertTrue(result.valid)
        self.assertEqual(2200.0, result.value)

    def test_rejects_forward_reference(self):
        result = execute_program("add(#0, 1)", self.table)
        self.assertFalse(result.valid)
        self.assertEqual("invalid_reference", result.error_type)

    def test_rejects_division_by_zero(self):
        result = execute_program("divide(1, 0)", self.table)
        self.assertFalse(result.valid)
        self.assertEqual("division_by_zero", result.error_type)

    def test_rejects_non_numeric_table_row(self):
        result = execute_program("table_average(not numeric, none)", self.table)
        self.assertFalse(result.valid)
        self.assertEqual("non_numeric_table_row", result.error_type)


if __name__ == "__main__":
    unittest.main()
