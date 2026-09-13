import unittest

from finprm.evaluation import binary_metrics, select_threshold


class EvaluationMetricTests(unittest.TestCase):
    def test_binary_metrics_for_perfect_predictions(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_threshold_tie_prefers_value_closest_to_half(self):
        threshold = select_threshold([0, 1], [0.4, 0.6])

        self.assertEqual(threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
