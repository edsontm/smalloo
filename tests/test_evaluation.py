from __future__ import annotations

import unittest

from src.mmb.detection import Detection
from src.mmb.evaluation import evaluate_detections


class EvaluationTests(unittest.TestCase):
    def test_scores_precision_recall_and_false_alarms(self) -> None:
        predictions = [[Detection(bbox=[10, 10, 15, 15], centroid=[12, 12], area=36.0, confidence=0.9)]]
        ground_truth = [[[10.0, 10.0, 15.0, 15.0]]]

        result = evaluate_detections(predictions, ground_truth)
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.ap, 1.0)
        self.assertAlmostEqual(result.false_alarms, 0.0)

    def test_false_alarm_count_increases_with_extra_prediction(self) -> None:
        predictions = [
            [Detection(bbox=[10, 10, 15, 15], centroid=[12, 12], area=36.0, confidence=0.9)],
            [Detection(bbox=[30, 30, 35, 35], centroid=[32, 32], area=36.0, confidence=0.8)],
        ]
        ground_truth = [[[10.0, 10.0, 15.0, 15.0]], []]

        result = evaluate_detections(predictions, ground_truth)
        self.assertAlmostEqual(result.tp, 1.0)
        self.assertAlmostEqual(result.fp, 1.0)
        self.assertAlmostEqual(result.fn, 0.0)
        self.assertLess(result.precision, 1.0)


if __name__ == '__main__':
    unittest.main()