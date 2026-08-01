from __future__ import annotations

import unittest

import numpy as np

from src.mmb.detection import MotionDetector


class DetectionTests(unittest.TestCase):
    def test_detects_blob_bbox_and_contour(self) -> None:
        mask = np.zeros((50, 60), dtype=np.uint8)
        mask[10:16, 20:28] = 1

        detector = MotionDetector(min_area=5, max_area=500)
        dets = detector.detect(mask)
        self.assertEqual(len(dets), 1)
        x1, y1, x2, y2 = dets[0].bbox
        self.assertLessEqual(x1, 20)
        self.assertGreaterEqual(x2, 27)
        self.assertLessEqual(y1, 10)
        self.assertGreaterEqual(y2, 15)
        self.assertIsNotNone(dets[0].contour)
        self.assertGreater(len(dets[0].contour or []), 0)


if __name__ == '__main__':
    unittest.main()
