from __future__ import annotations

import unittest

import numpy as np

from src.mmb.background import BackgroundModeler


class BackgroundTests(unittest.TestCase):
    def test_temporal_median_background(self) -> None:
        frames = [np.zeros((32, 32), dtype=np.uint8) for _ in range(5)]
        frames[2][10:12, 10:12] = 255

        modeler = BackgroundModeler(method='temporal_median')
        out = modeler.fit(frames)
        self.assertEqual(len(out.background_frames), 5)
        self.assertLess(float(np.max(out.background_frames[0])), 1.0)

    def test_running_average_background(self) -> None:
        frames = [np.zeros((16, 16), dtype=np.uint8), np.ones((16, 16), dtype=np.uint8) * 100]
        modeler = BackgroundModeler(method='running_average', alpha=0.5)
        out = modeler.fit(frames)
        self.assertEqual(len(out.background_frames), 2)
        self.assertGreater(float(out.background_frames[1].mean()), 0.0)


if __name__ == '__main__':
    unittest.main()
