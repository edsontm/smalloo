from __future__ import annotations

import unittest

import numpy as np

from src.mmb.multi_frame_difference import MultiFrameDifferenceConfig, MultiFrameDifferencer


class MultiFrameDifferenceTests(unittest.TestCase):
    def test_accumulates_temporal_windows(self) -> None:
        frames = []
        for index in range(4):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[4:6, 4 + index : 6 + index] = 255
            frames.append(frame)

        differencer = MultiFrameDifferencer(MultiFrameDifferenceConfig(temporal_windows=(1, 2), normalize_frames=True))
        result = differencer.build(frames)

        self.assertEqual(len(result.differences), 4)
        self.assertEqual(float(result.differences[0].sum()), 0.0)
        self.assertGreater(float(result.differences[1].sum()), 0.0)
        self.assertGreater(float(result.differences[2].sum()), float(result.differences[1].sum()) * 0.5)


if __name__ == '__main__':
    unittest.main()