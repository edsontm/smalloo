from __future__ import annotations

import unittest

import numpy as np

from src.mmb.robust_matrix_completion import RobustMatrixCompletion, RobustMatrixCompletionConfig


class RobustMatrixCompletionTests(unittest.TestCase):
    def test_separates_low_rank_background_and_sparse_motion(self) -> None:
        frames = []
        for index in range(4):
            frame = np.full((12, 12), 0.1, dtype=np.float32)
            if index in {1, 2}:
                frame[5:7, 5 + index - 1 : 7 + index - 1] = 1.0
            frames.append(frame)

        completion = RobustMatrixCompletion(RobustMatrixCompletionConfig(max_iter=60, tol=1e-5))
        result = completion.decompose(frames)

        self.assertEqual(len(result.low_rank_frames), 4)
        self.assertEqual(len(result.sparse_frames), 4)
        self.assertLess(result.residual_norm, 0.25)
        self.assertGreater(float(np.abs(result.sparse_frames[1]).sum()), 0.0)


if __name__ == '__main__':
    unittest.main()