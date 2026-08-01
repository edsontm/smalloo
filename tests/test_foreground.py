from __future__ import annotations

import unittest

import numpy as np

from src.mmb.foreground import ForegroundConfig, ForegroundExtractor


class ForegroundTests(unittest.TestCase):
    def test_extracts_sparse_foreground_blob(self) -> None:
        sparse = np.zeros((64, 64), dtype=np.float32)
        sparse[30:35, 20:25] = 10.0

        extractor = ForegroundExtractor(ForegroundConfig(threshold='adaptive', kernel_size=3, min_component_size=3))
        mask = extractor.extract(sparse)
        self.assertGreater(int(mask.sum()), 0)


if __name__ == '__main__':
    unittest.main()
