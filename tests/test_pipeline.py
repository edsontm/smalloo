from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.mmb.pipeline import MMBPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_end_to_end_synthetic_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            h, w = 64, 64
            frames = []
            for i in range(10):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                # Camera translation + moving object.
                ox = 20 + i
                oy = 30 + (i // 2)
                x0 = min(w - 4, max(0, ox))
                y0 = min(h - 4, max(0, oy))
                frame[y0:y0 + 4, x0:x0 + 4] = [255, 255, 255]
                frames.append(frame)

            config = {
                'multi_frame_difference': {'temporal_windows': [1, 2], 'normalize_frames': True},
                'robust_matrix_completion': {'max_iter': 40, 'tol': 1e-5},
                'foreground': {'threshold': 'adaptive', 'morphology_kernel': 3, 'min_component_size': 3},
                'detection': {'min_area': 3, 'max_area': 200},
            }

            pipeline = MMBPipeline(config)
            out = pipeline.run(frames=frames, output_dir=tmp / 'results')
            self.assertEqual(out.metrics['num_frames'], 10)
            self.assertIn('detection', out.metrics)
            self.assertGreaterEqual(out.metrics['detection']['precision'], 0.0)

            results_dir = tmp / 'results'
            self.assertTrue((results_dir / 'detections.json').exists())
            self.assertTrue((results_dir / 'tracks.json').exists())
            self.assertTrue((results_dir / 'metrics.json').exists())
            self.assertTrue((results_dir / 'background.png').exists())
            self.assertTrue((results_dir / 'visualization.mp4').exists() or any(results_dir.glob('visualization.*')))


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
