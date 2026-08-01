from __future__ import annotations

import unittest

import torch

from scripts.train_mmb_prior import _build_priors
from src.mmb_prior_model import MMBPriorDetector


class MMBPriorModelTests(unittest.TestCase):
    def test_build_priors_from_train_payload(self) -> None:
        payload = {
            'images': [
                {'id': 1, 'width': 100, 'height': 100},
                {'id': 2, 'width': 100, 'height': 100},
            ],
            'annotations': [
                {'image_id': 1, 'bbox': [10, 10, 20, 20], 'category_id': 0},
                {'image_id': 2, 'bbox': [20, 20, 20, 20], 'category_id': 0},
            ],
        }
        priors = _build_priors(payload)
        self.assertAlmostEqual(float(priors['center_x_norm']), 0.25, places=4)
        self.assertAlmostEqual(float(priors['center_y_norm']), 0.25, places=4)
        self.assertAlmostEqual(float(priors['width_norm']), 0.20, places=4)
        self.assertAlmostEqual(float(priors['height_norm']), 0.20, places=4)
        self.assertEqual(int(priors['category_id']), 0)

    def test_model_forward_returns_detection_dict(self) -> None:
        model = MMBPriorDetector(
            center_x_norm=0.5,
            center_y_norm=0.5,
            width_norm=0.25,
            height_norm=0.25,
            category_id=0,
            confidence=0.95,
        )
        image = torch.zeros((1, 3, 200, 100), dtype=torch.float32)
        result = model(image)

        self.assertIn('boxes', result)
        self.assertIn('scores', result)
        self.assertIn('labels', result)
        self.assertEqual(tuple(result['boxes'].shape), (1, 4))
        self.assertEqual(tuple(result['scores'].shape), (1,))
        self.assertEqual(tuple(result['labels'].shape), (1,))


if __name__ == '__main__':
    unittest.main()
