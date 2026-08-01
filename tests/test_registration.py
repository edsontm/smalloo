from __future__ import annotations

import unittest

import numpy as np

from src.mmb.registration import FrameRegistrar


class RegistrationTests(unittest.TestCase):
    def test_registration_reduces_translation_error(self) -> None:
        base = np.zeros((64, 64), dtype=np.uint8)
        base[20:30, 24:34] = 255

        shifted = np.zeros_like(base)
        shifted[22:32, 27:37] = 255

        registrar = FrameRegistrar(max_features=300)
        result = registrar.register([base, shifted])

        before = float(np.mean(np.abs(base.astype(np.float32) - shifted.astype(np.float32))))
        after = float(np.mean(np.abs(result.registered_frames[0].astype(np.float32) - result.registered_frames[1].astype(np.float32))))
        self.assertLessEqual(after, before)


if __name__ == '__main__':
    unittest.main()
