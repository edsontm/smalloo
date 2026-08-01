from __future__ import annotations

import unittest

from src.mmb.detection import Detection
from src.mmb.tracking import SatelliteObjectTracker


class TrackingTests(unittest.TestCase):
    def test_tracks_single_object_trajectory(self) -> None:
        tracker = SatelliteObjectTracker(max_distance=20, max_missing_frames=2)

        for frame_idx in range(5):
            x = 10 + frame_idx
            y = 20 + frame_idx
            det = Detection(bbox=[x, y, x + 3, y + 3], centroid=[x + 1.5, y + 1.5], area=9, confidence=0.9)
            tracker.update(frame_idx, [det])

        tracks = tracker.finalize()
        self.assertGreaterEqual(len(tracks), 1)
        self.assertGreaterEqual(max(len(t.frames) for t in tracks), 4)


if __name__ == '__main__':
    unittest.main()
