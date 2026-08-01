from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.mmb.detection import Detection


@dataclass
class TrackState:
    track_id: int
    frames: List[int] = field(default_factory=list)
    positions: List[List[float]] = field(default_factory=list)
    bounding_boxes: List[List[float]] = field(default_factory=list)
    misses: int = 0


@dataclass
class _KalmanTrack:
    track_id: int
    x: np.ndarray
    p: np.ndarray
    state: TrackState
    misses: int = 0


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian assignment for rectangular cost matrix.

    Returns (row, col) assignments minimizing cost.
    """

    if cost.size == 0:
        return []

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    max_cost = float(cost.max() if cost.size else 0.0)
    padded = np.full((n, n), max_cost + 1e6, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: List[Tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_rows and j <= n_cols:
            assignments.append((i - 1, j - 1))
    return assignments


class SatelliteObjectTracker:
    """Multi-object tracker using Kalman prediction + Hungarian assignment."""

    def __init__(self, max_distance: float = 50.0, max_missing_frames: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: List[_KalmanTrack] = []

        dt = 1.0
        self.f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.q = np.eye(4, dtype=np.float64) * 1e-2
        self.r = np.eye(2, dtype=np.float64) * 2.0

    def _init_track(self, frame_idx: int, det: Detection) -> _KalmanTrack:
        x = np.array([det.centroid[0], det.centroid[1], 0.0, 0.0], dtype=np.float64)
        p = np.eye(4, dtype=np.float64) * 10.0
        state = TrackState(
            track_id=self._next_track_id,
            frames=[frame_idx],
            positions=[list(det.centroid)],
            bounding_boxes=[list(det.bbox)],
        )
        track = _KalmanTrack(track_id=self._next_track_id, x=x, p=p, state=state)
        self._next_track_id += 1
        return track

    def _predict(self, track: _KalmanTrack) -> None:
        track.x = self.f @ track.x
        track.p = self.f @ track.p @ self.f.T + self.q

    def _update(self, track: _KalmanTrack, z: np.ndarray) -> None:
        y = z - (self.h @ track.x)
        s = self.h @ track.p @ self.h.T + self.r
        k = track.p @ self.h.T @ np.linalg.inv(s)
        track.x = track.x + (k @ y)
        i = np.eye(4)
        track.p = (i - (k @ self.h)) @ track.p

    def update(self, frame_idx: int, detections: List[Detection]) -> None:
        for track in self._tracks:
            self._predict(track)

        if not self._tracks and detections:
            self._tracks = [self._init_track(frame_idx, det) for det in detections]
            return

        if not detections:
            kept: List[_KalmanTrack] = []
            for track in self._tracks:
                track.misses += 1
                if track.misses <= self.max_missing_frames:
                    kept.append(track)
            self._tracks = kept
            return

        track_points = np.array([[t.x[0], t.x[1]] for t in self._tracks], dtype=np.float64)
        det_points = np.array([det.centroid for det in detections], dtype=np.float64)

        cost = np.linalg.norm(track_points[:, None, :] - det_points[None, :, :], axis=2)
        assignments = _hungarian(cost)

        matched_tracks = set()
        matched_dets = set()

        for track_idx, det_idx in assignments:
            dist = float(cost[track_idx, det_idx])
            if dist > self.max_distance:
                continue
            track = self._tracks[track_idx]
            det = detections[det_idx]
            self._update(track, np.array(det.centroid, dtype=np.float64))
            track.misses = 0
            track.state.frames.append(frame_idx)
            track.state.positions.append([float(track.x[0]), float(track.x[1])])
            track.state.bounding_boxes.append(list(det.bbox))
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

        new_tracks: List[_KalmanTrack] = []
        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.misses += 1
            if track.misses <= self.max_missing_frames:
                new_tracks.append(track)

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                new_tracks.append(self._init_track(frame_idx, det))

        self._tracks = new_tracks

    def finalize(self) -> List[TrackState]:
        return [track.state for track in self._tracks]

    @staticmethod
    def summarize(tracks: List[TrackState]) -> Dict[str, float]:
        lengths = [len(t.frames) for t in tracks]
        return {
            'num_tracks': float(len(tracks)),
            'mean_track_length': float(np.mean(lengths) if lengths else 0.0),
            'max_track_length': float(max(lengths) if lengths else 0.0),
            'id_switches': 0.0,
        }
