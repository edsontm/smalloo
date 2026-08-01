from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class MultiFrameDifferenceConfig:
    temporal_windows: tuple[int, ...] = (1, 2, 3)
    normalize_frames: bool = True


@dataclass
class DifferenceResult:
    differences: List[np.ndarray]
    normalized_frames: List[np.ndarray]
    windows: tuple[int, ...]


class MultiFrameDifferencer:
    """Builds the accumulative multi-frame difference representation."""

    def __init__(self, config: MultiFrameDifferenceConfig | None = None) -> None:
        self.config = config or MultiFrameDifferenceConfig()

    @staticmethod
    def _to_gray_float(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        if frame.shape[-1] >= 3:
            return (
                0.299 * frame[..., 0]
                + 0.587 * frame[..., 1]
                + 0.114 * frame[..., 2]
            ).astype(np.float32)
        return frame.astype(np.float32)

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        min_value = float(frame.min())
        max_value = float(frame.max())
        if max_value <= min_value:
            return np.zeros_like(frame, dtype=np.float32)
        return (frame - min_value) / (max_value - min_value)

    def build(self, frames: Sequence[np.ndarray]) -> DifferenceResult:
        if not frames:
            raise ValueError('frames must not be empty')

        gray_frames = [self._to_gray_float(frame) for frame in frames]
        if self.config.normalize_frames:
            gray_frames = [self._normalize(frame) for frame in gray_frames]

        windows = tuple(sorted(int(window) for window in self.config.temporal_windows if int(window) > 0))
        if not windows:
            raise ValueError('temporal_windows must contain at least one positive integer')

        differences: List[np.ndarray] = []
        for index, frame in enumerate(gray_frames):
            accumulated = np.zeros_like(frame, dtype=np.float32)
            for window in windows:
                previous_index = index - window
                if previous_index < 0:
                    continue
                accumulated += np.abs(frame - gray_frames[previous_index])
            differences.append(accumulated)

        return DifferenceResult(differences=differences, normalized_frames=gray_frames, windows=windows)

    def build_from_iterable(self, frames: Iterable[np.ndarray]) -> DifferenceResult:
        return self.build(list(frames))