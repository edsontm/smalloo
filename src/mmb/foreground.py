from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np


@dataclass
class ForegroundConfig:
    threshold: Union[str, float] = 'adaptive'
    kernel_size: int = 3
    min_component_size: int = 5
    normalize_foreground: bool = True


class ForegroundExtractor:
    """Normalize, threshold, and clean sparse foreground responses."""

    def __init__(self, config: ForegroundConfig | None = None) -> None:
        self.config = config or ForegroundConfig()

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
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max <= x_min:
            return np.zeros_like(x, dtype=np.float32)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def _binary_erode(binary: np.ndarray, kernel_size: int) -> np.ndarray:
        radius = kernel_size // 2
        height, width = binary.shape
        padded = np.pad(binary, radius, mode='constant', constant_values=0)
        output = np.zeros_like(binary, dtype=np.uint8)
        for row in range(height):
            for col in range(width):
                window = padded[row : row + kernel_size, col : col + kernel_size]
                output[row, col] = 1 if np.all(window > 0) else 0
        return output

    @staticmethod
    def _binary_dilate(binary: np.ndarray, kernel_size: int) -> np.ndarray:
        radius = kernel_size // 2
        height, width = binary.shape
        padded = np.pad(binary, radius, mode='constant', constant_values=0)
        output = np.zeros_like(binary, dtype=np.uint8)
        for row in range(height):
            for col in range(width):
                window = padded[row : row + kernel_size, col : col + kernel_size]
                output[row, col] = 1 if np.any(window > 0) else 0
        return output

    @classmethod
    def _open_close(cls, binary: np.ndarray, kernel_size: int) -> np.ndarray:
        opened = cls._binary_dilate(cls._binary_erode(binary, kernel_size), kernel_size)
        return cls._binary_erode(cls._binary_dilate(opened, kernel_size), kernel_size)

    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape
        visited = np.zeros_like(binary, dtype=np.uint8)
        cleaned = np.zeros_like(binary, dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                if binary[row, col] == 0 or visited[row, col] != 0:
                    continue

                stack = [(row, col)]
                visited[row, col] = 1
                component: List[tuple[int, int]] = []
                while stack:
                    current_row, current_col = stack.pop()
                    component.append((current_row, current_col))
                    for d_row in (-1, 0, 1):
                        for d_col in (-1, 0, 1):
                            if d_row == 0 and d_col == 0:
                                continue
                            next_row = current_row + d_row
                            next_col = current_col + d_col
                            if next_row < 0 or next_col < 0 or next_row >= height or next_col >= width:
                                continue
                            if binary[next_row, next_col] == 0 or visited[next_row, next_col] != 0:
                                continue
                            visited[next_row, next_col] = 1
                            stack.append((next_row, next_col))

                if len(component) >= self.config.min_component_size:
                    for component_row, component_col in component:
                        cleaned[component_row, component_col] = 1

        return cleaned

    def extract(self, foreground: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
        foreground_g = self._to_gray_float(foreground)
        if background is not None:
            foreground_g = np.abs(foreground_g - self._to_gray_float(background))

        if self.config.normalize_foreground:
            foreground_g = self._normalize(foreground_g)

        if self.config.threshold == 'adaptive':
            threshold = float(foreground_g.mean() + 0.5 * foreground_g.std())
        else:
            threshold = float(self.config.threshold)

        binary = (foreground_g >= threshold).astype(np.uint8)
        if self.config.kernel_size > 1:
            binary = self._open_close(binary, int(self.config.kernel_size))
        return self._remove_small_components(binary)

    def extract_sequence(
        self,
        foreground_frames: Sequence[np.ndarray],
        background_frames: Sequence[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        if background_frames is None:
            background_frames = [None] * len(foreground_frames)
        return [self.extract(foreground, background) for foreground, background in zip(foreground_frames, background_frames)]
