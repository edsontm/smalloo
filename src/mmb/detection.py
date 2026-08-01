from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Detection:
    bbox: List[float]
    centroid: List[float]
    area: float
    confidence: float
    contour: List[List[float]] | None = None

    def to_dict(self) -> Dict[str, float | List[float] | List[List[float]] | None]:
        return asdict(self)


class MotionDetector:
    """Generates connected-component object proposals from motion masks."""

    def __init__(self, min_area: int = 5, max_area: int = 500) -> None:
        self.min_area = min_area
        self.max_area = max_area

    @staticmethod
    def _connected_components(binary: np.ndarray) -> List[List[tuple[int, int]]]:
        height, width = binary.shape
        visited = np.zeros_like(binary, dtype=np.uint8)
        components: List[List[tuple[int, int]]] = []

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

                components.append(component)

        return components

    @staticmethod
    def _component_contour(component: Sequence[tuple[int, int]]) -> List[List[float]]:
        pixels = set(component)
        contour: List[List[float]] = []
        for row, col in component:
            is_boundary = False
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (row + d_row, col + d_col) not in pixels:
                    is_boundary = True
                    break
            if is_boundary:
                contour.append([float(col), float(row)])
        return contour

    def detect(self, mask: np.ndarray) -> List[Detection]:
        binary = (mask > 0).astype(np.uint8)
        detections: List[Detection] = []

        for component in self._connected_components(binary):
            area = len(component)
            if area < self.min_area or area > self.max_area:
                continue

            rows = [pixel[0] for pixel in component]
            cols = [pixel[1] for pixel in component]
            row_min, row_max = min(rows), max(rows)
            col_min, col_max = min(cols), max(cols)
            centroid = [float(sum(cols) / area), float(sum(rows) / area)]
            confidence = min(1.0, area / max(1.0, float(self.max_area)))
            contour = self._component_contour(component)

            detections.append(
                Detection(
                    bbox=[float(col_min), float(row_min), float(col_max), float(row_max)],
                    centroid=centroid,
                    area=float(area),
                    confidence=float(confidence),
                    contour=contour,
                )
            )

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        output = frame.copy()
        if output.ndim == 2:
            output = np.stack([output, output, output], axis=-1)

        height, width = output.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            output[y1 : y2 + 1, x1] = [255, 0, 0]
            output[y1 : y2 + 1, x2] = [255, 0, 0]
            output[y1, x1 : x2 + 1] = [255, 0, 0]
            output[y2, x1 : x2 + 1] = [255, 0, 0]
        return output


def generate_object_proposals(mask: np.ndarray, detector: MotionDetector | None = None) -> List[Detection]:
    return (detector or MotionDetector()).detect(mask)
