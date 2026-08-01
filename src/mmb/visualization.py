from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from src.mmb.detection import Detection
from src.mmb.tracking import TrackState


LOGGER = logging.getLogger(__name__)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame.copy()
    return np.stack([frame, frame, frame], axis=-1)


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = _ensure_rgb(frame)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        out[y1:y2 + 1, x1] = [0, 255, 0]
        out[y1:y2 + 1, x2] = [0, 255, 0]
        out[y1, x1:x2 + 1] = [0, 255, 0]
        out[y2, x1:x2 + 1] = [0, 255, 0]
    return out


def draw_tracks(frame: np.ndarray, tracks: List[TrackState], frame_idx: int) -> np.ndarray:
    out = _ensure_rgb(frame)
    for track in tracks:
        for idx, f_idx in enumerate(track.frames):
            if f_idx != frame_idx:
                continue
            x, y = track.positions[idx]
            xi = int(round(x))
            yi = int(round(y))
            y0 = max(0, yi - 2)
            y1 = min(out.shape[0], yi + 3)
            x0 = max(0, xi - 2)
            x1 = min(out.shape[1], xi + 3)
            out[y0:y1, x0:x1] = [255, 0, 0]
    return out


def save_video(frames: List[np.ndarray], output_path: Path, fps: int = 10) -> None:
    if not frames:
        return

    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            rgb = _ensure_rgb(frame).astype(np.uint8)
            bgr = rgb[..., ::-1]
            writer.write(bgr)
        writer.release()
    except Exception as exc:
        try:
            import imageio.v2 as imageio

            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                str(output_path),
                [_ensure_rgb(frame).astype(np.uint8) for frame in frames],
                fps=fps,
            )
        except Exception as imageio_exc:
            LOGGER.warning('Could not save visualization video: %s; fallback failed: %s', exc, imageio_exc)
