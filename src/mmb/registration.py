from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    registered_frames: List[np.ndarray]
    transforms: List[np.ndarray]
    match_visualizations: List[np.ndarray]


class FrameRegistrar:
    """Registers a frame sequence to remove global camera/satellite motion.

    Preferred implementation uses ORB + affine estimation.
    Falls back to phase-correlation translation if OpenCV ORB is unavailable.
    """

    def __init__(self, max_features: int = 1000, good_match_ratio: float = 0.25) -> None:
        self.max_features = max_features
        self.good_match_ratio = good_match_ratio

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            # RGB to gray
            return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.uint8)
        raise ValueError('Unsupported frame shape for grayscale conversion.')

    @staticmethod
    def _warp_affine(frame: np.ndarray, affine_2x3: np.ndarray) -> np.ndarray:
        try:
            import cv2

            h, w = frame.shape[:2]
            return cv2.warpAffine(frame, affine_2x3, (w, h), flags=cv2.INTER_LINEAR)
        except Exception:
            # Numpy fallback with nearest neighbor for translation-only transforms.
            tx = int(round(float(affine_2x3[0, 2])))
            ty = int(round(float(affine_2x3[1, 2])))
            out = np.zeros_like(frame)
            h, w = frame.shape[:2]
            src_y0 = max(0, -ty)
            src_y1 = min(h, h - ty)
            src_x0 = max(0, -tx)
            src_x1 = min(w, w - tx)
            dst_y0 = max(0, ty)
            dst_y1 = min(h, h + ty)
            dst_x0 = max(0, tx)
            dst_x1 = min(w, w + tx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
            return out

    def _orb_affine(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)

        orb = cv2.ORB_create(nfeatures=self.max_features)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            LOGGER.warning('Insufficient ORB features; using identity transform.')
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), np.zeros((1, 1, 3), dtype=np.uint8)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if not matches:
            LOGGER.warning('No ORB matches; using identity transform.')
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), np.zeros((1, 1, 3), dtype=np.uint8)

        matches = sorted(matches, key=lambda m: m.distance)
        keep = max(6, int(len(matches) * self.good_match_ratio))
        good = matches[:keep]

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        affine, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if affine is None:
            affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        vis = cv2.drawMatches(prev_gray, kp1, curr_gray, kp2, good[:40], None, flags=2)
        return affine.astype(np.float32), vis

    def _phase_translation_affine(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prev_gray = self._to_gray(prev_frame).astype(np.float32)
        curr_gray = self._to_gray(curr_frame).astype(np.float32)

        prev_fft = np.fft.fft2(prev_gray)
        curr_fft = np.fft.fft2(curr_gray)
        cross_power = prev_fft * np.conj(curr_fft)
        denom = np.maximum(np.abs(cross_power), 1e-8)
        corr = np.fft.ifft2(cross_power / denom)
        max_pos = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        shift_y, shift_x = max_pos

        h, w = prev_gray.shape
        if shift_y > h // 2:
            shift_y -= h
        if shift_x > w // 2:
            shift_x -= w

        affine = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
        vis = np.zeros((1, 1, 3), dtype=np.uint8)
        return affine, vis

    def register(self, frames: List[np.ndarray]) -> RegistrationResult:
        if not frames:
            return RegistrationResult(registered_frames=[], transforms=[], match_visualizations=[])

        registered = [frames[0]]
        transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)]
        visuals = [np.zeros((1, 1, 3), dtype=np.uint8)]

        try:
            import cv2  # noqa: F401
            use_orb = True
        except Exception:
            use_orb = False
            LOGGER.warning('OpenCV unavailable. Falling back to phase-correlation registration.')

        cumulative = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        for idx in range(1, len(frames)):
            prev = registered[idx - 1]
            curr = frames[idx]
            if use_orb:
                local_affine, vis = self._orb_affine(prev, curr)
            else:
                local_affine, vis = self._phase_translation_affine(prev, curr)

            local_3x3 = np.vstack([local_affine, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
            cum_3x3 = np.vstack([cumulative, np.array([0.0, 0.0, 1.0], dtype=np.float32)]) @ local_3x3
            cumulative = cum_3x3[:2, :]

            LOGGER.info('Frame %d transform:\n%s', idx, cumulative)
            warped = self._warp_affine(curr, cumulative)
            registered.append(warped)
            transforms.append(cumulative.copy())
            visuals.append(vis)

        return RegistrationResult(registered_frames=registered, transforms=transforms, match_visualizations=visuals)
