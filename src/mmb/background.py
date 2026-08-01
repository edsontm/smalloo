from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)

BackgroundMethod = Literal['temporal_median', 'running_average', 'robust_pca']


@dataclass
class BackgroundResult:
    background_frames: List[np.ndarray]
    foreground_component: List[np.ndarray]


class BackgroundModeler:
    """Background modeling for registered frame sequences."""

    def __init__(
        self,
        method: BackgroundMethod = 'robust_pca',
        alpha: float = 0.05,
        max_iter: int = 80,
        tol: float = 1e-6,
    ) -> None:
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _to_gray_float(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float32)
        return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)

    def _temporal_median(self, frames: List[np.ndarray]) -> BackgroundResult:
        gray = np.stack([self._to_gray_float(f) for f in frames], axis=0)
        median_bg = np.median(gray, axis=0)
        bg_frames = [median_bg.copy() for _ in frames]
        fg = [np.abs(gray[i] - median_bg) for i in range(gray.shape[0])]
        return BackgroundResult(background_frames=bg_frames, foreground_component=fg)

    def _running_average(self, frames: List[np.ndarray]) -> BackgroundResult:
        gray = [self._to_gray_float(f) for f in frames]
        bg = gray[0].copy()
        backgrounds: List[np.ndarray] = [bg.copy()]
        fg: List[np.ndarray] = [np.abs(gray[0] - bg)]
        for idx in range(1, len(gray)):
            bg = self.alpha * gray[idx] + (1.0 - self.alpha) * bg
            backgrounds.append(bg.copy())
            fg.append(np.abs(gray[idx] - bg))
        return BackgroundResult(background_frames=backgrounds, foreground_component=fg)

    @staticmethod
    def _svd_shrink(x: np.ndarray, tau: float) -> np.ndarray:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        s_thr = np.maximum(s - tau, 0.0)
        return (u * s_thr) @ vt

    @staticmethod
    def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

    def _robust_pca_ialm(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Inexact ALM for robust PCA: X = L + S.
        m, n = x.shape
        lam = 1.0 / np.sqrt(max(m, n))
        norm_x = np.linalg.norm(x, ord='fro')

        l = np.zeros_like(x)
        s = np.zeros_like(x)
        y = x / max(np.linalg.norm(x, ord=2), np.linalg.norm(x.reshape(-1), ord=np.inf) / lam, 1e-8)
        mu = 1.25 / max(np.linalg.norm(x, ord=2), 1e-8)
        mu_bar = mu * 1e7
        rho = 1.5

        for it in range(self.max_iter):
            l = self._svd_shrink(x - s + (1.0 / mu) * y, 1.0 / mu)
            s = self._soft_threshold(x - l + (1.0 / mu) * y, lam / mu)
            residual = x - l - s
            y = y + mu * residual
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(residual, ord='fro') / max(norm_x, 1e-8)
            if err < self.tol:
                LOGGER.info('RPCA converged at iter=%d err=%.6e', it + 1, err)
                break
        return l, s

    def _robust_pca(self, frames: List[np.ndarray]) -> BackgroundResult:
        gray = [self._to_gray_float(f) for f in frames]
        h, w = gray[0].shape
        matrix = np.stack([g.reshape(-1) for g in gray], axis=1)
        low_rank, sparse = self._robust_pca_ialm(matrix)

        bg_frames = [low_rank[:, i].reshape(h, w) for i in range(low_rank.shape[1])]
        fg = [np.abs(sparse[:, i].reshape(h, w)) for i in range(sparse.shape[1])]
        return BackgroundResult(background_frames=bg_frames, foreground_component=fg)

    def fit(self, frames: List[np.ndarray]) -> BackgroundResult:
        if not frames:
            return BackgroundResult(background_frames=[], foreground_component=[])

        if self.method == 'temporal_median':
            return self._temporal_median(frames)
        if self.method == 'running_average':
            return self._running_average(frames)
        if self.method == 'robust_pca':
            return self._robust_pca(frames)
        raise ValueError(f'Unsupported background method: {self.method}')
