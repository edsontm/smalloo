from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class RobustMatrixCompletionConfig:
    lambda_value: float | None = None
    mu: float | None = None
    rho: float = 1.5
    max_iter: int = 100
    tol: float = 1e-6
    mu_bar_scale: float = 1e7


@dataclass
class RobustMatrixCompletionResult:
    low_rank_frames: List[np.ndarray]
    sparse_frames: List[np.ndarray]
    iterations: int
    residual_norm: float


class RobustMatrixCompletion:
    """Robust PCA / ALM decomposition for video matrices."""

    def __init__(self, config: RobustMatrixCompletionConfig | None = None) -> None:
        self.config = config or RobustMatrixCompletionConfig()

    @staticmethod
    def _to_frame_stack(frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError('frames must not be empty')

        stack = [np.asarray(frame, dtype=np.float32) for frame in frames]
        shape = stack[0].shape
        for frame in stack:
            if frame.shape != shape:
                raise ValueError('all frames must have the same shape')
        return np.stack(stack, axis=0)

    @staticmethod
    def _shrink(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def decompose(self, frames: Sequence[np.ndarray]) -> RobustMatrixCompletionResult:
        stack = self._to_frame_stack(frames)
        num_frames, height, width = stack.shape
        matrix = stack.reshape(num_frames, height * width).T

        fro_norm = float(np.linalg.norm(matrix, ord='fro'))
        if fro_norm == 0.0:
            zeros = [np.zeros((height, width), dtype=np.float32) for _ in range(num_frames)]
            return RobustMatrixCompletionResult(
                low_rank_frames=zeros,
                sparse_frames=zeros,
                iterations=0,
                residual_norm=0.0,
            )

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
        lambda_value = self.config.lambda_value or (1.0 / np.sqrt(max(matrix.shape)))
        dual_norm = max(spectral_norm, float(np.max(np.abs(matrix)) / max(lambda_value, 1e-12)))
        y = matrix / dual_norm if dual_norm > 0 else np.zeros_like(matrix)

        mu = self.config.mu or (1.25 / max(spectral_norm, 1e-12))
        mu_bar = mu * self.config.mu_bar_scale
        low_rank = np.zeros_like(matrix)
        sparse = np.zeros_like(matrix)
        residual_norm = float('inf')

        for iteration in range(1, self.config.max_iter + 1):
            u, singular_values, vt = np.linalg.svd(matrix - sparse + (y / mu), full_matrices=False)
            shrunk = np.maximum(singular_values - (1.0 / mu), 0.0)
            rank = int(np.count_nonzero(shrunk > 0.0))
            if rank > 0:
                low_rank = (u[:, :rank] * shrunk[:rank]) @ vt[:rank, :]
            else:
                low_rank = np.zeros_like(matrix)

            temp = matrix - low_rank + (y / mu)
            sparse = self._shrink(temp, lambda_value / mu)

            residual = matrix - low_rank - sparse
            residual_norm = float(np.linalg.norm(residual, ord='fro') / fro_norm)
            y = y + mu * residual
            if residual_norm <= self.config.tol:
                break
            mu = min(mu * self.config.rho, mu_bar)
        else:
            iteration = self.config.max_iter

        low_rank_frames = low_rank.T.reshape(num_frames, height, width)
        sparse_frames = sparse.T.reshape(num_frames, height, width)
        return RobustMatrixCompletionResult(
            low_rank_frames=[frame.astype(np.float32) for frame in low_rank_frames],
            sparse_frames=[frame.astype(np.float32) for frame in sparse_frames],
            iterations=iteration,
            residual_norm=residual_norm,
        )