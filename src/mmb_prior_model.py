from __future__ import annotations

from typing import Dict

import torch


class MMBPriorDetector(torch.nn.Module):
    """Tiny scripted detector that predicts one category-aware prior box per image.

    This is a bootstrap baseline to enable a real train->export->infer loop
    before integrating the full MMB architecture.
    """

    def __init__(
        self,
        center_x_norm: float,
        center_y_norm: float,
        width_norm: float,
        height_norm: float,
        category_id: int,
        confidence: float,
    ) -> None:
        super().__init__()
        self.register_buffer('center_x_norm', torch.tensor(float(center_x_norm), dtype=torch.float32))
        self.register_buffer('center_y_norm', torch.tensor(float(center_y_norm), dtype=torch.float32))
        self.register_buffer('width_norm', torch.tensor(float(width_norm), dtype=torch.float32))
        self.register_buffer('height_norm', torch.tensor(float(height_norm), dtype=torch.float32))
        self.register_buffer('category_id', torch.tensor(int(category_id), dtype=torch.int64))
        self.register_buffer('confidence', torch.tensor(float(confidence), dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Expected shape: [B, C, H, W]. In this pipeline B is always 1.
        if images.dim() != 4:
            raise RuntimeError('Expected input tensor shape [B, C, H, W].')

        height = images.shape[2]
        width = images.shape[3]

        cx = self.center_x_norm * float(width)
        cy = self.center_y_norm * float(height)
        bw = self.width_norm * float(width)
        bh = self.height_norm * float(height)

        x1 = torch.clamp(cx - (bw / 2.0), min=0.0, max=float(width))
        y1 = torch.clamp(cy - (bh / 2.0), min=0.0, max=float(height))
        x2 = torch.clamp(cx + (bw / 2.0), min=0.0, max=float(width))
        y2 = torch.clamp(cy + (bh / 2.0), min=0.0, max=float(height))

        boxes = torch.stack([x1, y1, x2, y2], dim=0).reshape(1, 4)
        scores = self.confidence.reshape(1)
        labels = self.category_id.reshape(1)
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
        }
