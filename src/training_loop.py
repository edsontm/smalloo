from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict


def _torch_available() -> bool:
    return importlib.util.find_spec('torch') is not None


def run_training_smoke(seed: int, steps: int = 3) -> Dict[str, Any]:
    if not _torch_available():
        return {
            'status': 'skipped',
            'reason': 'torch_not_installed',
            'seed': seed,
            'steps': steps,
        }

    import torch  # type: ignore

    torch.manual_seed(seed)
    device = os.environ.get('SMALLOO_DEVICE', 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = torch.nn.Linear(4, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []

    for _ in range(steps):
        features = torch.randn(8, 4, device=device)
        targets = torch.randn(8, 1, device=device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return {
        'status': 'completed',
        'seed': seed,
        'steps': steps,
        'device_used': device,
        'final_loss': losses[-1],
        'losses': losses,
    }