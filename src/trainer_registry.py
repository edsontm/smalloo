from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.training_loop import run_training_smoke
from src.viso_mmb import run_mmb_viso


TrainerResult = Dict[str, Any]
TrainerCallable = Callable[..., TrainerResult]


TRAINER_REGISTRY: Dict[str, TrainerCallable] = {
    'smoke': run_training_smoke,
    'mmb': run_mmb_viso,
}


def available_trainers() -> List[str]:
    return sorted(TRAINER_REGISTRY)


def resolve_trainer(name: str) -> TrainerCallable:
    if name not in TRAINER_REGISTRY:
        raise KeyError(f'Unknown trainer {name!r}. Available trainers: {available_trainers()}')
    return TRAINER_REGISTRY[name]