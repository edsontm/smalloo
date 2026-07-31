from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.training_loop import run_training_smoke


TrainerResult = Dict[str, Any]
TrainerCallable = Callable[[int, int], TrainerResult]


def _mmb_not_implemented(seed: int, steps: int) -> TrainerResult:
    return {
        'status': 'blocked',
        'reason': 'mmb_runtime_not_implemented',
        'seed': seed,
        'steps': steps,
        'next_step': 'Add the real MMB training and evaluation code under src/ and bind it here.',
    }


TRAINER_REGISTRY: Dict[str, TrainerCallable] = {
    'smoke': run_training_smoke,
    'mmb': _mmb_not_implemented,
}


def available_trainers() -> List[str]:
    return sorted(TRAINER_REGISTRY)


def resolve_trainer(name: str) -> TrainerCallable:
    if name not in TRAINER_REGISTRY:
        raise KeyError(f'Unknown trainer {name!r}. Available trainers: {available_trainers()}')
    return TRAINER_REGISTRY[name]