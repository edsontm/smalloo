from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import build_run_manifest, validate_experiment
from src.runtime_profile import apply_runtime_environment
from src.trainer_registry import available_trainers, resolve_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare a training runtime for an experiment.')
    parser.add_argument('--slug', required=True, help='Experiment slug.')
    parser.add_argument('--dataset-profile', help='Dataset profile from configs/datasets.')
    parser.add_argument('--seed', type=int, help='Optional single-seed override for local runs.')
    parser.add_argument('--smoke-steps', type=int, default=0, help='Run a minimal training smoke loop for N steps.')
    parser.add_argument('--trainer', default='smoke', choices=available_trainers(), help='Trainer backend to execute.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_experiment(args.slug, args.dataset_profile)
    if not validation['valid']:
        print(json.dumps({'validation': validation}, indent=2))
        raise SystemExit(1)

    runtime = apply_runtime_environment()
    manifest = build_run_manifest(args.slug, args.dataset_profile)
    seeds = [args.seed] if args.seed is not None else manifest['seeds']
    smoke_runs = []
    if args.smoke_steps > 0:
        trainer = resolve_trainer(args.trainer)
        for seed in seeds:
            smoke_runs.append(trainer(seed=seed, steps=args.smoke_steps, manifest=manifest))

    payload = {
        'slug': manifest['slug'],
        'title': manifest['title'],
        'dataset_profile': manifest['dataset_profile'],
        'dataset_root': manifest['dataset_root'],
        'runtime': runtime.to_dict(),
        'seeds': seeds,
        'trainer': args.trainer,
        'status': 'runtime_prepared',
        'smoke_runs': smoke_runs,
        'next_step': 'Replace the smoke loop with the real training loop while keeping SMALLOO_* as the single runtime contract.',
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()