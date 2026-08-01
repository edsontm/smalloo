from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayes_opt import BayesianOptimization

from src.mmb_complete import run_complete_mmb
from src.viso_evaluation import evaluate_viso_detection, load_coco_annotations


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('PyYAML is required for configs/mmb.yaml.') from exc
    return yaml.safe_load(path.read_text())


def _resolve_split_paths(dataset_root: Path, split: str) -> tuple[Path, Path, str]:
    candidates: Dict[str, List[tuple[str, str]]] = {
        'val': [('instances_val2017.json', 'val2017'), ('instances_val.json', 'val2017')],
        'train': [('instances_train2017.json', 'train2017'), ('instances_train.json', 'train2017')],
        'test': [('instances_test2017.json', 'test2017'), ('instances_test.json', 'test2017')],
    }

    def _first_existing(items: List[tuple[str, str]]) -> tuple[Path, Path] | None:
        for ann_name, image_dir_name in items:
            ann_path = dataset_root / 'Annotations' / ann_name
            image_dir = dataset_root / image_dir_name
            if ann_path.exists() and image_dir.exists():
                return ann_path, image_dir
        return None

    primary = _first_existing(candidates[split])
    if primary is not None:
        ann_path, image_dir = primary
        return ann_path, image_dir, split

    fallback_order = ['train', 'test'] if split == 'val' else ['train', 'val', 'test']
    for fallback_split in fallback_order:
        fallback = _first_existing(candidates[fallback_split])
        if fallback is not None:
            ann_path, image_dir = fallback
            return ann_path, image_dir, fallback_split

    raise FileNotFoundError(f'No available split found under {dataset_root}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bayesian optimization for paper-aligned MMB complete tuning.')
    parser.add_argument('--dataset', default='ship', help='Dataset subset under VISO/coco (e.g., ship, car, plane).')
    parser.add_argument('--split', default='val', choices=['val', 'train', 'test'], help='Target split for parameter search.')
    parser.add_argument('--max-images', type=int, default=32)
    parser.add_argument('--init-points', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = ROOT / 'VISO' / 'coco' / args.dataset
    ann_path, image_dir, resolved_split = _resolve_split_paths(dataset_root, args.split)
    payload = load_coco_annotations(ann_path)
    max_images = int(max(1, args.max_images))

    experiment_root = ROOT / 'research' / 'experiments' / 'v9_mmb_ship_bayesian_validation_tuning'
    result_root = experiment_root / 'artifacts' / 'results'
    result_root.mkdir(parents=True, exist_ok=True)

    evaluations: List[Dict[str, Any]] = []
    trial_counter = {'index': 0}

    def objective(amfd_k: float, lrmc_l: float, lrmc_k: float, pf_length: float, pf_radius: float, pf_min_occurrences: float) -> float:
        # Paper-aligned parameterization.
        params = {
            'amfd_k': float(np.clip(amfd_k, 3.0, 5.5)),
            'lrmc_l': int(np.clip(round(lrmc_l), 2, 8)),
            'lrmc_k': float(np.clip(lrmc_k, 2.0, 4.5)),
            'pf_length': int(np.clip(round(pf_length), 4, 7)),
            'pf_radius': float(np.clip(pf_radius, 5.0, 9.0)),
            'pf_min_occurrences': int(np.clip(round(pf_min_occurrences), 2, 5)),
            'score': 0.95,
        }

        trial_index = trial_counter['index']
        trial_counter['index'] += 1
        trial_output_dir = result_root / f'trial_{trial_index:03d}'
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_complete_mmb(
            images=payload['images'],
            annotations=payload['annotations'],
            image_dir=image_dir,
            max_images=max_images,
            intervention=params,
        )
        evaluation = evaluate_viso_detection(
            gt_annotations=result['ground_truth'],
            pred_annotations=result['predictions'],
        )

        trial = {
            'trial': trial_index,
            'params': params,
            'objective': float(evaluation['f1']),
            'metrics': {
                'precision': float(evaluation['precision']),
                'recall': float(evaluation['recall']),
                'f1': float(evaluation['f1']),
                'ap': float(evaluation['ap']),
                'mAP': float(evaluation['mAP']),
                'tp': int(evaluation['tp']),
                'fp': int(evaluation['fp']),
                'fn': int(evaluation['fn']),
            },
            'algorithm': result.get('algorithm', {}),
            'evaluated_images': len(result.get('evaluated_image_ids', [])),
            'predictions': len(result.get('predictions', [])),
            'ground_truth': len(result.get('ground_truth', [])),
            'output_dir': str(trial_output_dir),
        }
        evaluations.append(trial)
        (trial_output_dir / 'summary.json').write_text(json.dumps(trial, indent=2))
        return float(evaluation['f1'])

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            # Paper guidance:
            # AMFD threshold factor k ~ 4
            # LRMC temporal parameter L best at 4
            # PF uses length 5 and 7x7 neighborhood
            'amfd_k': (3.0, 5.5),
            'lrmc_l': (2.0, 8.0),
            'lrmc_k': (2.0, 4.5),
            'pf_length': (4.0, 7.0),
            'pf_radius': (5.0, 9.0),
            'pf_min_occurrences': (2.0, 5.0),
        },
        random_state=args.seed,
        verbose=2,
    )

    optimizer.maximize(init_points=max(1, args.init_points), n_iter=max(1, args.iterations))

    best_eval = max(evaluations, key=lambda item: float(item['objective'])) if evaluations else None
    payload_out = {
        'experiment': 'v9_mmb_ship_bayesian_validation_tuning',
        'optimizer': 'bayesian-optimization/BayesianOptimization',
        'seed': args.seed,
        'dataset': args.dataset,
        'requested_split': args.split,
        'resolved_split': resolved_split,
        'annotations_path': str(ann_path.relative_to(ROOT)),
        'image_dir': str(image_dir.relative_to(ROOT)),
        'max_images': max_images,
        'init_points': int(max(1, args.init_points)),
        'iterations': int(max(1, args.iterations)),
        'objective': 'maximize_f1',
        'space': {
            'amfd_k': [3.0, 5.5],
            'lrmc_l': [2, 8],
            'lrmc_k': [2.0, 4.5],
            'pf_length': [4, 7],
            'pf_radius': [5.0, 9.0],
            'pf_min_occurrences': [2, 5],
        },
        'paper_reference': {
            'amfd_k': 4.0,
            'lrmc_l': 4,
            'pf_length': 5,
            'pf_radius': 7.0,
            'notes': 'Ranges were centered near the paper values and expanded conservatively.',
        },
        'best': best_eval,
        'optimizer_best': optimizer.max,
        'trials': evaluations,
    }

    summary_path = result_root / 'bayes_opt_summary.json'
    summary_path.write_text(json.dumps(payload_out, indent=2))
    print(json.dumps(payload_out, indent=2))


if __name__ == '__main__':
    main()
