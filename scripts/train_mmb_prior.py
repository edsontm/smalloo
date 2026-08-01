from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import build_run_manifest
from src.mmb_prior_model import MMBPriorDetector
from src.viso_evaluation import load_coco_annotations


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _build_priors(train_payload: Dict[str, Any]) -> Dict[str, float | int]:
    images = {int(img['id']): img for img in train_payload.get('images', [])}
    anns = train_payload.get('annotations', [])
    if not anns:
        raise ValueError('Train annotations are empty; cannot build MMB priors.')

    sum_cx_norm = 0.0
    sum_cy_norm = 0.0
    sum_w_norm = 0.0
    sum_h_norm = 0.0
    counts_by_category: Dict[int, int] = {}

    valid = 0
    for ann in anns:
        image = images.get(int(ann['image_id']))
        if image is None:
            continue
        width = float(image.get('width', 0.0))
        height = float(image.get('height', 0.0))
        if width <= 0 or height <= 0:
            continue

        x, y, w, h = [float(v) for v in ann['bbox']]
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)

        sum_cx_norm += _safe_div(cx, width)
        sum_cy_norm += _safe_div(cy, height)
        sum_w_norm += _safe_div(max(w, 1.0), width)
        sum_h_norm += _safe_div(max(h, 1.0), height)
        category_id = int(ann.get('category_id', 0))
        counts_by_category[category_id] = counts_by_category.get(category_id, 0) + 1
        valid += 1

    if valid <= 0:
        raise ValueError('No valid train annotations after normalization checks.')

    majority_category = max(counts_by_category, key=counts_by_category.get)

    return {
        'center_x_norm': max(0.0, min(1.0, sum_cx_norm / valid)),
        'center_y_norm': max(0.0, min(1.0, sum_cy_norm / valid)),
        'width_norm': max(1e-4, min(1.0, sum_w_norm / valid)),
        'height_norm': max(1e-4, min(1.0, sum_h_norm / valid)),
        'category_id': int(majority_category),
        'confidence': 0.95,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train and export a bootstrap MMB TorchScript checkpoint.')
    parser.add_argument('--slug', required=True, help='Experiment slug (e.g., v1_mmb_baseline_reproduction).')
    parser.add_argument('--dataset-profile', default='viso', help='Dataset profile to use (default: viso).')
    parser.add_argument('--output', help='Optional checkpoint output path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_run_manifest(args.slug, args.dataset_profile)

    dataset_cfg = manifest['dataset']
    dataset_root = ROOT / manifest['dataset_root']
    subset_root = dataset_root / dataset_cfg['variant'] / dataset_cfg['subset']
    train_annotation_path = subset_root / 'Annotations' / dataset_cfg['splits']['train']['annotation_file']

    train_payload = load_coco_annotations(train_annotation_path)
    priors = _build_priors(train_payload)

    model = MMBPriorDetector(
        center_x_norm=float(priors['center_x_norm']),
        center_y_norm=float(priors['center_y_norm']),
        width_norm=float(priors['width_norm']),
        height_norm=float(priors['height_norm']),
        category_id=int(priors['category_id']),
        confidence=float(priors['confidence']),
    )
    scripted = torch.jit.script(model.eval())

    default_output = (
        ROOT
        / 'research'
        / 'experiments'
        / manifest['slug']
        / 'artifacts'
        / 'models'
        / f"{manifest['slug']}_{manifest['dataset_profile']}_mmb_prior.ts"
    )
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    payload = {
        'status': 'completed',
        'slug': manifest['slug'],
        'dataset_profile': manifest['dataset_profile'],
        'train_annotation_file': str(train_annotation_path.relative_to(ROOT)),
        'checkpoint': str(output_path.relative_to(ROOT)),
        'priors': priors,
        'notes': [
            'Bootstrap MMB checkpoint exported as TorchScript for real inference wiring.',
            'This is a deterministic prior detector, not the full MMB architecture from the paper.',
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
