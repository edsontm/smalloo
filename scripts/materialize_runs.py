from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import ROOT as PROJECT_ROOT
from src.experiment_config import build_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create deterministic run manifests for an experiment.')
    parser.add_argument('--slug', required=True, help='Experiment slug.')
    parser.add_argument('--dataset-profile', help='Dataset profile from configs/datasets.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_run_manifest(args.slug, args.dataset_profile)
    runs_dir = PROJECT_ROOT / 'research' / 'experiments' / args.slug / 'artifacts' / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for seed in manifest['seeds']:
        run_payload = {
            'slug': manifest['slug'],
            'title': manifest['title'],
            'dataset_profile': manifest['dataset_profile'],
            'dataset_root': manifest['dataset_root'],
            'seed': seed,
            'baseline': manifest['baseline'],
            'objective': manifest['objective'],
            'intervention': manifest.get('intervention', {}),
            'phases': manifest['phases'],
            'runtime': manifest['runtime'],
        }
        output_path = runs_dir / f"{args.slug}_{manifest['dataset_profile']}_seed{seed}.json"
        output_path.write_text(json.dumps(run_payload, indent=2))
        created.append(str(output_path.relative_to(PROJECT_ROOT)))

    print(json.dumps({'runtime': manifest['runtime'], 'created': created}, indent=2))


if __name__ == '__main__':
    main()