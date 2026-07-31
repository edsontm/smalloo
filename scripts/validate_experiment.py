from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import build_run_manifest, validate_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate experiment scaffold and dataset layout.')
    parser.add_argument('--slug', required=True, help='Experiment slug.')
    parser.add_argument('--dataset-profile', help='Dataset profile from configs/datasets.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_experiment(args.slug, args.dataset_profile)
    manifest = build_run_manifest(args.slug, args.dataset_profile)
    payload = {
        'validation': validation,
        'run_manifest': manifest,
    }
    print(json.dumps(payload, indent=2))
    if not validation['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()