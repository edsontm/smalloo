from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.runtime_profile import apply_runtime_environment


ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / 'configs'
EXPERIMENTS_DIR = ROOT / 'research' / 'experiments'
REQUIRED_EXPERIMENT_FILES = [
    'README.md',
    '01_problem_statement.md',
    '02_research_hypothesis.md',
    '03_literature_review.md',
    '04_experiment_plan.md',
    '05_implementation_notes.md',
    '06_benchmark.md',
    '07_ablation.md',
    '08_statistical_validation.md',
    '09_discussion.md',
    '10_blog_post.md',
    '11_decision.md',
    'artifacts/README.md',
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def experiment_config_path(slug: str) -> Path:
    return CONFIGS_DIR / 'experiments' / f'{slug}.json'


def dataset_profile_path(profile_name: str) -> Path:
    return CONFIGS_DIR / 'datasets' / f'{profile_name}.json'


def load_experiment_config(slug: str) -> Dict[str, Any]:
    path = experiment_config_path(slug)
    if not path.exists():
        raise FileNotFoundError(f'Experiment config not found: {path}')
    return _load_json(path)


def load_dataset_profile(profile_name: str) -> Dict[str, Any]:
    path = dataset_profile_path(profile_name)
    if not path.exists():
        raise FileNotFoundError(f'Dataset profile not found: {path}')
    return _load_json(path)


def resolve_dataset_root(profile_name: str) -> Path:
    profile = load_dataset_profile(profile_name)
    dataset_root = ROOT / profile['root_dir']
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')
    return dataset_root


def _validate_required_files(slug: str) -> List[str]:
    experiment_dir = EXPERIMENTS_DIR / slug
    missing = []
    for relative_path in REQUIRED_EXPERIMENT_FILES:
        if not (experiment_dir / relative_path).exists():
            missing.append(relative_path)
    return missing


def _validate_dataset_layout(experiment_config: Dict[str, Any], dataset_root: Path) -> List[str]:
    missing = []
    dataset_cfg = experiment_config['dataset']
    subset_root = dataset_root / dataset_cfg['variant'] / dataset_cfg['subset']
    if not subset_root.exists():
        return [str(subset_root.relative_to(ROOT))]

    annotations_dir = subset_root / 'Annotations'
    if not annotations_dir.exists():
        missing.append(str(annotations_dir.relative_to(ROOT)))

    for split_name, split_cfg in dataset_cfg['splits'].items():
        image_dir = subset_root / split_cfg['image_dir']
        annotation_file = annotations_dir / split_cfg['annotation_file']
        if not image_dir.exists():
            missing.append(str(image_dir.relative_to(ROOT)))
        if not annotation_file.exists():
            missing.append(str(annotation_file.relative_to(ROOT)))
    return missing


def build_run_manifest(slug: str, dataset_profile_name: str | None = None) -> Dict[str, Any]:
    experiment_config = load_experiment_config(slug)
    dataset_profile_name = dataset_profile_name or experiment_config['dataset']['default_profile']
    dataset_root = resolve_dataset_root(dataset_profile_name)
    runtime_profile = apply_runtime_environment()

    return {
        'slug': slug,
        'title': experiment_config['experiment']['title'],
        'dataset_profile': dataset_profile_name,
        'dataset_root': str(dataset_root.relative_to(ROOT)),
        'dataset': experiment_config['dataset'],
        'baseline': experiment_config['experiment']['baseline'],
        'objective': experiment_config['experiment']['objective'],
        'seeds': experiment_config['execution']['seeds'],
        'phases': experiment_config['execution']['phases'],
        'runtime': runtime_profile.to_dict(),
    }


def validate_experiment(slug: str, dataset_profile_name: str | None = None) -> Dict[str, Any]:
    experiment_config = load_experiment_config(slug)
    dataset_profile_name = dataset_profile_name or experiment_config['dataset']['default_profile']
    dataset_root = resolve_dataset_root(dataset_profile_name)
    missing_files = _validate_required_files(slug)
    missing_dataset_paths = _validate_dataset_layout(experiment_config, dataset_root)

    return {
        'slug': slug,
        'dataset_profile': dataset_profile_name,
        'experiment_dir': str((EXPERIMENTS_DIR / slug).relative_to(ROOT)),
        'dataset_root': str(dataset_root.relative_to(ROOT)),
        'missing_files': missing_files,
        'missing_dataset_paths': missing_dataset_paths,
        'valid': not missing_files and not missing_dataset_paths,
    }