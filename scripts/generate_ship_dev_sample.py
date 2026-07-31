from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path('/Users/edsontm/dev/smalloo')
COCO_ROOT = ROOT / 'VISO' / 'coco'
DEV_SAMPLE_ROOT = ROOT / 'devsample' / 'coco'
DEFAULT_DATASETS = ('ship',)
FRACTION = 0.10
MIN_PER_STRATUM = 10

Stratum = Tuple[str, int]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def evenly_spaced_indices(length: int, sample_size: int) -> List[int]:
    if sample_size >= length:
        return list(range(length))
    if sample_size <= 1:
        return [length // 2]
    step = (length - 1) / (sample_size - 1)
    chosen = []
    for idx in range(sample_size):
        candidate = int(round(idx * step))
        if chosen and candidate <= chosen[-1]:
            candidate = chosen[-1] + 1
        chosen.append(min(candidate, length - (sample_size - idx)))
    return chosen


def select_evenly(items: List[dict], sample_size: int) -> List[dict]:
    indices = evenly_spaced_indices(len(items), sample_size)
    return [items[index] for index in indices]


def copy_many(paths: Iterable[Tuple[Path, Path]]) -> None:
    for source, target in paths:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def resolve_split_mappings(source_dir: Path) -> List[Tuple[str, str]]:
    image_splits = sorted(
        path.name for path in source_dir.iterdir() if path.is_dir() and path.name != 'Annotations'
    )
    mappings: List[Tuple[str, str]] = []
    for ann_path in sorted((source_dir / 'Annotations').glob('instances_*.json')):
        split_stem = ann_path.stem.removeprefix('instances_')
        if split_stem in image_splits:
            mappings.append((split_stem, split_stem))
            continue
        candidate = f'{split_stem}2017'
        if candidate in image_splits:
            mappings.append((split_stem, candidate))
            continue
        raise FileNotFoundError(
            f'No image directory found for annotation split {split_stem!r} in {source_dir}'
        )
    return mappings


def build_subset(source_dir: Path, target_dir: Path, ann_split: str, image_split: str) -> Dict[str, object]:
    ann_path = source_dir / 'Annotations' / f'instances_{ann_split}.json'
    data = json.loads(ann_path.read_text())

    annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
    for annotation in data['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)

    images_by_stratum: Dict[Stratum, List[dict]] = defaultdict(list)
    for image in sorted(data['images'], key=lambda item: item['file_name']):
        key = (f"{image['width']}x{image['height']}", len(annotations_by_image[image['id']]))
        images_by_stratum[key].append(image)

    selected_images: List[dict] = []
    selected_annotations: List[dict] = []
    stratum_summary = {}

    for stratum, images in sorted(images_by_stratum.items()):
        sample_size = min(len(images), max(MIN_PER_STRATUM, math.ceil(len(images) * FRACTION)))
        chosen_images = select_evenly(images, sample_size)
        selected_images.extend(chosen_images)
        for image in chosen_images:
            selected_annotations.extend(annotations_by_image[image['id']])
        stratum_summary[f'{stratum[0]}__labels_{stratum[1]}'] = {
            'source_images': len(images),
            'sample_images': len(chosen_images),
        }

    selected_image_ids = {image['id'] for image in selected_images}
    selected_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] in selected_image_ids]
    selected_images = sorted(selected_images, key=lambda item: item['file_name'])

    subset = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': data['categories'],
    }

    target_image_dir = target_dir / image_split
    target_xml_dir = target_dir / 'Annotations' / image_split
    target_ann_path = target_dir / 'Annotations' / f'instances_{ann_split}.json'
    target_ann_path.parent.mkdir(parents=True, exist_ok=True)
    target_ann_path.write_text(json.dumps(subset, indent=2))

    image_copy_jobs = []
    xml_copy_jobs = []
    for image in selected_images:
        image_name = image['file_name']
        xml_name = Path(image_name).with_suffix('.xml').name
        image_copy_jobs.append((source_dir / image_split / image_name, target_image_dir / image_name))
        xml_copy_jobs.append((source_dir / 'Annotations' / image_split / xml_name, target_xml_dir / xml_name))

    copy_many(image_copy_jobs)
    copy_many(xml_copy_jobs)

    labels_per_image = Counter(len(annotations_by_image[image['id']]) for image in selected_images)
    return {
        'annotation_split': ann_split,
        'image_split': image_split,
        'source_images': len(data['images']),
        'source_annotations': len(data['annotations']),
        'sample_images': len(selected_images),
        'sample_annotations': len(selected_annotations),
        'labels_per_image': dict(sorted(labels_per_image.items())),
        'strata': stratum_summary,
    }


def generate_dataset(dataset_name: str) -> Dict[str, object]:
    source_dir = COCO_ROOT / dataset_name
    target_dir = DEV_SAMPLE_ROOT / dataset_name
    ensure_clean_dir(target_dir)
    split_mappings = resolve_split_mappings(source_dir)
    summaries = [
        build_subset(source_dir, target_dir, ann_split, image_split)
        for ann_split, image_split in split_mappings
    ]
    manifest = {
        'dataset': dataset_name,
        'source': str(source_dir.relative_to(ROOT)),
        'target': str(target_dir.relative_to(ROOT)),
        'layout_note': 'Rename devsample to VISO to reuse code paths that expect the original dataset root.',
        'sampling': {
            'strategy': 'deterministic_even_spacing_per_stratum',
            'fraction': FRACTION,
            'min_per_stratum': MIN_PER_STRATUM,
            'strata': ['image_resolution', 'labels_per_image'],
        },
        'splits': summaries,
    }
    (target_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (target_dir / 'README.md').write_text(
        f'# {dataset_name}\n\n'
        f'Subamostra deterministica de VISO/coco/{dataset_name} para desenvolvimento e testes rapidos.\n\n'
        '- Estratificacao por resolucao da imagem e numero de rotulos por imagem.\n'
        '- Selecao sistematica ao longo da ordem dos arquivos para cobrir o conjunto inteiro.\n'
        '- Conteudo: imagens, JSON COCO por split e XMLs correspondentes.\n'
        '- Estrutura espelhada de VISO para permitir alternancia por renomeacao da pasta raiz.\n'
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate deterministic dev samples for VISO COCO subsets.')
    parser.add_argument('datasets', nargs='*', default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DEV_SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = [generate_dataset(dataset_name) for dataset_name in args.datasets]
    print(json.dumps(manifests, indent=2))


if __name__ == '__main__':
    main()
