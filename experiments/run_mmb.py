from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mmb.pipeline import MMBPipeline


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('PyYAML is required for configs/mmb.yaml.') from exc
    return yaml.safe_load(path.read_text())


def _count_video_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _load_image_frames(image_dir: Path, image_names: List[str] | None = None) -> List[Any]:
    frames: List[Any] = []
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Pillow is required for image-directory inputs.') from exc

    if image_names is None:
        paths = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        )
    else:
        paths = [image_dir / name for name in image_names]

    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert('RGB')))
    return frames


def _load_gt_boxes_by_frame(gt_json: Path) -> List[List[List[float]]]:
    payload = json.loads(gt_json.read_text())
    images = sorted(
        payload.get('images', []),
        key=lambda item: str(item.get('file_name', '')),
    )
    boxes_by_index: List[List[List[float]]] = [[] for _ in images]
    index_by_id = {int(img['id']): idx for idx, img in enumerate(images)}
    for ann in payload.get('annotations', []):
        image_id = int(ann['image_id'])
        frame_index = index_by_id.get(image_id)
        if frame_index is None:
            continue
        x, y, w, h = [float(v) for v in ann['bbox']]
        boxes_by_index[frame_index].append([x, y, x + w, y + h])
    return boxes_by_index


def _load_gt_images(gt_json: Path) -> List[str]:
    payload = json.loads(gt_json.read_text())
    return [str(item.get('file_name', '')) for item in sorted(payload.get('images', []), key=lambda item: str(item.get('file_name', '')))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run full MMB pipeline experiment.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', help='Path to input video file.')
    group.add_argument('--images-dir', help='Path to a directory of input frames.')
    parser.add_argument('--output', default='results/mmb', help='Output directory.')
    parser.add_argument('--gt-json', help='Optional COCO annotations for detection/tracking metrics.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(Path(args.config))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = None
    gt_boxes_by_frame = None
    if args.gt_json:
        gt_json_path = Path(args.gt_json)
        gt_boxes_candidate = _load_gt_boxes_by_frame(gt_json_path)
        if args.images_dir:
            image_names = _load_gt_images(gt_json_path)
            frames = _load_image_frames(Path(args.images_dir), image_names=image_names)
        elif args.video:
            if len(gt_boxes_candidate) == _count_video_frames(Path(args.video)):
                gt_boxes_by_frame = gt_boxes_candidate
        if args.images_dir and frames is not None and len(gt_boxes_candidate) == len(frames):
            gt_boxes_by_frame = gt_boxes_candidate

    pipeline = MMBPipeline(config=config)
    if args.images_dir:
        if frames is None:
            frames = _load_image_frames(Path(args.images_dir))
        result = pipeline.run(output_dir=output_dir, gt_boxes_by_frame=gt_boxes_by_frame, frames=frames)
    else:
        result = pipeline.run(video_path=Path(args.video), output_dir=output_dir, gt_boxes_by_frame=gt_boxes_by_frame)

    csv_path = output_dir / 'experiment_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['precision', 'recall', 'ap', 'f1', 'mota', 'motp', 'idf1', 'num_tracks'],
        )
        writer.writeheader()
        writer.writerow(
            {
                'precision': result.metrics['detection']['precision'],
                'recall': result.metrics['detection']['recall'],
                'ap': result.metrics['detection']['ap'],
                'f1': result.metrics['detection']['f1'],
                'mota': result.metrics['tracking']['mota'],
                'motp': result.metrics['tracking']['motp'],
                'idf1': result.metrics['tracking']['idf1'],
                'num_tracks': result.metrics['tracking']['num_tracks'],
            }
        )

    print(json.dumps({'status': 'completed', 'output_dir': str(output_dir), 'csv': str(csv_path)}, indent=2))


if __name__ == '__main__':
    main()
