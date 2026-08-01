from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_mmb import _load_gt_boxes_by_frame, _load_gt_images, _load_image_frames
from src.mmb.pipeline import MMBPipeline
from src.viso_evaluation import evaluate_viso_detection, load_coco_annotations


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('PyYAML is required for configs/mmb.yaml.') from exc
    return yaml.safe_load(path.read_text())


def main() -> None:
    ship_root = ROOT / 'VISO' / 'coco' / 'ship'
    ann_path = ship_root / 'Annotations' / 'instances_test2017.json'
    image_dir = ship_root / 'test2017'
    config = _load_yaml(ROOT / 'configs' / 'mmb.yaml')

    payload = load_coco_annotations(ann_path)
    image_names = _load_gt_images(ann_path)
    frames = _load_image_frames(image_dir, image_names=image_names)
    gt_boxes_by_frame = _load_gt_boxes_by_frame(ann_path)
    category_id = int((payload.get('annotations') or [{}])[0].get('category_id', 1))

    max_images = 32
    frames = frames[:max_images]
    image_names = image_names[:max_images]
    gt_boxes_by_frame = gt_boxes_by_frame[:max_images]
    payload_images = payload['images'][:max_images]
    payload_annotations = [ann for ann in payload['annotations'] if int(ann['image_id']) in {image['id'] for image in payload_images}]

    sweep_values = [0.15, 0.20, 0.25, 0.30, 0.40]
    results: List[Dict[str, Any]] = []
    threshold = 0.30
    for lambda_value in sweep_values:
        run_config = dict(config)
        run_config = json.loads(json.dumps(run_config))
        run_config['foreground']['threshold'] = threshold
        run_config['foreground']['min_component_size'] = 1
        run_config['detection']['min_area'] = 5
        run_config['detection']['max_area'] = 1200
        run_config['multi_frame_difference'] = {'temporal_windows': [1, 2, 3], 'normalize_frames': True}
        run_config['robust_matrix_completion'] = {'lambda_value': lambda_value, 'max_iter': 60, 'tol': 1e-5}

        output_dir = ROOT / 'research' / 'experiments' / 'v7_mmb_ship_sweep' / 'artifacts' / 'results' / f'lambda_{lambda_value:.2f}'
        pipeline = MMBPipeline(run_config)
        result = pipeline.run(frames=frames, output_dir=output_dir, gt_boxes_by_frame=gt_boxes_by_frame)
        pred_annotations: List[Dict[str, Any]] = []
        for image, detections in zip(payload_images, result.detections):
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                pred_annotations.append(
                    {
                        'image_id': image['id'],
                        'category_id': category_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': detection.confidence,
                    }
                )

        evaluation = evaluate_viso_detection(gt_annotations=payload_annotations, pred_annotations=pred_annotations)
        summary = {
            'threshold': threshold,
            'lambda_value': lambda_value,
            'output_dir': str(output_dir),
            'metrics': {
                'precision': evaluation['precision'],
                'recall': evaluation['recall'],
                'f1': evaluation['f1'],
                'ap': evaluation['ap'],
                'mAP': evaluation['mAP'],
                'tp': evaluation['tp'],
                'fp': evaluation['fp'],
                'fn': evaluation['fn'],
            },
            'pipeline_metrics': result.metrics,
        }
        (output_dir / 'summary.json').parent.mkdir(parents=True, exist_ok=True)
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        results.append(summary)

    best = max(results, key=lambda item: item['metrics']['f1'])
    manifest = {'experiment': 'v7_mmb_ship_sweep', 'results': results, 'best': best}
    report_path = ROOT / 'research' / 'experiments' / 'v7_mmb_ship_sweep' / 'artifacts' / 'results' / 'sweep_summary.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()