from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class EvalMatch:
    image_id: int
    category_id: int
    score: float
    is_true_positive: bool


def _to_xyxy(bbox: List[float]) -> BBox:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _overlaps(a: BBox, b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _intersection_area(a: BBox, b: BBox) -> float:
    if not _overlaps(a, b):
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _compute_ap(precisions: List[float], recalls: List[float]) -> float:
    if not precisions or not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def _safe_div(numer: float, denom: float) -> float:
    return 0.0 if denom <= 0 else numer / denom


def evaluate_viso_detection(
    gt_annotations: List[Dict[str, Any]],
    pred_annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate detections using VISO overlap criterion (not IoU threshold).

    A prediction is a true positive when its box overlaps any unmatched ground-truth box
    with the same image_id and category_id.
    """

    gt_by_key: DefaultDict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    gt_count_by_class: DefaultDict[int, int] = defaultdict(int)
    classes = set()

    for ann in gt_annotations:
        image_id = int(ann['image_id'])
        category_id = int(ann['category_id'])
        gt_by_key[(image_id, category_id)].append({'bbox': _to_xyxy(ann['bbox']), 'matched': False})
        gt_count_by_class[category_id] += 1
        classes.add(category_id)

    preds_by_class: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in pred_annotations:
        image_id = int(ann['image_id'])
        category_id = int(ann['category_id'])
        score = float(ann.get('score', 1.0))
        preds_by_class[category_id].append(
            {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': _to_xyxy(ann['bbox']),
                'score': score,
            }
        )
        classes.add(category_id)

    per_class: Dict[str, Any] = {}
    ap_values: List[float] = []
    all_matches: List[EvalMatch] = []

    for class_id in sorted(classes):
        class_preds = sorted(preds_by_class[class_id], key=lambda x: x['score'], reverse=True)
        total_gt = gt_count_by_class[class_id]

        tp = 0
        fp = 0
        precisions: List[float] = []
        recalls: List[float] = []

        for pred in class_preds:
            key = (pred['image_id'], class_id)
            candidates = gt_by_key.get(key, [])
            best_index = -1
            best_overlap = 0.0

            for idx, gt in enumerate(candidates):
                if gt['matched']:
                    continue
                overlap_area = _intersection_area(pred['bbox'], gt['bbox'])
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_index = idx

            if best_index >= 0 and best_overlap > 0.0:
                candidates[best_index]['matched'] = True
                tp += 1
                is_tp = True
            else:
                fp += 1
                is_tp = False

            all_matches.append(
                EvalMatch(
                    image_id=pred['image_id'],
                    category_id=class_id,
                    score=pred['score'],
                    is_true_positive=is_tp,
                )
            )

            precisions.append(_safe_div(tp, tp + fp))
            recalls.append(_safe_div(tp, total_gt))

        fn = max(total_gt - tp, 0)
        final_precision = _safe_div(tp, tp + fp)
        final_recall = _safe_div(tp, total_gt)
        final_f1 = _safe_div(2.0 * final_precision * final_recall, final_precision + final_recall)
        ap = _compute_ap(precisions, recalls)
        if total_gt > 0:
            ap_values.append(ap)

        per_class[str(class_id)] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': final_precision,
            'recall': final_recall,
            'f1': final_f1,
            'ap': ap,
            'pr_curve': {'precision': precisions, 'recall': recalls},
        }

    global_tp = sum(v['tp'] for v in per_class.values())
    global_fp = sum(v['fp'] for v in per_class.values())
    global_fn = sum(v['fn'] for v in per_class.values())

    precision = _safe_div(global_tp, global_tp + global_fp)
    recall = _safe_div(global_tp, global_tp + global_fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    matches_sorted = sorted(all_matches, key=lambda m: m.score, reverse=True)
    pr_tp = 0
    pr_fp = 0
    pr_precision: List[float] = []
    pr_recall: List[float] = []
    total_gt_all = sum(gt_count_by_class.values())

    for match in matches_sorted:
        if match.is_true_positive:
            pr_tp += 1
        else:
            pr_fp += 1
        pr_precision.append(_safe_div(pr_tp, pr_tp + pr_fp))
        pr_recall.append(_safe_div(pr_tp, total_gt_all))

    return {
        'protocol': 'viso_overlap_match',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': _compute_ap(pr_precision, pr_recall),
        'mAP': _safe_div(sum(ap_values), len(ap_values)),
        'tp': global_tp,
        'fp': global_fp,
        'fn': global_fn,
        'pr_curve': {'precision': pr_precision, 'recall': pr_recall},
        'per_class': per_class,
    }


def load_coco_annotations(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())