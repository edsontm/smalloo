from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.mmb.detection import Detection


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    ap: float
    false_alarms: float
    tp: float
    fp: float
    fn: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'precision': float(self.precision),
            'recall': float(self.recall),
            'ap': float(self.ap),
            'false_alarms': float(self.false_alarms),
            'tp': float(self.tp),
            'fp': float(self.fp),
            'fn': float(self.fn),
        }


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1 + 1.0) * (inter_y2 - inter_y1 + 1.0)
    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def evaluate_detections(
    predictions_by_frame: Sequence[Sequence[Detection]],
    ground_truth_by_frame: Sequence[Sequence[Sequence[float]]],
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    flattened_predictions: List[tuple[int, Detection]] = []
    for frame_index, detections in enumerate(predictions_by_frame):
        for detection in detections:
            flattened_predictions.append((frame_index, detection))

    used_ground_truth = [set() for _ in ground_truth_by_frame]
    true_positives = 0
    false_positives = 0
    precision_curve: List[float] = []
    recall_curve: List[float] = []

    total_ground_truth = sum(len(frame_boxes) for frame_boxes in ground_truth_by_frame)
    sorted_predictions = sorted(flattened_predictions, key=lambda item: item[1].confidence, reverse=True)

    for rank, (frame_index, detection) in enumerate(sorted_predictions, start=1):
        best_iou = 0.0
        best_index = None
        for gt_index, gt_box in enumerate(ground_truth_by_frame[frame_index]):
            if gt_index in used_ground_truth[frame_index]:
                continue
            score = _iou(detection.bbox, gt_box)
            if score > best_iou:
                best_iou = score
                best_index = gt_index

        if best_index is not None and best_iou >= iou_threshold:
            used_ground_truth[frame_index].add(best_index)
            true_positives += 1
        else:
            false_positives += 1

        precision_curve.append(true_positives / rank)
        recall_curve.append(true_positives / total_ground_truth if total_ground_truth > 0 else 0.0)

    false_negatives = total_ground_truth - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0

    ap = 0.0
    if total_ground_truth > 0 and precision_curve:
        precision_envelope = [0.0] + precision_curve + [0.0]
        recall_envelope = [0.0] + recall_curve + [1.0]
        for index in range(len(precision_envelope) - 2, -1, -1):
            precision_envelope[index] = max(precision_envelope[index], precision_envelope[index + 1])
        for index in range(len(recall_envelope) - 1):
            ap += (recall_envelope[index + 1] - recall_envelope[index]) * precision_envelope[index + 1]

    return EvaluationResult(
        precision=float(precision),
        recall=float(recall),
        ap=float(ap),
        false_alarms=float(false_positives),
        tp=float(true_positives),
        fp=float(false_positives),
        fn=float(false_negatives),
    )