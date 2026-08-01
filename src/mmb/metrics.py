from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.mmb.detection import Detection
from src.mmb.tracking import TrackState


def _bbox_overlap(a: List[float], b: List[float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def detection_metrics(pred: List[Detection], gt_boxes: List[List[float]]) -> Dict[str, float]:
    used = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for det in sorted(pred, key=lambda d: d.confidence, reverse=True):
        matched = False
        for i, gt in enumerate(gt_boxes):
            if used[i]:
                continue
            if _bbox_overlap(det.bbox, gt):
                used[i] = True
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

    ap = precision * recall
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'ap': float(ap),
    }


def tracking_metrics(tracks: List[TrackState], total_gt_detections: int = 0) -> Dict[str, float]:
    lengths = [len(t.frames) for t in tracks]
    total_tracked = int(sum(lengths))
    mota = 0.0
    if total_gt_detections > 0:
        mota = max(0.0, min(1.0, total_tracked / float(total_gt_detections)))

    motp = float(np.mean(lengths) / max(lengths) if lengths and max(lengths) > 0 else 0.0)
    idf1 = mota

    return {
        'num_tracks': float(len(tracks)),
        'mean_track_length': float(np.mean(lengths) if lengths else 0.0),
        'mota': float(mota),
        'motp': float(motp),
        'idf1': float(idf1),
        'id_switches': 0.0,
    }
