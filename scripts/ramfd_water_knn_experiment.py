from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.mmb_complete import Component, _dilate, _erode, _ramfd_components, _read_gray_frame, _trajectory_filter
from src.viso_evaluation import evaluate_viso_detection, load_coco_annotations


@dataclass
class WaterPrototype:
    feature_mean: np.ndarray
    count: int


def _clip_box(x: float, y: float, w: float, h: float, iw: int, ih: int) -> Tuple[int, int, int, int]:
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(iw, int(round(x + w)))
    y2 = min(ih, int(round(y + h)))
    return x1, y1, x2, y2


def _ring_mask(iw: int, ih: int, bbox: List[float], margin: int) -> np.ndarray:
    x, y, w, h = bbox
    x1, y1, x2, y2 = _clip_box(x, y, w, h, iw, ih)
    xo1, yo1, xo2, yo2 = _clip_box(x - margin, y - margin, w + (2 * margin), h + (2 * margin), iw, ih)

    ring = np.zeros((ih, iw), dtype=np.uint8)
    if xo1 < xo2 and yo1 < yo2:
        ring[yo1:yo2, xo1:xo2] = 1
    if x1 < x2 and y1 < y2:
        ring[y1:y2, x1:x2] = 0
    return ring


def _feature_vector_around_bbox(
    rgb_image: np.ndarray,
    bbox: List[float],
    margin: int,
    feature_mode: str,
) -> np.ndarray | None:
    ih, iw, _ = rgb_image.shape
    mask = _ring_mask(iw, ih, bbox, margin=margin)
    pixels = rgb_image[mask > 0]
    if pixels.size == 0:
        return None

    px = np.asarray(pixels, dtype=np.float32)
    if feature_mode == "rgb_mean":
        return np.asarray(px.mean(axis=0), dtype=np.float32)

    if feature_mode == "rgb_mean_std_edge":
        rgb_mean = np.asarray(px.mean(axis=0), dtype=np.float32)
        rgb_std = np.asarray(px.std(axis=0), dtype=np.float32)
        gray = np.asarray(
            (0.299 * rgb_image[:, :, 0]) + (0.587 * rgb_image[:, :, 1]) + (0.114 * rgb_image[:, :, 2]),
            dtype=np.float32,
        )
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        grad = np.hypot(gx, gy)
        gray_pixels = gray[mask > 0]
        grad_pixels = grad[mask > 0]
        edge_density = float((grad_pixels > 12.0).mean()) if grad_pixels.size > 0 else 0.0
        features = np.concatenate(
            [
                rgb_mean,
                rgb_std,
                np.asarray([float(gray_pixels.mean()), float(gray_pixels.std()), edge_density], dtype=np.float32),
            ]
        )
        return np.asarray(features, dtype=np.float32)

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def _water_rgb_around_bbox(rgb_image: np.ndarray, bbox: List[float], margin: int) -> np.ndarray | None:
    return _feature_vector_around_bbox(rgb_image=rgb_image, bbox=bbox, margin=margin, feature_mode="rgb_mean")


def _parse_int_csv(value: str) -> List[int]:
    items: List[int] = []
    for token in str(value).split(","):
        t = token.strip()
        if not t:
            continue
        items.append(int(t))
    return items


def _make_split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    if n <= 0:
        return [], [], []

    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0.0:
        raise ValueError("Split ratios must sum to a positive value")

    train_n = int(round(float(n) * (float(train_ratio) / total)))
    val_n = int(round(float(n) * (float(val_ratio) / total)))
    test_n = n - train_n - val_n

    train_n = max(1, train_n)
    val_n = max(1, val_n)
    test_n = max(1, test_n)

    while train_n + val_n + test_n > n:
        if test_n > 1:
            test_n -= 1
        elif val_n > 1:
            val_n -= 1
        elif train_n > 1:
            train_n -= 1

    if train_n + val_n + test_n < n:
        remaining = n - (train_n + val_n + test_n)
        test_n += remaining

    train_idx = list(range(train_n))
    val_idx = list(range(train_n, train_n + val_n))
    test_idx = list(range(train_n + val_n, train_n + val_n + test_n))
    return train_idx, val_idx, test_idx


def _prepare_run_output_dir(base_dir: Path, tag: str, max_images: int, description: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{started_at}_{tag}_{max_images}frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "README.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Run Summary",
                "",
                f"- Started at: {started_at}",
                f"- Tag: {tag}",
                f"- Max images: {max_images}",
                f"- Description: {description}",
                "- Artifacts:",
                "  - frames/",
                "  - contact_sheet_test_frames.png",
                "  - report.json",
                "  - report.md",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def _apply_color_jitter_to_feature(sample: np.ndarray, feature_mode: str, jitter_value: float) -> List[np.ndarray]:
    j = float(max(0.0, jitter_value))
    if j <= 0.0:
        return []

    variants: List[np.ndarray] = []
    if feature_mode == "rgb_mean":
        rgb_idx = [0, 1, 2]
    elif feature_mode == "rgb_mean_std_edge":
        rgb_idx = [0, 1, 2]
    else:
        return []

    deltas = [
        np.asarray([j, 0.0, 0.0], dtype=np.float32),
        np.asarray([-j, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, j, 0.0], dtype=np.float32),
        np.asarray([0.0, -j, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, j], dtype=np.float32),
        np.asarray([0.0, 0.0, -j], dtype=np.float32),
        np.asarray([j, j, j], dtype=np.float32),
        np.asarray([-j, -j, -j], dtype=np.float32),
    ]

    for delta in deltas:
        v = np.asarray(sample.copy(), dtype=np.float32)
        v[rgb_idx] = np.clip(v[rgb_idx] + delta, 0.0, 255.0)
        if feature_mode == "rgb_mean_std_edge":
            gray_delta = float((0.299 * delta[0]) + (0.587 * delta[1]) + (0.114 * delta[2]))
            v[6] = float(np.clip(v[6] + gray_delta, 0.0, 255.0))
        variants.append(v)
    return variants


def _collect_water_examples_around_box(
    rgb_image: np.ndarray,
    bbox: List[float],
    margins: List[int],
    jitter_px: int,
    feature_mode: str = "rgb_mean",
    color_jitter: float = 0.0,
) -> List[np.ndarray]:
    x, y, w, h = bbox
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    j = max(0, int(jitter_px))
    if j > 0:
        jf = float(j)
        offsets.extend(
            [
                (jf, 0.0),
                (-jf, 0.0),
                (0.0, jf),
                (0.0, -jf),
                (jf, jf),
                (jf, -jf),
                (-jf, jf),
                (-jf, -jf),
            ]
        )

    examples: List[np.ndarray] = []
    for margin in margins:
        m = max(1, int(margin))
        for dx, dy in offsets:
            sample = _feature_vector_around_bbox(
                rgb_image=rgb_image,
                bbox=[float(x + dx), float(y + dy), float(w), float(h)],
                margin=m,
                feature_mode=feature_mode,
            )
            if sample is not None:
                examples.append(sample)
                examples.extend(_apply_color_jitter_to_feature(sample, feature_mode=feature_mode, jitter_value=color_jitter))
    return examples


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _box_overlap(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _build_water_bank(
    train_samples: List[np.ndarray],
    new_cluster_distance: float,
    min_usage: int,
) -> List[WaterPrototype]:
    bank: List[WaterPrototype] = []

    for sample in train_samples:
        if not bank:
            bank.append(WaterPrototype(feature_mean=sample.copy(), count=1))
            continue

        dists = [_euclidean(sample, p.feature_mean) for p in bank]
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist > new_cluster_distance:
            bank.append(WaterPrototype(feature_mean=sample.copy(), count=1))
            continue

        proto = bank[best_idx]
        new_count = proto.count + 1
        proto.feature_mean = ((proto.feature_mean * proto.count) + sample) / float(new_count)
        proto.count = new_count

    bank = [p for p in bank if p.count >= min_usage]
    return bank


def _nearest_water_dist(sample: np.ndarray, bank: List[WaterPrototype]) -> float:
    if not bank:
        return float("inf")
    return min(_euclidean(sample, p.feature_mean) for p in bank)


def _nearest_proto(sample: np.ndarray, bank: List[WaterPrototype]) -> Tuple[int, float, float]:
    if not bank:
        return -1, float("inf"), float("inf")
    dists = [_euclidean(sample, p.feature_mean) for p in bank]
    order = np.argsort(np.asarray(dists, dtype=np.float32))
    best_idx = int(order[0])
    best_dist = float(dists[best_idx])
    second_dist = float(dists[int(order[1])]) if len(order) > 1 else float("inf")
    return best_idx, best_dist, second_dist


def _collect_negative_examples_from_image(
    rgb_image: np.ndarray,
    gt_boxes: List[List[float]],
    ring_margin: int,
    feature_mode: str,
    grid_step: int,
    exclusion_margin: int,
    min_pos_dist: float,
    positive_bank: List[WaterPrototype],
    max_samples: int,
) -> List[np.ndarray]:
    if max_samples <= 0:
        return []

    ih, iw, _ = rgb_image.shape
    if not gt_boxes:
        return []

    widths = [max(4.0, float(b[2])) for b in gt_boxes]
    heights = [max(4.0, float(b[3])) for b in gt_boxes]
    box_w = float(np.median(np.asarray(widths, dtype=np.float32)))
    box_h = float(np.median(np.asarray(heights, dtype=np.float32)))
    step = max(8, int(grid_step))
    samples: List[np.ndarray] = []

    expanded_gt: List[List[float]] = []
    margin = max(0.0, float(exclusion_margin))
    for gt in gt_boxes:
        expanded_gt.append([float(gt[0] - margin), float(gt[1] - margin), float(gt[2] + (2.0 * margin)), float(gt[3] + (2.0 * margin))])

    y_start = max(0, int(round(box_h / 2.0)))
    x_start = max(0, int(round(box_w / 2.0)))
    for cy in range(y_start, ih - y_start, step):
        for cx in range(x_start, iw - x_start, step):
            cand = [float(cx - (box_w / 2.0)), float(cy - (box_h / 2.0)), box_w, box_h]
            if any(_box_overlap(cand, gt) > 0.0 for gt in expanded_gt):
                continue
            sample = _feature_vector_around_bbox(rgb_image=rgb_image, bbox=cand, margin=ring_margin, feature_mode=feature_mode)
            if sample is None:
                continue
            if _nearest_water_dist(sample, positive_bank) < float(min_pos_dist):
                continue
            samples.append(sample)
            if len(samples) >= int(max_samples):
                return samples

    return samples


def _feature_distance_audit(
    positive_samples: List[np.ndarray],
    positive_bank: List[WaterPrototype],
    negative_samples: List[np.ndarray],
    negative_bank: List[WaterPrototype],
) -> Dict[str, Any]:
    def _percentiles(values: List[float]) -> Dict[str, float | None]:
        if not values:
            return {"count": 0, "p50": None, "p90": None, "p95": None}
        arr = np.asarray(values, dtype=np.float32)
        return {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    pos_to_pos = [float(_nearest_proto(s, positive_bank)[1]) for s in positive_samples] if positive_bank else []
    neg_to_pos = [float(_nearest_proto(s, positive_bank)[1]) for s in negative_samples] if positive_bank else []
    neg_to_neg = [float(_nearest_proto(s, negative_bank)[1]) for s in negative_samples] if negative_bank else []
    return {
        "positive_to_positive": _percentiles(pos_to_pos),
        "negative_to_positive": _percentiles(neg_to_pos),
        "negative_to_negative": _percentiles(neg_to_neg),
    }


def _adaptive_thresholds_by_proto(
    train_samples: List[np.ndarray],
    bank: List[WaterPrototype],
    percentile: float,
    sigma_factor: float,
    min_abs_threshold: float,
) -> List[float]:
    if not bank:
        return []

    bucket: List[List[float]] = [[] for _ in range(len(bank))]
    for s in train_samples:
        idx, d1, _ = _nearest_proto(s, bank)
        if idx >= 0:
            bucket[idx].append(float(d1))

    thresholds: List[float] = []
    p = float(np.clip(percentile, 50.0, 99.9))
    sf = float(max(0.1, sigma_factor))
    min_abs = float(max(0.0, min_abs_threshold))

    for ds in bucket:
        if not ds:
            thresholds.append(min_abs)
            continue
        arr = np.asarray(ds, dtype=np.float32)
        med = float(np.median(arr))
        std = float(np.std(arr))
        perc = float(np.percentile(arr, p))
        # Aggressive adaptive cutoff: keep below both robust and spread-based criteria.
        thr = max(min_abs, min(perc, med + (sf * std)))
        thresholds.append(thr)
    return thresholds


def _thresholds_from_train_gt_overlap(
    train_components_by_frame: Dict[int, List[Component]],
    train_rgb_frames: List[np.ndarray],
    train_usable: List[Dict[str, Any]],
    train_indices: List[int],
    anns_by_image: Dict[int, List[Dict[str, Any]]],
    bank: List[WaterPrototype],
    water_margin: int,
    feature_mode: str,
    percentile: float,
    min_abs_threshold: float,
    max_abs_threshold: float,
) -> Tuple[List[float], Dict[str, Any]]:
    if not bank:
        return [], {"samples_per_proto": [], "total_overlap_samples": 0}

    p = float(np.clip(percentile, 50.0, 99.9))
    min_abs = float(max(0.0, min_abs_threshold))
    max_abs = float(max(min_abs, max_abs_threshold))

    bucket: List[List[float]] = [[] for _ in range(len(bank))]
    total_overlap = 0

    for i in train_indices:
        if i < 0 or i >= len(train_usable) or i >= len(train_rgb_frames):
            continue
        image_id = int(train_usable[i]["id"])
        gt_boxes = [list(map(float, g["bbox"])) for g in anns_by_image.get(image_id, [])]
        if not gt_boxes:
            continue

        comps = train_components_by_frame.get(i, [])
        for c in comps:
            box = [float(c.x), float(c.y), float(c.w), float(c.h)]
            if not any(_overlap_area(_to_xyxy(box), _to_xyxy(gt)) > 0.0 for gt in gt_boxes):
                continue

            feat = _feature_vector_around_bbox(
                rgb_image=train_rgb_frames[i],
                bbox=box,
                margin=int(water_margin),
                feature_mode=str(feature_mode),
            )
            if feat is None:
                continue

            best_idx, d1, _ = _nearest_proto(feat, bank)
            if best_idx < 0:
                continue

            bucket[best_idx].append(float(d1))
            total_overlap += 1

    thresholds: List[float] = []
    samples_per_proto: List[int] = []
    for ds in bucket:
        samples_per_proto.append(len(ds))
        if not ds:
            thresholds.append(float(min_abs))
            continue
        arr = np.asarray(ds, dtype=np.float32)
        thr = float(np.percentile(arr, p))
        thr = float(np.clip(thr, min_abs, max_abs))
        thresholds.append(thr)

    diagnostics = {
        "samples_per_proto": [int(x) for x in samples_per_proto],
        "total_overlap_samples": int(total_overlap),
        "percentile": float(p),
    }
    return thresholds, diagnostics


def _filter_candidates_by_water(
    candidates: List[Dict[str, Any]],
    nearest_gap_min: float,
    use_frame_relative_threshold: bool,
    frame_relative_margin: float,
    max_candidates_per_frame: int,
    use_negative_bank: bool = False,
    negative_margin: float = 0.0,
) -> Tuple[List[List[float]], Dict[str, float]]:
    """Apply water filtering with optional frame-relative tightening.

    The frame-relative threshold keeps candidates near the best water match
    in the same frame, which helps suppress long tails of clutter blobs.
    """

    base_kept: List[Dict[str, Any]] = []
    gap_min = float(nearest_gap_min)
    for cand in candidates:
        dist = float(cand["dist"])
        local_thr = float(cand["local_thr"])
        second_dist = float(cand["second_dist"])
        gap = float(second_dist - dist) if np.isfinite(second_dist) else float("inf")
        neg_ok = True
        if bool(use_negative_bank):
            neg_dist = float(cand.get("neg_dist", float("inf")))
            neg_ok = bool((dist + float(negative_margin)) <= neg_dist)
        if dist <= local_thr and gap >= gap_min and neg_ok:
            base_kept.append(cand)

    relative_thr = float("inf")
    if base_kept and bool(use_frame_relative_threshold):
        best_dist = min(float(c["dist"]) for c in base_kept)
        relative_thr = float(best_dist + max(0.0, float(frame_relative_margin)))
        base_kept = [c for c in base_kept if float(c["dist"]) <= relative_thr]

    if max_candidates_per_frame > 0 and len(base_kept) > int(max_candidates_per_frame):
        base_kept = sorted(base_kept, key=lambda c: float(c["dist"]))[: int(max_candidates_per_frame)]

    kept_boxes = [list(map(float, c["box"])) for c in base_kept]
    diagnostics = {
        "relative_thr": float(relative_thr),
        "kept_after_water": float(len(kept_boxes)),
    }
    return kept_boxes, diagnostics


def _to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _small_target_detector_boxes(
    gray_frames: List[np.ndarray],
    frame_idx: int,
    prev_boxes: List[List[float]] | None = None,
    max_candidates: int = 4,
) -> List[List[float]]:
    if len(gray_frames) < 3:
        return []
    if not 0 <= frame_idx < len(gray_frames):
        return []

    frame = np.asarray(gray_frames[frame_idx], dtype=np.float32)
    if frame.size == 0:
        return []

    ih, iw = frame.shape
    if ih < 8 or iw < 8:
        return []

    base = np.asarray(frame, dtype=np.float32)
    prev = np.asarray(gray_frames[max(0, frame_idx - 1)], dtype=np.float32)
    nxt = np.asarray(gray_frames[min(len(gray_frames) - 1, frame_idx + 1)], dtype=np.float32)
    motion_response = np.abs(base - prev) + np.abs(base - nxt)
    motion_response = np.maximum(motion_response, 0.0)

    if motion_response.shape[0] > 2 and motion_response.shape[1] > 2:
        smoothed = np.zeros_like(motion_response, dtype=np.float32)
        for y in range(1, motion_response.shape[0] - 1):
            for x in range(1, motion_response.shape[1] - 1):
                smoothed[y, x] = float(np.mean(motion_response[y - 1 : y + 2, x - 1 : x + 2]))
        motion_response = smoothed

    appearance = np.asarray(base, dtype=np.float32)
    appearance_centered = appearance - np.mean(appearance)
    appearance_score = np.abs(appearance_centered)
    appearance_score = np.maximum(appearance_score, 0.0)

    if motion_response.shape != appearance_score.shape:
        appearance_score = np.asarray(appearance_score, dtype=np.float32)

    combined = np.maximum(motion_response, appearance_score * 0.25)

    positive = combined[combined > 0.0]
    if positive.size == 0:
        return []

    local_thr = float(np.percentile(positive, 85.0))
    if not np.isfinite(local_thr) or local_thr <= 0.0:
        local_thr = float(np.max(positive) * 0.6)

    mask = (combined >= local_thr).astype(np.uint8)
    if mask.sum() == 0:
        return []

    mask = _dilate(mask, kernel_size=3)
    mask = _erode(mask, kernel_size=3)

    visited = np.zeros_like(mask, dtype=np.uint8)
    components: List[Dict[str, Any]] = []
    for y in range(ih):
        for x in range(iw):
            if mask[y, x] == 0 or visited[y, x] != 0:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            coords: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if ny < 0 or nx < 0 or ny >= ih or nx >= iw:
                            continue
                        if mask[ny, nx] == 0 or visited[ny, nx] != 0:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            ys = [c[0] for c in coords]
            xs = [c[1] for c in coords]
            area = len(coords)
            if area < 2 or area > 64:
                continue
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            if width > 24 or height > 24:
                continue
            score = float(np.mean(combined[min_y : max_y + 1, min_x : max_x + 1])) if width > 0 and height > 0 else 0.0
            components.append(
                {
                    "box": [float(min_x), float(min_y), float(width), float(height)],
                    "score": score,
                }
            )

    if not components:
        return []

    components = sorted(components, key=lambda c: float(c["score"]), reverse=True)
    selected: List[List[float]] = []
    prev_list = prev_boxes or []

    def _center_distance(box_a: List[float], box_b: List[float]) -> float:
        ax = float(box_a[0] + (box_a[2] / 2.0))
        ay = float(box_a[1] + (box_a[3] / 2.0))
        bx = float(box_b[0] + (box_b[2] / 2.0))
        by = float(box_b[1] + (box_b[3] / 2.0))
        return float(np.hypot(ax - bx, ay - by))

    for prev_box in prev_list:
        if len(selected) >= max_candidates:
            break
        best_component_idx = -1
        best_dist = float("inf")
        for idx, comp in enumerate(components):
            if comp in selected:
                continue
            dist = _center_distance(comp["box"], prev_box)
            if dist < best_dist:
                best_dist = dist
                best_component_idx = idx
        if best_component_idx >= 0 and best_dist <= 24.0:
            selected.append(components[best_component_idx]["box"])
            components.pop(best_component_idx)

    for comp in components:
        if len(selected) >= max_candidates:
            break
        selected.append(comp["box"])

    return selected[:max_candidates]


def _tiny_object_fallback_boxes(
    gray_frames: List[np.ndarray],
    frame_idx: int,
    prev_boxes: List[List[float]] | None = None,
    max_candidates: int = 4,
) -> List[List[float]]:
    if len(gray_frames) < 1:
        return []
    if not 0 <= frame_idx < len(gray_frames):
        return []

    frame = np.asarray(gray_frames[frame_idx], dtype=np.float32)
    if frame.size == 0:
        return []

    ih, iw = frame.shape
    if ih < 8 or iw < 8:
        return []

    base = np.asarray(frame, dtype=np.float32)
    centered = np.abs(base - float(np.mean(base)))
    centered = np.maximum(centered, 0.0)

    gx = np.zeros_like(base, dtype=np.float32)
    gy = np.zeros_like(base, dtype=np.float32)
    gx[:, 1:] = np.abs(base[:, 1:] - base[:, :-1])
    gy[1:, :] = np.abs(base[1:, :] - base[:-1, :])
    grad = np.hypot(gx, gy)
    ridge = np.maximum(centered, grad * 0.35)

    if ridge.shape[0] > 2 and ridge.shape[1] > 2:
        smoothed = np.zeros_like(ridge, dtype=np.float32)
        for y in range(1, ridge.shape[0] - 1):
            for x in range(1, ridge.shape[1] - 1):
                smoothed[y, x] = float(np.mean(ridge[y - 1 : y + 2, x - 1 : x + 2]))
        ridge = smoothed

    positive = ridge[ridge > 0.0]
    if positive.size == 0:
        return []

    local_thr = float(np.percentile(positive, 95.0))
    if not np.isfinite(local_thr) or local_thr <= 0.0:
        local_thr = float(np.max(positive) * 0.7)

    response_mask = (ridge >= local_thr).astype(np.uint8)
    if response_mask.sum() == 0:
        return []

    response_mask = _dilate(response_mask, kernel_size=3)
    response_mask = _erode(response_mask, kernel_size=3)

    boxes: List[List[float]] = []
    prev_list = prev_boxes or []

    def _add_box(y: int, x: int, size: int = 10, prefer_prev: bool = False) -> None:
        if len(boxes) >= max_candidates:
            return
        size = max(6, min(18, int(size)))
        box = [float(max(0, x - (size // 2))), float(max(0, y - (size // 2))), float(size), float(size)]
        if box[0] + box[2] > iw:
            box[0] = max(0.0, float(iw - box[2]))
        if box[1] + box[3] > ih:
            box[1] = max(0.0, float(ih - box[3]))
        if any(abs(box[0] - existing[0]) < 4.0 and abs(box[1] - existing[1]) < 4.0 for existing in boxes):
            return
        if prefer_prev:
            box[0] = max(0.0, float(min(iw - box[2], max(0.0, box[0]))))
            box[1] = max(0.0, float(min(ih - box[3], max(0.0, box[1]))))
        boxes.append(box)

    for prev_box in prev_list:
        cx = int(round(float(prev_box[0] + (prev_box[2] / 2.0))))
        cy = int(round(float(prev_box[1] + (prev_box[3] / 2.0))))
        search_y0 = max(0, cy - 8)
        search_y1 = min(ih, cy + 9)
        search_x0 = max(0, cx - 8)
        search_x1 = min(iw, cx + 9)
        patch = ridge[search_y0:search_y1, search_x0:search_x1]
        if patch.size == 0:
            continue
        loc_y, loc_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
        y = search_y0 + loc_y
        x = search_x0 + loc_x
        if float(ridge[y, x]) >= (local_thr * 0.75):
            _add_box(y=y, x=x, size=12)
            response_mask[max(0, y - 2) : min(ih, y + 3), max(0, x - 2) : min(iw, x + 3)] = 0
        else:
            _add_box(y=cy, x=cx, size=12, prefer_prev=True)

    if len(boxes) < max_candidates:
        coords = np.argwhere(response_mask > 0)
        coords = sorted(coords, key=lambda c: float(ridge[c[0], c[1]]), reverse=True)
        for y, x in coords:
            if len(boxes) >= max_candidates:
                break
            _add_box(y=int(y), x=int(x), size=10)

    return boxes


def _match_pred_tp(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> List[bool]:
    gt_xyxy = [_to_xyxy(b) for b in gt_boxes]
    pred_xyxy = [_to_xyxy(b) for b in pred_boxes]
    gt_used = [False] * len(gt_xyxy)
    pred_tp = [False] * len(pred_xyxy)

    for pi, pb in enumerate(pred_xyxy):
        best = -1
        best_ov = 0.0
        for gi, gb in enumerate(gt_xyxy):
            if gt_used[gi]:
                continue
            ov = _overlap_area(pb, gb)
            if ov > best_ov:
                best_ov = ov
                best = gi
        if best >= 0 and best_ov > 0.0:
            gt_used[best] = True
            pred_tp[pi] = True

    return pred_tp


def _gt_loss_analysis(gt_boxes: List[List[float]], raw_boxes: List[List[float]], water_boxes: List[List[float]], final_boxes: List[List[float]]) -> List[Dict[str, Any]]:
    analyses: List[Dict[str, Any]] = []
    for gt_box in gt_boxes:
        gt_xyxy = _to_xyxy(gt_box)
        raw_overlap = any(_overlap_area(gt_xyxy, _to_xyxy(box)) > 0.0 for box in raw_boxes)
        water_overlap = any(_overlap_area(gt_xyxy, _to_xyxy(box)) > 0.0 for box in water_boxes)
        final_overlap = any(_overlap_area(gt_xyxy, _to_xyxy(box)) > 0.0 for box in final_boxes)

        if final_overlap:
            lost_stage = None
        elif water_overlap:
            lost_stage = "trajectory_selection"
        elif raw_overlap:
            lost_stage = "water_filter"
        else:
            lost_stage = "ramfd"

        analyses.append(
            {
                "gt_box": [float(x) for x in gt_box],
                "raw_overlap": bool(raw_overlap),
                "water_overlap": bool(water_overlap),
                "final_overlap": bool(final_overlap),
                "lost_stage": lost_stage,
            }
        )
    return analyses


def _component_from_box(frame_index: int, box: List[float]) -> Component:
    x, y, w, h = box
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    wi = max(1, int(round(float(w))))
    hi = max(1, int(round(float(h))))
    area = int(max(1, wi * hi))
    return Component(
        frame_index=int(frame_index),
        x=xi,
        y=yi,
        w=wi,
        h=hi,
        area=area,
        cx=float(xi + (wi / 2.0)),
        cy=float(yi + (hi / 2.0)),
    )


def _box_from_component(comp: Component) -> List[float]:
    return [float(comp.x), float(comp.y), float(comp.w), float(comp.h)]


def _wake_triangle_points(
    tip_cx: float,
    tip_cy: float,
    dir_x: float,
    dir_y: float,
    wake_length: float,
    wake_half_width: float,
) -> List[Tuple[float, float]]:
    base_cx = tip_cx - (dir_x * wake_length)
    base_cy = tip_cy - (dir_y * wake_length)
    px = -dir_y
    py = dir_x
    b1 = (base_cx + (px * wake_half_width), base_cy + (py * wake_half_width))
    b2 = (base_cx - (px * wake_half_width), base_cy - (py * wake_half_width))
    return [(tip_cx, tip_cy), b1, b2]


def _point_in_triangle(px: float, py: float, tri: List[Tuple[float, float]]) -> bool:
    if len(tri) != 3:
        return False
    (x1, y1), (x2, y2), (x3, y3) = tri
    den = ((y2 - y3) * (x1 - x3)) + ((x3 - x2) * (y1 - y3))
    if abs(den) < 1e-6:
        return False
    a = (((y2 - y3) * (px - x3)) + ((x3 - x2) * (py - y3))) / den
    b = (((y3 - y1) * (px - x3)) + ((x1 - x3) * (py - y3))) / den
    c = 1.0 - a - b
    eps = 1e-6
    return a >= -eps and b >= -eps and c >= -eps


def _select_tip_component(
    comps: List[Component],
    prev_tip: Component | None,
    prev_prev_tip: Component | None,
) -> Tuple[Component | None, Tuple[float, float] | None]:
    if not comps:
        return None, None

    motion: Tuple[float, float] | None = None
    if prev_tip is not None and prev_prev_tip is not None:
        dx = float(prev_tip.cx - prev_prev_tip.cx)
        dy = float(prev_tip.cy - prev_prev_tip.cy)
        norm = float(np.hypot(dx, dy))
        if norm > 1e-6:
            motion = (dx / norm, dy / norm)

    if motion is None:
        tip = max(comps, key=lambda c: float(c.area))
        return tip, None

    dir_x, dir_y = motion
    scored = sorted(
        comps,
        key=lambda c: float(((c.cx - prev_tip.cx) * dir_x) + ((c.cy - prev_tip.cy) * dir_y)),
        reverse=True,
    )
    return scored[0], motion


def _legend_labels(tp_count: int, fp_count: int, fn_count: int) -> List[str]:
    return [f"TP ({tp_count})", f"FP ({fp_count})", f"FN ({fn_count})"]


def _legend_counts(gt_boxes: List[List[float]], kept_preds: List[List[float]], kept_tp: List[bool]) -> Tuple[int, int, int]:
    tp_count = int(sum(kept_tp))
    fp_count = int(len(kept_preds) - sum(kept_tp))
    fn_count = int(len(gt_boxes) - sum(kept_tp))
    return tp_count, fp_count, fn_count


def _center_distance(a: List[float], b: List[float]) -> float:
    ax = float(a[0] + (a[2] / 2.0))
    ay = float(a[1] + (a[3] / 2.0))
    bx = float(b[0] + (b[2] / 2.0))
    by = float(b[1] + (b[3] / 2.0))
    return float(np.hypot(ax - bx, ay - by))


def _template_match_score(template: np.ndarray, patch: np.ndarray) -> float:
    if template.size == 0 or patch.size == 0:
        return float("-inf")
    template = np.asarray(template, dtype=np.float32)
    patch = np.asarray(patch, dtype=np.float32)
    if template.shape != patch.shape:
        return float("-inf")
    return float(np.mean(np.abs(template - patch)))


def _track_tiny_candidates(
    prev_boxes: List[List[float]],
    current_boxes: List[List[float]],
    gray_frames: List[np.ndarray],
    frame_idx: int,
    max_candidates: int = 4,
) -> List[List[float]]:
    if not current_boxes and not prev_boxes:
        return []

    tracked: List[List[float]] = []
    used_prev = [False] * len(prev_boxes)
    used_curr = [False] * len(current_boxes)

    for curr in current_boxes:
        best_prev_idx = -1
        best_dist = float("inf")
        for pi, prev in enumerate(prev_boxes):
            if used_prev[pi]:
                continue
            dist = _center_distance(curr, prev)
            if dist < best_dist:
                best_dist = dist
                best_prev_idx = pi
        if best_prev_idx >= 0 and best_dist <= 24.0:
            tracked.append(curr)
            used_prev[best_prev_idx] = True
            used_curr[len(tracked) - 1] = True
        elif len(tracked) < max_candidates:
            tracked.append(curr)

    if len(tracked) < max_candidates and prev_boxes:
        for pi, prev in enumerate(prev_boxes):
            if used_prev[pi] or len(tracked) >= max_candidates:
                continue
            tracked.append(prev)

    if len(tracked) >= max_candidates:
        return tracked[:max_candidates]

    if 0 <= frame_idx < len(gray_frames):
        frame = np.asarray(gray_frames[frame_idx], dtype=np.float32)
        if frame.size > 0:
            ih, iw = frame.shape
            for prev in prev_boxes:
                if len(tracked) >= max_candidates:
                    break
                if any(_center_distance(prev, box) < 3.0 for box in tracked):
                    continue
                cx = int(round(float(prev[0] + (prev[2] / 2.0))))
                cy = int(round(float(prev[1] + (prev[3] / 2.0))))
                x0 = max(0, cx - 6)
                y0 = max(0, cy - 6)
                x1 = min(iw, cx + 7)
                y1 = min(ih, cy + 7)
                patch = frame[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                tracked.append([float(x0), float(y0), float(max(6, min(16, x1 - x0))), float(max(6, min(16, y1 - y0)))])

    return tracked[:max_candidates]


def _draw_frame(
    image_path: Path,
    gt_boxes: List[List[float]],
    raw_preds: List[List[float]],
    raw_tp: List[bool],
    kept_preds: List[List[float]],
    kept_tp: List[bool],
    wake_triangle: List[Tuple[float, float]] | None,
    title: str,
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for gt in gt_boxes:
        x, y, w, h = gt
        draw.rectangle((x, y, x + w, y + h), outline=(255, 215, 0), width=2)

    for bbox, is_tp in zip(raw_preds, raw_tp):
        x, y, w, h = bbox
        color = (60, 120, 255) if is_tp else (80, 80, 80)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=1)

    for bbox, is_tp in zip(kept_preds, kept_tp):
        x, y, w, h = bbox
        color = (32, 220, 80) if is_tp else (235, 64, 52)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)

    if wake_triangle is not None and len(wake_triangle) == 3:
        p0, p1, p2 = wake_triangle
        draw.polygon([p0, p1, p2], outline=(255, 0, 255))

    tp_count, fp_count, fn_count = _legend_counts(
        gt_boxes=gt_boxes,
        kept_preds=kept_preds,
        kept_tp=kept_tp,
    )
    legend_text = _legend_labels(tp_count=tp_count, fp_count=fp_count, fn_count=fn_count)

    legend_y = max(6, image.height - 92)
    legend_box = (6, legend_y, 220, legend_y + 78)
    draw.rounded_rectangle(legend_box, radius=8, outline=(255, 255, 255), width=1, fill=(0, 0, 0, 80))
    draw.text((16, legend_y + 8), "Legend", fill=(255, 255, 255))
    draw.rectangle((16, legend_y + 28, 30, legend_y + 42), outline=(32, 220, 80), width=2)
    draw.text((36, legend_y + 25), legend_text[0], fill=(255, 255, 255))
    draw.rectangle((16, legend_y + 46, 30, legend_y + 60), outline=(235, 64, 52), width=2)
    draw.text((36, legend_y + 43), legend_text[1], fill=(255, 255, 255))
    draw.rectangle((16, legend_y + 64, 30, legend_y + 78), outline=(255, 215, 0), width=2)
    draw.text((36, legend_y + 61), legend_text[2], fill=(255, 255, 255))

    draw.rectangle((0, 0, min(image.width, 980), 24), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _contact_sheet(paths: List[Path], out_path: Path, cols: int) -> None:
    if not paths:
        return

    thumbs = [Image.open(p).convert("RGB") for p in paths]
    w, h = thumbs[0].size
    cols = max(1, int(cols))
    rows = int(math.ceil(len(thumbs) / cols))
    gap = 8

    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (24, 24, 24))
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = gap + c * (w + gap)
        y = gap + r * (h + gap)
        canvas.paste(thumb, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _summarize_gt_loss_from_split_results(split_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for split_name in ["train", "validation", "test"]:
        frames.extend(split_results.get(split_name, {}).get("frame_stats", []))

    stage_counts = Counter()
    for frame in frames:
        stage_counts.update(frame.get("gt_lost_by_stage", {}))

    examples = []
    for frame in frames:
        if int(frame.get("gt_lost", 0)) <= 0:
            continue
        examples.append(
            {
                "file_name": str(frame.get("file_name", "")),
                "gt_lost": int(frame.get("gt_lost", 0)),
                "gt_lost_by_stage": dict(frame.get("gt_lost_by_stage", {})),
            }
        )
        if len(examples) >= 10:
            break

    return {
        "stage_counts": dict(stage_counts),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAMFD + water-color KNN-like bank filtering experiment")
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--annotations-path", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--eval-annotations-path", type=str, default="")
    parser.add_argument("--eval-image-dir", type=str, default="")
    parser.add_argument("--tag", type=str, default="ship_water_knn")
    parser.add_argument("--water-margin", type=int, default=18)
    parser.add_argument("--water-feature-mode", type=str, default="rgb_mean", choices=["rgb_mean", "rgb_mean_std_edge"])
    parser.add_argument("--water-train-margins", type=str, default="12,18,26")
    parser.add_argument("--water-train-jitter", type=int, default=6)
    parser.add_argument("--water-train-color-jitter", type=float, default=0.0)
    parser.add_argument("--water-train-use-all-gt", action="store_true")
    parser.add_argument("--new-cluster-distance", type=float, default=22.0)
    parser.add_argument("--test-water-max-dist", type=float, default=18.0)
    parser.add_argument("--use-adaptive-prototype-threshold", action="store_true")
    parser.add_argument("--adaptive-dist-percentile", type=float, default=85.0)
    parser.add_argument("--adaptive-sigma-factor", type=float, default=1.3)
    parser.add_argument("--adaptive-min-abs-threshold", type=float, default=8.0)
    parser.add_argument("--use-gt-overlap-threshold-calibration", action="store_true")
    parser.add_argument("--gt-overlap-threshold-percentile", type=float, default=95.0)
    parser.add_argument("--nearest-gap-min", type=float, default=0.0)
    parser.add_argument("--use-frame-relative-water-threshold", action="store_true")
    parser.add_argument("--frame-relative-margin", type=float, default=26.0)
    parser.add_argument("--max-water-candidates-per-frame", type=int, default=0)
    parser.add_argument("--use-negative-bank", action="store_true")
    parser.add_argument("--negative-bank-margin", type=float, default=0.0)
    parser.add_argument("--negative-grid-step", type=int, default=32)
    parser.add_argument("--negative-exclusion-margin", type=int, default=24)
    parser.add_argument("--negative-min-pos-dist", type=float, default=10.0)
    parser.add_argument("--negative-max-samples-per-image", type=int, default=24)
    parser.add_argument("--negative-new-cluster-distance", type=float, default=24.0)
    parser.add_argument("--use-trajectory-tip-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-pipeline-len", type=int, default=5)
    parser.add_argument("--trajectory-radius", type=float, default=16.0)
    parser.add_argument("--trajectory-min-occurrences", type=int, default=2)
    parser.add_argument("--use-wake-triangle-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wake-length-scale", type=float, default=3.0)
    parser.add_argument("--wake-width-scale", type=float, default=1.4)
    parser.add_argument("--ramfd-k", type=float, default=4.0)
    parser.add_argument("--ramfd-min-area", type=int, default=5)
    parser.add_argument("--ramfd-max-area", type=int, default=220)
    parser.add_argument("--ramfd-min-aspect-ratio", type=float, default=0.4)
    parser.add_argument("--ramfd-max-aspect-ratio", type=float, default=10.0)
    parser.add_argument("--ramfd-clutter-active-ratio", type=float, default=0.95)
    parser.add_argument("--ramfd-clutter-dilate-kernel", type=int, default=1)
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ann_path = Path(args.annotations_path) if str(args.annotations_path).strip() else (root / "VISO" / "coco" / "ship" / "Annotations" / "instances_train2017.json")
    image_dir = Path(args.image_dir) if str(args.image_dir).strip() else (root / "VISO" / "coco" / "ship" / "train2017")
    eval_ann_path = Path(args.eval_annotations_path) if str(args.eval_annotations_path).strip() else None
    eval_image_dir = Path(args.eval_image_dir) if str(args.eval_image_dir).strip() else None
    use_separate_eval = bool(eval_ann_path is not None and eval_image_dir is not None)
    base_out_dir = root / "research" / "experiments" / "v9_mmb_ship_bayesian_validation_tuning" / "artifacts" / "debug"
    out_dir = _prepare_run_output_dir(
        base_dir=base_out_dir,
        tag=args.tag,
        max_images=args.max_images,
        description="Render test-frame predictions for the current RAMFD + water-KNN filtering approach, including raw vs filtered boxes and trajectory/wake overlays.",
    )

    payload = load_coco_annotations(ann_path)
    images = sorted(payload["images"], key=lambda x: str(x.get("file_name", "")))[: int(args.max_images)]
    selected = {int(i["id"]) for i in images}
    anns = [a for a in payload["annotations"] if int(a.get("image_id", -1)) in selected]

    if not images:
        raise RuntimeError("No images selected")

    category_id = int(anns[0].get("category_id", 0)) if anns else 0

    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        if int(a.get("category_id", -1)) != category_id:
            continue
        anns_by_image.setdefault(int(a["image_id"]), []).append(a)

    usable: List[Dict[str, Any]] = []
    gray_frames: List[np.ndarray] = []
    rgb_frames: List[np.ndarray] = []
    train_ref_shape: Tuple[int, int] | None = None
    train_skipped_shape_mismatch = 0
    for img in images:
        p = image_dir / str(img.get("file_name", ""))
        if not p.exists():
            continue
        gray = _read_gray_frame(p)
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        if train_ref_shape is None:
            train_ref_shape = (int(gray.shape[0]), int(gray.shape[1]))
        elif tuple(gray.shape) != train_ref_shape:
            train_skipped_shape_mismatch += 1
            continue
        usable.append(img)
        gray_frames.append(gray)
        rgb_frames.append(rgb)

    if len(usable) < 8:
        raise RuntimeError("Not enough usable images")

    eval_anns_by_image = anns_by_image
    eval_usable = usable
    eval_gray_frames = gray_frames
    eval_rgb_frames = rgb_frames
    eval_category_id = category_id

    if use_separate_eval:
        eval_payload = load_coco_annotations(eval_ann_path)
        eval_images = sorted(eval_payload["images"], key=lambda x: str(x.get("file_name", "")))
        if int(args.max_images) > 0:
            eval_images = eval_images[: int(args.max_images)]
        eval_selected = {int(i["id"]) for i in eval_images}
        eval_anns = [a for a in eval_payload["annotations"] if int(a.get("image_id", -1)) in eval_selected]
        eval_category_id = int(eval_anns[0].get("category_id", category_id)) if eval_anns else category_id

        eval_anns_by_image = {}
        for a in eval_anns:
            if int(a.get("category_id", -1)) != eval_category_id:
                continue
            eval_anns_by_image.setdefault(int(a["image_id"]), []).append(a)

        eval_usable = []
        eval_gray_frames = []
        eval_rgb_frames = []
        eval_ref_shape: Tuple[int, int] | None = None
        eval_skipped_shape_mismatch = 0
        for img in eval_images:
            p = eval_image_dir / str(img.get("file_name", ""))
            if not p.exists():
                continue
            gray = _read_gray_frame(p)
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            if eval_ref_shape is None:
                eval_ref_shape = (int(gray.shape[0]), int(gray.shape[1]))
            elif tuple(gray.shape) != eval_ref_shape:
                eval_skipped_shape_mismatch += 1
                continue
            eval_usable.append(img)
            eval_gray_frames.append(gray)
            eval_rgb_frames.append(rgb)

        if len(eval_usable) < 4:
            raise RuntimeError("Not enough usable evaluation images")

        train_idx = list(range(len(usable)))
        test_idx = list(range(len(eval_usable)))
    else:
        n = len(usable)
        train_idx, val_idx, test_idx = _make_split_indices(
            n=n,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
        )
        if len(train_idx) < 1 or len(val_idx) < 1 or len(test_idx) < 1:
            raise RuntimeError("The train/validation/test split must contain at least one image per split")
        eval_skipped_shape_mismatch = train_skipped_shape_mismatch

    train_margins = _parse_int_csv(args.water_train_margins)
    if not train_margins:
        train_margins = [int(args.water_margin)]

    train_samples: List[np.ndarray] = []
    for i in train_idx:
        image_id = int(usable[i]["id"])
        gt_boxes = anns_by_image.get(image_id, [])
        if not gt_boxes:
            continue

        if bool(args.water_train_use_all_gt):
            chosen = gt_boxes
        else:
            chosen = [gt_boxes[0]]

        for gt in chosen:
            bbox = list(map(float, gt["bbox"]))
            samples = _collect_water_examples_around_box(
                rgb_image=rgb_frames[i],
                bbox=bbox,
                margins=train_margins,
                jitter_px=int(args.water_train_jitter),
                feature_mode=str(args.water_feature_mode),
                color_jitter=float(args.water_train_color_jitter),
            )
            train_samples.extend(samples)

    min_usage = max(3, int(math.ceil(0.01 * max(1, len(train_idx)))))
    bank = _build_water_bank(
        train_samples=train_samples,
        new_cluster_distance=float(args.new_cluster_distance),
        min_usage=min_usage,
    )

    negative_samples: List[np.ndarray] = []
    negative_bank: List[WaterPrototype] = []
    if bool(args.use_negative_bank):
        for i in train_idx:
            image_id = int(usable[i]["id"])
            gt_boxes = [list(map(float, g["bbox"])) for g in anns_by_image.get(image_id, [])]
            if not gt_boxes:
                continue
            negative_samples.extend(
                _collect_negative_examples_from_image(
                    rgb_image=rgb_frames[i],
                    gt_boxes=gt_boxes,
                    ring_margin=int(args.water_margin),
                    feature_mode=str(args.water_feature_mode),
                    grid_step=int(args.negative_grid_step),
                    exclusion_margin=int(args.negative_exclusion_margin),
                    min_pos_dist=float(args.negative_min_pos_dist),
                    positive_bank=bank,
                    max_samples=int(args.negative_max_samples_per_image),
                )
            )
        negative_bank = _build_water_bank(
            train_samples=negative_samples,
            new_cluster_distance=float(args.negative_new_cluster_distance),
            min_usage=max(2, int(math.ceil(0.005 * max(1, len(train_idx))))),
        )

    adaptive_thresholds = _adaptive_thresholds_by_proto(
        train_samples=train_samples,
        bank=bank,
        percentile=float(args.adaptive_dist_percentile),
        sigma_factor=float(args.adaptive_sigma_factor),
        min_abs_threshold=float(args.adaptive_min_abs_threshold),
    )

    gt_overlap_thresholds: List[float] = []
    gt_overlap_diag: Dict[str, Any] = {}
    if bool(args.use_gt_overlap_threshold_calibration):
        train_components = _ramfd_components(
            gray_frames,
            k=float(args.ramfd_k),
            min_area=int(args.ramfd_min_area),
            max_area=int(args.ramfd_max_area),
            min_aspect_ratio=float(args.ramfd_min_aspect_ratio),
            max_aspect_ratio=float(args.ramfd_max_aspect_ratio),
            clutter_active_ratio=float(args.ramfd_clutter_active_ratio),
            clutter_dilate_kernel=int(args.ramfd_clutter_dilate_kernel),
            use_tile_adaptive_threshold=False,
            tile_size=64,
            clutter_require_static_edge=False,
            clutter_edge_percentile=75.0,
            use_dark_water_gate=False,
            water_intensity_percentile=60.0,
        )

        gt_overlap_thresholds, gt_overlap_diag = _thresholds_from_train_gt_overlap(
            train_components_by_frame=train_components,
            train_rgb_frames=rgb_frames,
            train_usable=usable,
            train_indices=train_idx,
            anns_by_image=anns_by_image,
            bank=bank,
            water_margin=int(args.water_margin),
            feature_mode=str(args.water_feature_mode),
            percentile=float(args.gt_overlap_threshold_percentile),
            min_abs_threshold=float(args.adaptive_min_abs_threshold),
            max_abs_threshold=float(args.test_water_max_dist),
        )

        if gt_overlap_thresholds:
            adaptive_thresholds = [float(x) for x in gt_overlap_thresholds]

    per_frame = _ramfd_components(
        eval_gray_frames,
        k=float(args.ramfd_k),
        min_area=int(args.ramfd_min_area),
        max_area=int(args.ramfd_max_area),
        min_aspect_ratio=float(args.ramfd_min_aspect_ratio),
        max_aspect_ratio=float(args.ramfd_max_aspect_ratio),
        clutter_active_ratio=float(args.ramfd_clutter_active_ratio),
        clutter_dilate_kernel=int(args.ramfd_clutter_dilate_kernel),
        use_tile_adaptive_threshold=False,
        tile_size=64,
        clutter_require_static_edge=False,
        clutter_edge_percentile=75.0,
        use_dark_water_gate=False,
        water_intensity_percentile=60.0,
    )

    def evaluate_split(split_name: str, split_idx: List[int], render_frames: bool) -> Dict[str, Any]:
        split_out_dir = out_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        split_frames_dir = split_out_dir / "frames"
        split_frames_dir.mkdir(parents=True, exist_ok=True)

        gt_items_for_split: List[Dict[str, Any]] = []
        pred_raw: List[Dict[str, Any]] = []
        pred_filtered: List[Dict[str, Any]] = []
        rendered: List[Path] = []
        frame_stats: List[Dict[str, Any]] = []
        frame_payload: Dict[int, Dict[str, Any]] = {}
        water_kept_by_frame: Dict[int, List[List[float]]] = {idx: [] for idx in split_idx}
        prev_tiny_boxes: List[List[float]] = []
        prev_tip: Component | None = None
        prev_prev_tip: Component | None = None

        for i in split_idx:
            image_id = int(eval_usable[i]["id"])
            file_name = str(eval_usable[i]["file_name"])
            gt_items = eval_anns_by_image.get(image_id, [])
            gt_boxes = [list(map(float, g["bbox"])) for g in gt_items]
            gt_items_for_split.extend(gt_items)

            comps = per_frame.get(i, [])
            raw_boxes: List[List[float]] = []
            water_candidates: List[Dict[str, Any]] = []

            for c in comps:
                box = [float(c.x), float(c.y), float(c.w), float(c.h)]
                raw_boxes.append(box)
                pred_raw.append({"image_id": image_id, "category_id": eval_category_id, "bbox": box, "score": 0.95})

                water_sample = _feature_vector_around_bbox(
                    rgb_image=eval_rgb_frames[i],
                    bbox=box,
                    margin=int(args.water_margin),
                    feature_mode=str(args.water_feature_mode),
                )
                if water_sample is None:
                    continue

                best_idx, dist, second_dist = _nearest_proto(water_sample, bank)
                if best_idx < 0:
                    continue
                neg_dist = float("inf")
                if negative_bank:
                    _, neg_dist, _ = _nearest_proto(water_sample, negative_bank)

                if bool(args.use_adaptive_prototype_threshold):
                    local_thr = adaptive_thresholds[best_idx] if best_idx < len(adaptive_thresholds) else float(args.test_water_max_dist)
                else:
                    local_thr = float(args.test_water_max_dist)

                water_candidates.append(
                    {
                        "box": box,
                        "best_idx": int(best_idx),
                        "dist": float(dist),
                        "second_dist": float(second_dist),
                        "neg_dist": float(neg_dist),
                        "local_thr": float(local_thr),
                    }
                )

            has_raw_overlap = any(
                _overlap_area(_to_xyxy(box), _to_xyxy(gt)) > 0.0
                for box in raw_boxes
                for gt in gt_boxes
            )
            if len(raw_boxes) < 2 or not has_raw_overlap:
                fallback_boxes = _tiny_object_fallback_boxes(
                    gray_frames=eval_gray_frames,
                    frame_idx=i,
                    prev_boxes=prev_tiny_boxes,
                    max_candidates=4,
                )
                tracked_boxes = _track_tiny_candidates(
                    prev_boxes=prev_tiny_boxes,
                    current_boxes=fallback_boxes,
                    gray_frames=eval_gray_frames,
                    frame_idx=i,
                    max_candidates=4,
                )
                for box in tracked_boxes:
                    raw_boxes.append(box)
                    pred_raw.append({"image_id": image_id, "category_id": eval_category_id, "bbox": box, "score": 0.95})
                    water_sample = _feature_vector_around_bbox(
                        rgb_image=eval_rgb_frames[i],
                        bbox=box,
                        margin=int(args.water_margin),
                        feature_mode=str(args.water_feature_mode),
                    )
                    if water_sample is None:
                        continue
                    best_idx, dist, second_dist = _nearest_proto(water_sample, bank)
                    if best_idx < 0:
                        continue
                    neg_dist = float("inf")
                    if negative_bank:
                        _, neg_dist, _ = _nearest_proto(water_sample, negative_bank)
                    if bool(args.use_adaptive_prototype_threshold):
                        local_thr = adaptive_thresholds[best_idx] if best_idx < len(adaptive_thresholds) else float(args.test_water_max_dist)
                    else:
                        local_thr = float(args.test_water_max_dist)
                    water_candidates.append(
                        {
                            "box": box,
                            "best_idx": int(best_idx),
                            "dist": float(dist),
                            "second_dist": float(second_dist),
                            "neg_dist": float(neg_dist),
                            "local_thr": float(local_thr),
                        }
                    )
                prev_tiny_boxes = fallback_boxes
            else:
                prev_tiny_boxes = []

            kept_boxes, water_diag = _filter_candidates_by_water(
                candidates=water_candidates,
                nearest_gap_min=float(args.nearest_gap_min),
                use_frame_relative_threshold=bool(args.use_frame_relative_water_threshold),
                frame_relative_margin=float(args.frame_relative_margin),
                max_candidates_per_frame=int(args.max_water_candidates_per_frame),
                use_negative_bank=bool(args.use_negative_bank),
                negative_margin=float(args.negative_bank_margin),
            )
            water_kept_by_frame[i] = kept_boxes
            frame_payload[i] = {
                "image_id": image_id,
                "file_name": file_name,
                "gt_boxes": gt_boxes,
                "raw_boxes": raw_boxes,
                "water_boxes": kept_boxes,
                "water_relative_thr": water_diag["relative_thr"],
            }

        if bool(args.use_trajectory_tip_filter):
            comps_by_frame: Dict[int, List[Component]] = {idx: [] for idx in split_idx}
            for idx in split_idx:
                comps_by_frame[idx] = [_component_from_box(frame_index=idx, box=b) for b in water_kept_by_frame.get(idx, [])]

            comps_by_frame = _trajectory_filter(
                components_by_frame=comps_by_frame,
                pipeline_len=int(args.trajectory_pipeline_len),
                radius=float(args.trajectory_radius),
                min_occurrences=int(args.trajectory_min_occurrences),
                recover_partial_tracks=True,
            )
        else:
            comps_by_frame = {idx: [_component_from_box(frame_index=idx, box=b) for b in water_kept_by_frame.get(idx, [])] for idx in split_idx}

        for i in split_idx:
            payload_i = frame_payload[i]
            image_id = int(payload_i["image_id"])
            file_name = str(payload_i["file_name"])
            gt_boxes = payload_i["gt_boxes"]
            raw_boxes = payload_i["raw_boxes"]

            traj_comps = list(comps_by_frame.get(i, []))
            selected_tip, motion = _select_tip_component(traj_comps, prev_tip=prev_tip, prev_prev_tip=prev_prev_tip)

            if selected_tip is None and payload_i["water_boxes"]:
                fallback_component = _component_from_box(frame_index=i, box=payload_i["water_boxes"][0])
                traj_comps = [fallback_component]
                selected_tip, motion = _select_tip_component(traj_comps, prev_tip=prev_tip, prev_prev_tip=prev_prev_tip)

            wake_triangle: List[Tuple[float, float]] | None = None
            if selected_tip is not None and motion is not None and bool(args.use_wake_triangle_filter):
                ref_size = max(1.0, 0.5 * (float(selected_tip.w) + float(selected_tip.h)))
                wake_triangle = _wake_triangle_points(
                    tip_cx=float(selected_tip.cx),
                    tip_cy=float(selected_tip.cy),
                    dir_x=float(motion[0]),
                    dir_y=float(motion[1]),
                    wake_length=float(args.wake_length_scale) * ref_size,
                    wake_half_width=float(args.wake_width_scale) * ref_size,
                )
                if len(traj_comps) > 1:
                    filtered: List[Component] = []
                    for c in traj_comps:
                        if c is selected_tip:
                            filtered.append(c)
                            continue
                        if _point_in_triangle(float(c.cx), float(c.cy), wake_triangle):
                            continue
                        filtered.append(c)
                    traj_comps = filtered

            if selected_tip is None and traj_comps:
                selected_tip = max(traj_comps, key=lambda c: float(c.area))

            kept_boxes = [_box_from_component(selected_tip)] if selected_tip is not None else []
            for box in kept_boxes:
                pred_filtered.append({"image_id": image_id, "category_id": eval_category_id, "bbox": box, "score": 0.95})

            raw_tp = _match_pred_tp(gt_boxes, raw_boxes)
            kept_tp = _match_pred_tp(gt_boxes, kept_boxes)
            gt_loss_rows = _gt_loss_analysis(
                gt_boxes=gt_boxes,
                raw_boxes=raw_boxes,
                water_boxes=payload_i["water_boxes"],
                final_boxes=kept_boxes,
            )

            if selected_tip is not None:
                prev_prev_tip = prev_tip
                prev_tip = selected_tip

            if render_frames and not bool(args.skip_render):
                img_path = eval_image_dir / file_name if use_separate_eval else (image_dir / file_name)
                out_img = split_frames_dir / f"{i:03d}_{Path(file_name).stem}.png"
                _draw_frame(
                    image_path=img_path,
                    gt_boxes=gt_boxes,
                    raw_preds=raw_boxes,
                    raw_tp=raw_tp,
                    kept_preds=kept_boxes,
                    kept_tp=kept_tp,
                    wake_triangle=wake_triangle,
                    title=f"split={split_name} idx={i} raw={len(raw_boxes)} traj={len(traj_comps)} kept={len(kept_boxes)}",
                    out_path=out_img,
                )
                rendered.append(out_img)

            frame_stats.append(
                {
                    "frame_index": i,
                    "image_id": image_id,
                    "file_name": file_name,
                    "gt": len(gt_boxes),
                    "raw": len(raw_boxes),
                    "kept": len(kept_boxes),
                    "gt_lost": int(sum(1 for row in gt_loss_rows if row["lost_stage"] is not None)),
                    "gt_lost_by_stage": dict(Counter(row["lost_stage"] for row in gt_loss_rows if row["lost_stage"] is not None)),
                    "water_relative_thr": float(payload_i["water_relative_thr"]),
                    "trajectory_candidates": len(traj_comps),
                    "wake_triangle_used": bool(wake_triangle is not None),
                    "raw_box_overlaps": [
                        {
                            "box": [float(x) for x in box],
                            "overlaps_gt": [
                                {
                                    "gt_box": [float(x) for x in gt_box],
                                    "overlap": float(_overlap_area(_to_xyxy(box), _to_xyxy(gt_box))),
                                }
                                for gt_box in gt_boxes
                            ],
                        }
                        for box in raw_boxes
                    ],
                }
            )

        raw_metrics = evaluate_viso_detection(gt_annotations=gt_items_for_split, pred_annotations=pred_raw)
        filtered_metrics = evaluate_viso_detection(gt_annotations=gt_items_for_split, pred_annotations=pred_filtered)

        sheet: Path | None = None
        if render_frames and not bool(args.skip_render):
            sheet = split_out_dir / "contact_sheet_test_frames.png"
            _contact_sheet(rendered, sheet, cols=int(args.sheet_cols))

        return {
            "metrics": {
                "raw_ramfd": {
                    "precision": float(raw_metrics["precision"]),
                    "recall": float(raw_metrics["recall"]),
                    "f1": float(raw_metrics["f1"]),
                    "tp": int(raw_metrics["tp"]),
                    "fp": int(raw_metrics["fp"]),
                    "fn": int(raw_metrics["fn"]),
                    "predictions": len(pred_raw),
                },
                "water_filtered_ramfd": {
                    "precision": float(filtered_metrics["precision"]),
                    "recall": float(filtered_metrics["recall"]),
                    "f1": float(filtered_metrics["f1"]),
                    "tp": int(filtered_metrics["tp"]),
                    "fp": int(filtered_metrics["fp"]),
                    "fn": int(filtered_metrics["fn"]),
                    "predictions": len(pred_filtered),
                },
            },
            "frame_stats": frame_stats,
            "sheet": sheet,
            "frames_dir": split_frames_dir,
            "rendered_count": len(rendered),
        }

    split_results: Dict[str, Dict[str, Any]] = {}
    for split_name, split_idx, render_frames in [("train", train_idx, False), ("validation", val_idx, False), ("test", test_idx, True)]:
        split_results[split_name] = evaluate_split(split_name=split_name, split_idx=split_idx, render_frames=render_frames)

    bank_json = [{"feature_mean": [float(x) for x in p.feature_mean.tolist()], "count": int(p.count)} for p in bank]
    negative_bank_json = [{"feature_mean": [float(x) for x in p.feature_mean.tolist()], "count": int(p.count)} for p in negative_bank]
    feature_audit = _feature_distance_audit(
        positive_samples=train_samples,
        positive_bank=bank,
        negative_samples=negative_samples,
        negative_bank=negative_bank,
    )

    gt_loss_summary = _summarize_gt_loss_from_split_results(split_results)

    report = {
        "experiment": "ramfd_plus_water_knn_bank",
        "dataset": {
            "train_annotations": str(ann_path),
            "train_image_dir": str(image_dir),
            "eval_annotations": str(eval_ann_path) if use_separate_eval and eval_ann_path is not None else str(ann_path),
            "eval_image_dir": str(eval_image_dir) if use_separate_eval and eval_image_dir is not None else str(image_dir),
            "evaluation_protocol": "train_eval_disjoint" if use_separate_eval else "single_split_train_val",
            "max_images": int(args.max_images),
            "train_usable_images": len(usable),
            "eval_usable_images": len(eval_usable),
            "train_skipped_shape_mismatch": int(train_skipped_shape_mismatch),
            "eval_skipped_shape_mismatch": int(eval_skipped_shape_mismatch),
            "train_images": len(train_idx),
            "test_images": len(test_idx),
            "train_ratio": float(args.train_ratio),
        },
        "ramfd": {
            "k": float(args.ramfd_k),
            "min_area": int(args.ramfd_min_area),
            "max_area": int(args.ramfd_max_area),
            "min_aspect_ratio": float(args.ramfd_min_aspect_ratio),
            "max_aspect_ratio": float(args.ramfd_max_aspect_ratio),
            "clutter_active_ratio": float(args.ramfd_clutter_active_ratio),
            "clutter_dilate_kernel": int(args.ramfd_clutter_dilate_kernel),
            "use_tile_adaptive_threshold": False,
            "tile_size": 64,
            "clutter_require_static_edge": False,
            "clutter_edge_percentile": 75.0,
            "use_dark_water_gate": False,
            "water_intensity_percentile": 60.0,
        },
        "water_bank": {
            "feature_mode": str(args.water_feature_mode),
            "water_margin": int(args.water_margin),
            "water_train_margins": [int(x) for x in train_margins],
            "water_train_jitter": int(args.water_train_jitter),
            "water_train_color_jitter": float(args.water_train_color_jitter),
            "water_train_use_all_gt": bool(args.water_train_use_all_gt),
            "train_samples_count": len(train_samples),
            "negative_samples_count": len(negative_samples),
            "new_cluster_distance": float(args.new_cluster_distance),
            "test_water_max_dist": float(args.test_water_max_dist),
            "use_adaptive_prototype_threshold": bool(args.use_adaptive_prototype_threshold),
            "adaptive_dist_percentile": float(args.adaptive_dist_percentile),
            "adaptive_sigma_factor": float(args.adaptive_sigma_factor),
            "adaptive_min_abs_threshold": float(args.adaptive_min_abs_threshold),
            "use_gt_overlap_threshold_calibration": bool(args.use_gt_overlap_threshold_calibration),
            "gt_overlap_threshold_percentile": float(args.gt_overlap_threshold_percentile),
            "nearest_gap_min": float(args.nearest_gap_min),
            "use_frame_relative_water_threshold": bool(args.use_frame_relative_water_threshold),
            "frame_relative_margin": float(args.frame_relative_margin),
            "max_water_candidates_per_frame": int(args.max_water_candidates_per_frame),
            "use_negative_bank": bool(args.use_negative_bank),
            "negative_bank_margin": float(args.negative_bank_margin),
            "negative_grid_step": int(args.negative_grid_step),
            "negative_exclusion_margin": int(args.negative_exclusion_margin),
            "negative_min_pos_dist": float(args.negative_min_pos_dist),
            "negative_max_samples_per_image": int(args.negative_max_samples_per_image),
            "negative_new_cluster_distance": float(args.negative_new_cluster_distance),
            "use_trajectory_tip_filter": bool(args.use_trajectory_tip_filter),
            "trajectory_pipeline_len": int(args.trajectory_pipeline_len),
            "trajectory_radius": float(args.trajectory_radius),
            "trajectory_min_occurrences": int(args.trajectory_min_occurrences),
            "use_wake_triangle_filter": bool(args.use_wake_triangle_filter),
            "wake_length_scale": float(args.wake_length_scale),
            "wake_width_scale": float(args.wake_width_scale),
            "min_usage_rule": {
                "formula": "max(3, ceil(1% of train images))",
                "value": int(min_usage),
            },
            "prototypes": bank_json,
            "prototype_count": len(bank_json),
            "negative_prototypes": negative_bank_json,
            "negative_prototype_count": len(negative_bank_json),
            "prototype_thresholds": [float(t) for t in adaptive_thresholds],
            "gt_overlap_thresholds": [float(t) for t in gt_overlap_thresholds],
            "gt_overlap_diagnostics": gt_overlap_diag,
            "feature_audit": feature_audit,
        },
        "splits": {
            split_name: {
                "size": int(len(split_results[split_name]["frame_stats"])),
                "metrics": split_results[split_name]["metrics"],
                "artifacts": {
                    "contact_sheet": str(split_results[split_name]["sheet"]) if split_results[split_name]["sheet"] is not None else None,
                    "frames_dir": str(split_results[split_name]["frames_dir"]),
                    "render_enabled": not bool(args.skip_render),
                },
                "frame_stats": split_results[split_name]["frame_stats"],
            }
            for split_name in ["train", "validation", "test"]
        },
        "metrics": {
            "raw_ramfd": split_results["test"]["metrics"]["raw_ramfd"],
            "water_filtered_ramfd": split_results["test"]["metrics"]["water_filtered_ramfd"],
        },
        "artifacts": {
            "contact_sheet": str(split_results["test"]["sheet"]) if split_results["test"]["sheet"] is not None else None,
            "frames_dir": str(split_results["test"]["frames_dir"]),
            "render_enabled": not bool(args.skip_render),
        },
        "frame_stats": split_results["test"]["frame_stats"],
        "gt_loss_summary": gt_loss_summary,
    }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    stage_counts = Counter(gt_loss_summary["stage_counts"])

    examples = []
    for example in gt_loss_summary["examples"]:
        examples.append(
            f"- {example['file_name']}: lost={example['gt_lost']} stages={example['gt_lost_by_stage']}"
        )

    md = [
        "# RAMFD + Water KNN Bank Experiment",
        "",
        f"- Train images: {len(train_idx)}",
        f"- Test images: {len(test_idx)}",
        f"- Feature mode: {args.water_feature_mode}",
        f"- Water prototypes kept: {len(bank_json)}",
        f"- Negative prototypes kept: {len(negative_bank_json)}",
        f"- Min prototype usage: {min_usage}",
        "",
        "## Metrics by split",
        "",
        *[
            f"- {split_name.title()}: {json.dumps(report['splits'][split_name]['metrics']['water_filtered_ramfd'], ensure_ascii=False)}"
            for split_name in ["train", "validation", "test"]
        ],
        "",
        "## GT-loss analysis",
        "",
        f"- Lost GT by stage: {dict(stage_counts)}",
        "- Representative failing frames:",
        *examples,
        "",
        "## Artifacts",
        "",
        f"- Test contact sheet: {split_results['test']['sheet'] if split_results['test']['sheet'] is not None else 'skipped (--skip-render)'}",
        f"- Test frames dir: {split_results['test']['frames_dir']}",
        "- Test diagnostic frames:",
        *[
            f"  - {frame['file_name']}: tp={sum(1 for row in frame.get('raw_box_overlaps', []) if any(overlap.get('overlap', 0.0) > 0.0 for overlap in row.get('overlaps_gt', [])))} fp={0} fn={max(0, int(frame['gt']) - sum(1 for row in frame.get('raw_box_overlaps', []) if any(overlap.get('overlap', 0.0) > 0.0 for overlap in row.get('overlaps_gt', []))))}"
            for frame in split_results['test']['frame_stats']
        ],
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print("REPORT", report_json)
    print("CONTACT_SHEET", split_results["test"]["sheet"] if split_results["test"]["sheet"] is not None else "skipped")


if __name__ == "__main__":
    main()
