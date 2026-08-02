from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.mmb_complete import Component, _ramfd_components, _read_gray_frame, _trajectory_filter
from src.viso_evaluation import evaluate_viso_detection, load_coco_annotations


@dataclass
class WaterPrototype:
    rgb_mean: np.ndarray
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


def _water_rgb_around_bbox(rgb_image: np.ndarray, bbox: List[float], margin: int) -> np.ndarray | None:
    ih, iw, _ = rgb_image.shape
    mask = _ring_mask(iw, ih, bbox, margin=margin)
    pixels = rgb_image[mask > 0]
    if pixels.size == 0:
        return None
    return np.asarray(pixels.mean(axis=0), dtype=np.float32)


def _parse_int_csv(value: str) -> List[int]:
    items: List[int] = []
    for token in str(value).split(","):
        t = token.strip()
        if not t:
            continue
        items.append(int(t))
    return items


def _collect_water_examples_around_box(
    rgb_image: np.ndarray,
    bbox: List[float],
    margins: List[int],
    jitter_px: int,
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
            sample = _water_rgb_around_bbox(
                rgb_image,
                bbox=[float(x + dx), float(y + dy), float(w), float(h)],
                margin=m,
            )
            if sample is not None:
                examples.append(sample)
    return examples


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _build_water_bank(
    train_samples: List[np.ndarray],
    new_cluster_distance: float,
    min_usage: int,
) -> List[WaterPrototype]:
    bank: List[WaterPrototype] = []

    for sample in train_samples:
        if not bank:
            bank.append(WaterPrototype(rgb_mean=sample.copy(), count=1))
            continue

        dists = [_euclidean(sample, p.rgb_mean) for p in bank]
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist > new_cluster_distance:
            bank.append(WaterPrototype(rgb_mean=sample.copy(), count=1))
            continue

        proto = bank[best_idx]
        new_count = proto.count + 1
        proto.rgb_mean = ((proto.rgb_mean * proto.count) + sample) / float(new_count)
        proto.count = new_count

    bank = [p for p in bank if p.count >= min_usage]
    return bank


def _nearest_water_dist(sample: np.ndarray, bank: List[WaterPrototype]) -> float:
    if not bank:
        return float("inf")
    return min(_euclidean(sample, p.rgb_mean) for p in bank)


def _nearest_proto(sample: np.ndarray, bank: List[WaterPrototype]) -> Tuple[int, float, float]:
    if not bank:
        return -1, float("inf"), float("inf")
    dists = [_euclidean(sample, p.rgb_mean) for p in bank]
    order = np.argsort(np.asarray(dists, dtype=np.float32))
    best_idx = int(order[0])
    best_dist = float(dists[best_idx])
    second_dist = float(dists[int(order[1])]) if len(order) > 1 else float("inf")
    return best_idx, best_dist, second_dist


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


def _filter_candidates_by_water(
    candidates: List[Dict[str, Any]],
    nearest_gap_min: float,
    use_frame_relative_threshold: bool,
    frame_relative_margin: float,
    max_candidates_per_frame: int,
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
        if dist <= local_thr and gap >= gap_min:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RAMFD + water-color KNN-like bank filtering experiment")
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--annotations-path", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--eval-annotations-path", type=str, default="")
    parser.add_argument("--eval-image-dir", type=str, default="")
    parser.add_argument("--tag", type=str, default="ship_water_knn")
    parser.add_argument("--water-margin", type=int, default=18)
    parser.add_argument("--water-train-margins", type=str, default="12,18,26")
    parser.add_argument("--water-train-jitter", type=int, default=6)
    parser.add_argument("--water-train-use-all-gt", action="store_true")
    parser.add_argument("--new-cluster-distance", type=float, default=22.0)
    parser.add_argument("--test-water-max-dist", type=float, default=18.0)
    parser.add_argument("--use-adaptive-prototype-threshold", action="store_true")
    parser.add_argument("--adaptive-dist-percentile", type=float, default=85.0)
    parser.add_argument("--adaptive-sigma-factor", type=float, default=1.3)
    parser.add_argument("--adaptive-min-abs-threshold", type=float, default=8.0)
    parser.add_argument("--nearest-gap-min", type=float, default=0.0)
    parser.add_argument("--use-frame-relative-water-threshold", action="store_true")
    parser.add_argument("--frame-relative-margin", type=float, default=26.0)
    parser.add_argument("--max-water-candidates-per-frame", type=int, default=0)
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
    out_dir = root / "research" / "experiments" / "v9_mmb_ship_bayesian_validation_tuning" / "artifacts" / "debug" / f"{args.tag}_{args.max_images}frames"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        train_n = int(max(4, min(n - 4, round(n * float(args.train_ratio)))))
        train_idx = list(range(train_n))
        test_idx = list(range(train_n, n))
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
            )
            train_samples.extend(samples)

    min_usage = max(3, int(math.ceil(0.01 * max(1, len(train_idx)))))
    bank = _build_water_bank(
        train_samples=train_samples,
        new_cluster_distance=float(args.new_cluster_distance),
        min_usage=min_usage,
    )

    adaptive_thresholds = _adaptive_thresholds_by_proto(
        train_samples=train_samples,
        bank=bank,
        percentile=float(args.adaptive_dist_percentile),
        sigma_factor=float(args.adaptive_sigma_factor),
        min_abs_threshold=float(args.adaptive_min_abs_threshold),
    )

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

    gt_test: List[Dict[str, Any]] = []
    pred_raw: List[Dict[str, Any]] = []
    pred_filtered: List[Dict[str, Any]] = []

    rendered: List[Path] = []
    frame_stats: List[Dict[str, Any]] = []
    frame_payload: Dict[int, Dict[str, Any]] = {}
    water_kept_by_frame: Dict[int, List[List[float]]] = {idx: [] for idx in test_idx}

    for i in test_idx:
        image_id = int(eval_usable[i]["id"])
        file_name = str(eval_usable[i]["file_name"])
        gt_items = eval_anns_by_image.get(image_id, [])
        gt_boxes = [list(map(float, g["bbox"])) for g in gt_items]
        gt_test.extend(gt_items)

        comps = per_frame.get(i, [])
        raw_boxes: List[List[float]] = []
        water_candidates: List[Dict[str, Any]] = []

        for c in comps:
            box = [float(c.x), float(c.y), float(c.w), float(c.h)]
            raw_boxes.append(box)
            pred_raw.append({"image_id": image_id, "category_id": eval_category_id, "bbox": box, "score": 0.95})

            water_sample = _water_rgb_around_bbox(eval_rgb_frames[i], bbox=box, margin=int(args.water_margin))
            if water_sample is None:
                continue

            best_idx, dist, second_dist = _nearest_proto(water_sample, bank)
            if best_idx < 0:
                continue

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
                    "local_thr": float(local_thr),
                }
            )

        kept_boxes, water_diag = _filter_candidates_by_water(
            candidates=water_candidates,
            nearest_gap_min=float(args.nearest_gap_min),
            use_frame_relative_threshold=bool(args.use_frame_relative_water_threshold),
            frame_relative_margin=float(args.frame_relative_margin),
            max_candidates_per_frame=int(args.max_water_candidates_per_frame),
        )
        water_kept_by_frame[i] = kept_boxes
        frame_payload[i] = {
            "image_id": image_id,
            "file_name": file_name,
            "gt_boxes": gt_boxes,
            "raw_boxes": raw_boxes,
            "water_relative_thr": water_diag["relative_thr"],
        }

    if bool(args.use_trajectory_tip_filter):
        comps_by_frame: Dict[int, List[Component]] = {idx: [] for idx in test_idx}
        for idx in test_idx:
            comps_by_frame[idx] = [_component_from_box(frame_index=idx, box=b) for b in water_kept_by_frame.get(idx, [])]

        comps_by_frame = _trajectory_filter(
            components_by_frame=comps_by_frame,
            pipeline_len=int(args.trajectory_pipeline_len),
            radius=float(args.trajectory_radius),
            min_occurrences=int(args.trajectory_min_occurrences),
            recover_partial_tracks=True,
        )
    else:
        comps_by_frame = {idx: [_component_from_box(frame_index=idx, box=b) for b in water_kept_by_frame.get(idx, [])] for idx in test_idx}

    prev_tip: Component | None = None
    prev_prev_tip: Component | None = None

    for i in test_idx:
        payload_i = frame_payload[i]
        image_id = int(payload_i["image_id"])
        file_name = str(payload_i["file_name"])
        gt_boxes = payload_i["gt_boxes"]
        raw_boxes = payload_i["raw_boxes"]

        traj_comps = list(comps_by_frame.get(i, []))
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

        if selected_tip is not None:
            prev_prev_tip = prev_tip
            prev_tip = selected_tip

        if not bool(args.skip_render):
            img_path = eval_image_dir / file_name if use_separate_eval else (image_dir / file_name)
            out_img = out_dir / "frames" / f"{i:03d}_{Path(file_name).stem}.png"
            _draw_frame(
                image_path=img_path,
                gt_boxes=gt_boxes,
                raw_preds=raw_boxes,
                raw_tp=raw_tp,
                kept_preds=kept_boxes,
                kept_tp=kept_tp,
                wake_triangle=wake_triangle,
                title=f"idx={i} raw={len(raw_boxes)} traj={len(traj_comps)} kept={len(kept_boxes)}",
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
                "water_relative_thr": float(payload_i["water_relative_thr"]),
                "trajectory_candidates": len(traj_comps),
                "wake_triangle_used": bool(wake_triangle is not None),
            }
        )

    raw_metrics = evaluate_viso_detection(gt_annotations=gt_test, pred_annotations=pred_raw)
    filtered_metrics = evaluate_viso_detection(gt_annotations=gt_test, pred_annotations=pred_filtered)

    sheet: Path | None = None
    if not bool(args.skip_render):
        sheet = out_dir / "contact_sheet_test_frames.png"
        _contact_sheet(rendered, sheet, cols=int(args.sheet_cols))

    bank_json = [
        {"rgb_mean": [float(x) for x in p.rgb_mean.tolist()], "count": int(p.count)}
        for p in bank
    ]

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
            "water_margin": int(args.water_margin),
            "water_train_margins": [int(x) for x in train_margins],
            "water_train_jitter": int(args.water_train_jitter),
            "water_train_use_all_gt": bool(args.water_train_use_all_gt),
            "train_samples_count": len(train_samples),
            "new_cluster_distance": float(args.new_cluster_distance),
            "test_water_max_dist": float(args.test_water_max_dist),
            "use_adaptive_prototype_threshold": bool(args.use_adaptive_prototype_threshold),
            "adaptive_dist_percentile": float(args.adaptive_dist_percentile),
            "adaptive_sigma_factor": float(args.adaptive_sigma_factor),
            "adaptive_min_abs_threshold": float(args.adaptive_min_abs_threshold),
            "nearest_gap_min": float(args.nearest_gap_min),
            "use_frame_relative_water_threshold": bool(args.use_frame_relative_water_threshold),
            "frame_relative_margin": float(args.frame_relative_margin),
            "max_water_candidates_per_frame": int(args.max_water_candidates_per_frame),
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
            "prototype_thresholds": [float(t) for t in adaptive_thresholds],
        },
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
        "artifacts": {
            "contact_sheet": str(sheet) if sheet is not None else None,
            "frames_dir": str(out_dir / "frames"),
            "render_enabled": not bool(args.skip_render),
        },
        "frame_stats": frame_stats,
    }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# RAMFD + Water KNN Bank Experiment",
        "",
        f"- Train images: {len(train_idx)}",
        f"- Test images: {len(test_idx)}",
        f"- Water prototypes kept: {len(bank_json)}",
        f"- Min prototype usage: {min_usage}",
        "",
        "## Metrics",
        "",
        f"- Raw RAMFD: {report['metrics']['raw_ramfd']}",
        f"- Water-filtered RAMFD: {report['metrics']['water_filtered_ramfd']}",
        "",
        "## Artifacts",
        "",
        f"- Contact sheet: {sheet if sheet is not None else 'skipped (--skip-render)'}",
        f"- Frames dir: {out_dir / 'frames'}",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print("REPORT", report_json)
    print("CONTACT_SHEET", sheet)


if __name__ == "__main__":
    main()
