from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.mmb_complete import Component, _ramfd_components, _read_gray_frame, _trajectory_filter
from src.viso_evaluation import load_coco_annotations
from scripts.ramfd_water_knn_experiment import (
    _build_water_bank,
    _collect_water_examples_around_box,
    _filter_candidates_by_water,
    _nearest_proto,
    _point_in_triangle,
    _select_tip_component,
    _water_rgb_around_bbox,
    _wake_triangle_points,
)


def _to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _ov(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def _load_consistent_frames(image_dir: Path, images: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[np.ndarray]]:
    usable: List[Dict[str, Any]] = []
    gray_frames: List[np.ndarray] = []
    rgb_frames: List[np.ndarray] = []
    ref_shape: Tuple[int, int] | None = None

    for img in images:
        p = image_dir / str(img.get("file_name", ""))
        if not p.exists():
            continue
        g = _read_gray_frame(p)
        r = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        shp = (int(g.shape[0]), int(g.shape[1]))
        if ref_shape is None:
            ref_shape = shp
        elif shp != ref_shape:
            continue
        usable.append(img)
        gray_frames.append(g)
        rgb_frames.append(r)

    return usable, gray_frames, rgb_frames


def _contact_sheet(paths: List[Path], out_path: Path, cols: int = 2) -> None:
    if not paths:
        return
    thumbs = [Image.open(p).convert("RGB") for p in paths]
    w, h = thumbs[0].size
    rows = int(math.ceil(len(thumbs) / float(cols)))
    gap = 8
    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (24, 24, 24))
    for i, im in enumerate(thumbs):
        rr = i // cols
        cc = i % cols
        x = gap + cc * (w + gap)
        y = gap + rr * (h + gap)
        canvas.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _sample_knn_sea_points(
    rgb_image: np.ndarray,
    bank: List[np.ndarray],
    max_dist: float,
    nearest_gap_min: float,
    count: int,
    seed: int,
) -> List[Tuple[int, int]]:
    h, w, _ = rgb_image.shape
    if h <= 0 or w <= 0 or count <= 0 or not bank:
        return []

    rng = np.random.default_rng(seed)
    # Probe a random subset of pixels and keep those accepted by the trained water KNN rule.
    probe_n = min(max(count * 20, 2000), h * w)
    ys = rng.integers(0, h, size=probe_n)
    xs = rng.integers(0, w, size=probe_n)

    accepted: List[Tuple[int, int]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        sample = rgb_image[y, x].astype(np.float32)
        _, d1, d2 = _nearest_proto(sample, bank)
        if not np.isfinite(d1):
            continue
        if float(d1) > float(max_dist):
            continue
        gap = float(d2 - d1) if np.isfinite(d2) else float("inf")
        if gap < float(nearest_gap_min):
            continue
        accepted.append((int(x), int(y)))
        if len(accepted) >= count:
            break

    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Render N test images with GT/TP/FP/FN/TN overlays")
    parser.add_argument("--report-json", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="uniform",
        choices=["uniform", "prediction_order"],
        help="uniform: evenly spaced frames; prediction_order: first frames with prediction in temporal order",
    )
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--only-wrong",
        action="store_true",
        help="Render only frames with FP or FN outcomes",
    )
    parser.add_argument(
        "--max-wrong-frames",
        type=int,
        default=120,
        help="Maximum number of wrong frames to render when --only-wrong is set",
    )
    parser.add_argument(
        "--draw-sea-samples",
        action="store_true",
        help="Overlay sampled points that the trained water KNN would label as sea",
    )
    parser.add_argument("--sea-samples", type=int, default=100, help="Number of KNN sea points to draw per frame")
    parser.add_argument("--sea-seed", type=int, default=1337, help="Base RNG seed for sea point sampling")
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    root = Path(__file__).resolve().parents[1]
    ds = report["dataset"]
    ramfd = report["ramfd"]
    water = report["water_bank"]

    train_ann = root / str(ds["train_annotations"])
    train_dir = root / str(ds["train_image_dir"])
    eval_ann = root / str(ds["eval_annotations"])
    eval_dir = root / str(ds["eval_image_dir"])

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (report_path.parent / "test10_visual_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Train water bank
    train_payload = load_coco_annotations(train_ann)
    train_images = sorted(train_payload["images"], key=lambda x: str(x.get("file_name", "")))
    train_anns_all = train_payload["annotations"]
    train_cat = int(train_anns_all[0].get("category_id", 0)) if train_anns_all else 0
    train_anns: Dict[int, List[Dict[str, Any]]] = {}
    for a in train_anns_all:
        if int(a.get("category_id", -1)) != train_cat:
            continue
        train_anns.setdefault(int(a["image_id"]), []).append(a)

    train_usable, _, train_rgb = _load_consistent_frames(train_dir, train_images)

    margins = [int(x) for x in water.get("water_train_margins", [10, 16, 22, 30])]
    jitter = int(water.get("water_train_jitter", 8))
    use_all_gt = bool(water.get("water_train_use_all_gt", True))

    train_samples: List[np.ndarray] = []
    for i, img in enumerate(train_usable):
        gt_items = train_anns.get(int(img["id"]), [])
        if not gt_items:
            continue
        chosen = gt_items if use_all_gt else [gt_items[0]]
        for gt in chosen:
            bbox = list(map(float, gt["bbox"]))
            train_samples.extend(
                _collect_water_examples_around_box(
                    rgb_image=train_rgb[i],
                    bbox=bbox,
                    margins=margins,
                    jitter_px=jitter,
                    feature_mode="rgb_mean",
                    color_jitter=float(water.get("water_train_color_jitter", 0.0)),
                )
            )

    bank = _build_water_bank(
        train_samples=train_samples,
        new_cluster_distance=float(water.get("new_cluster_distance", 22.0)),
        min_usage=max(3, int(math.ceil(0.01 * max(1, len(train_usable))))),
    )

    # Eval data
    payload = load_coco_annotations(eval_ann)
    images = sorted(payload["images"], key=lambda x: str(x.get("file_name", "")))
    anns_all = payload["annotations"]
    eval_cat = int(anns_all[0].get("category_id", train_cat)) if anns_all else train_cat
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns_all:
        if int(a.get("category_id", -1)) != eval_cat:
            continue
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    usable, gray_frames, rgb_frames = _load_consistent_frames(eval_dir, images)
    if not usable:
        raise RuntimeError("No usable eval frames with consistent shape")

    n = len(usable)

    per_frame = _ramfd_components(
        gray_frames,
        k=float(ramfd.get("k", 2.2)),
        min_area=int(ramfd.get("min_area", 1)),
        max_area=int(ramfd.get("max_area", 1500)),
        min_aspect_ratio=float(ramfd.get("min_aspect_ratio", 0.2)),
        max_aspect_ratio=float(ramfd.get("max_aspect_ratio", 15.0)),
        clutter_active_ratio=float(ramfd.get("clutter_active_ratio", 0.995)),
        clutter_dilate_kernel=int(ramfd.get("clutter_dilate_kernel", 1)),
        use_tile_adaptive_threshold=False,
        tile_size=64,
        clutter_require_static_edge=False,
        clutter_edge_percentile=75.0,
        use_dark_water_gate=False,
        water_intensity_percentile=60.0,
    )

    water_kept: Dict[int, List[List[float]]] = {}
    for i in range(n):
        comps = per_frame.get(i, [])
        candidates = []
        for c in comps:
            box = [float(c.x), float(c.y), float(c.w), float(c.h)]
            sample = _water_rgb_around_bbox(rgb_frames[i], bbox=box, margin=int(water.get("water_margin", 18)))
            if sample is None:
                continue
            bi, d1, d2 = _nearest_proto(sample, bank)
            if bi < 0:
                continue
            candidates.append(
                {
                    "box": box,
                    "best_idx": int(bi),
                    "dist": float(d1),
                    "second_dist": float(d2),
                    "local_thr": float(water.get("test_water_max_dist", 100.0)),
                }
            )

        kept, _ = _filter_candidates_by_water(
            candidates=candidates,
            nearest_gap_min=float(water.get("nearest_gap_min", 0.0)),
            use_frame_relative_threshold=bool(water.get("use_frame_relative_water_threshold", True)),
            frame_relative_margin=float(water.get("frame_relative_margin", 26.0)),
            max_candidates_per_frame=int(water.get("max_water_candidates_per_frame", 0)),
        )
        water_kept[i] = kept

    use_traj = bool(water.get("use_trajectory_tip_filter", True))
    by_frame: Dict[int, List[Component]] = {i: [] for i in range(n)}
    if use_traj:
        for i in range(n):
            frame_comps = []
            for b in water_kept.get(i, []):
                x, y, w, h = b
                xi, yi = int(round(x)), int(round(y))
                wi, hi = max(1, int(round(w))), max(1, int(round(h)))
                frame_comps.append(
                    Component(
                        frame_index=i,
                        x=xi,
                        y=yi,
                        w=wi,
                        h=hi,
                        area=int(wi * hi),
                        cx=float(xi + wi / 2.0),
                        cy=float(yi + hi / 2.0),
                    )
                )
            by_frame[i] = frame_comps

        by_frame = _trajectory_filter(
            components_by_frame=by_frame,
            pipeline_len=int(water.get("trajectory_pipeline_len", 5)),
            radius=float(water.get("trajectory_radius", 16.0)),
            min_occurrences=int(water.get("trajectory_min_occurrences", 2)),
            recover_partial_tracks=True,
        )

    pred_by_frame: Dict[int, List[List[float]]] = {i: [] for i in range(n)}
    wake_by_frame: Dict[int, List[Tuple[float, float]] | None] = {i: None for i in range(n)}

    prev_tip: Component | None = None
    prev_prev_tip: Component | None = None
    for i in range(n):
        if use_traj:
            traj_comps = list(by_frame.get(i, []))
            tip, motion = _select_tip_component(traj_comps, prev_tip=prev_tip, prev_prev_tip=prev_prev_tip)
            wake_tri = None
            if tip is not None and motion is not None and bool(water.get("use_wake_triangle_filter", True)):
                ref_size = max(1.0, 0.5 * (float(tip.w) + float(tip.h)))
                wake_tri = _wake_triangle_points(
                    tip_cx=float(tip.cx),
                    tip_cy=float(tip.cy),
                    dir_x=float(motion[0]),
                    dir_y=float(motion[1]),
                    wake_length=float(water.get("wake_length_scale", 3.0)) * ref_size,
                    wake_half_width=float(water.get("wake_width_scale", 1.4)) * ref_size,
                )
                if len(traj_comps) > 1:
                    filtered: List[Component] = []
                    for c in traj_comps:
                        if c is tip:
                            filtered.append(c)
                            continue
                        if _point_in_triangle(float(c.cx), float(c.cy), wake_tri):
                            continue
                        filtered.append(c)
                    traj_comps = filtered

            if tip is None and traj_comps:
                tip = max(traj_comps, key=lambda c: float(c.area))

            if tip is not None:
                pred_by_frame[i] = [[float(tip.x), float(tip.y), float(tip.w), float(tip.h)]]
                prev_prev_tip = prev_tip
                prev_tip = tip
            wake_by_frame[i] = wake_tri
        else:
            if water_kept.get(i):
                pred_by_frame[i] = [max(water_kept[i], key=lambda b: float(b[2] * b[3]))]

    k = min(max(1, int(args.num_images)), n)
    if args.selection_mode == "prediction_order":
        pred_idxs = [i for i in range(n) if len(pred_by_frame.get(i, [])) > 0]
        idxs = pred_idxs[:k]
        if len(idxs) < k:
            idx_set = set(idxs)
            remaining = [i for i in range(n) if i not in idx_set]
            idxs.extend(remaining[: (k - len(idxs))])
    else:
        idxs = np.linspace(0, n - 1, num=k, dtype=int).tolist()
        idxs = sorted(list(dict.fromkeys(idxs)))

    C_GT = (255, 220, 0)
    C_TP = (40, 220, 80)
    C_FP = (255, 60, 60)
    C_FN = (255, 0, 255)
    C_TN = (0, 220, 255)
    C_WAKE = (180, 80, 255)
    C_SEA = (120, 250, 255)

    saved: List[Path] = []
    summary_rows: List[Dict[str, Any]] = []

    for i in idxs:
        info = usable[i]
        file_name = str(info["file_name"])
        image_id = int(info["id"])
        gt_boxes = [list(map(float, g["bbox"])) for g in anns_by_img.get(image_id, [])]
        pred_boxes = pred_by_frame.get(i, [])

        gt_xy = [_to_xyxy(b) for b in gt_boxes]
        pr_xy = [_to_xyxy(b) for b in pred_boxes]
        gt_used = [False] * len(gt_xy)
        pr_tp = [False] * len(pr_xy)

        for pi, pb in enumerate(pr_xy):
            best = -1
            best_ov = 0.0
            for gi, gb in enumerate(gt_xy):
                if gt_used[gi]:
                    continue
                ov = _ov(pb, gb)
                if ov > best_ov:
                    best_ov = ov
                    best = gi
            if best >= 0 and best_ov > 0.0:
                pr_tp[pi] = True
                gt_used[best] = True

        tp_idx = [pi for pi, ok in enumerate(pr_tp) if ok]
        fp_idx = [pi for pi, ok in enumerate(pr_tp) if not ok]
        fn_idx = [gi for gi, used in enumerate(gt_used) if not used]
        frame_tn = int(len(gt_boxes) == 0 and len(pred_boxes) == 0)

        if args.only_wrong and len(fp_idx) == 0 and len(fn_idx) == 0:
            continue

        image = Image.open(eval_dir / file_name).convert("RGB")
        draw = ImageDraw.Draw(image)

        for b in gt_boxes:
            x, y, w, h = b
            draw.rectangle((x, y, x + w, y + h), outline=C_GT, width=2)

        for pi, b in enumerate(pred_boxes):
            x, y, w, h = b
            if pi in tp_idx:
                draw.rectangle((x, y, x + w, y + h), outline=C_TP, width=3)
                draw.text((max(0, int(x)), max(0, int(y) - 14)), "TP", fill=C_TP)
            else:
                draw.rectangle((x, y, x + w, y + h), outline=C_FP, width=3)
                draw.text((max(0, int(x)), max(0, int(y) - 14)), "FP", fill=C_FP)

        for gi in fn_idx:
            x, y, w, h = gt_boxes[gi]
            for t in range(2):
                draw.rectangle((x - t, y - t, x + w + t, y + h + t), outline=C_FN, width=2)
            draw.text((max(0, int(x)), max(0, int(y) - 14)), "FN", fill=C_FN)

        tri = wake_by_frame.get(i)
        if tri is not None and len(tri) == 3:
            draw.polygon(tri, outline=C_WAKE)

        sea_points: List[Tuple[int, int]] = []
        if args.draw_sea_samples:
            sea_points = _sample_knn_sea_points(
                rgb_image=rgb_frames[i],
                bank=bank,
                max_dist=float(water.get("test_water_max_dist", 100.0)),
                nearest_gap_min=float(water.get("nearest_gap_min", 0.0)),
                count=max(0, int(args.sea_samples)),
                seed=int(args.sea_seed) + int(i),
            )
            for x, y in sea_points:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=C_SEA)

        if frame_tn:
            draw.rectangle((1, 1, image.width - 2, image.height - 2), outline=C_TN, width=3)

        draw.rectangle((0, 0, min(image.width - 1, 1150), 56), fill=(0, 0, 0))
        draw.text(
            (6, 6),
            f"idx={i} file={file_name} GT={len(gt_boxes)} TP={len(tp_idx)} FP={len(fp_idx)} FN={len(fn_idx)} TN={frame_tn}",
            fill=(255, 255, 255),
        )
        draw.text(
            (6, 28),
            "GT yellow | TP green | FP red | FN magenta | TN cyan frame | Wake violet triangle | Sea samples light-cyan",
            fill=(230, 230, 230),
        )

        out_img = frames_dir / f"{i:03d}_{Path(file_name).stem}.png"
        image.save(out_img)
        saved.append(out_img)
        summary_rows.append(
            {
                "frame_index": i,
                "file_name": file_name,
                "gt": len(gt_boxes),
                "tp": len(tp_idx),
                "fp": len(fp_idx),
                "fn": len(fn_idx),
                "tn": frame_tn,
                "sea_points_drawn": len(sea_points),
            }
        )

        if args.only_wrong and len(saved) >= max(1, int(args.max_wrong_frames)):
            break

    sheet = out_dir / "contact_sheet_test10.png"
    _contact_sheet(saved, sheet, cols=2)

    summary_path = out_dir / "summary_test10.json"
    summary_path.write_text(
        json.dumps({"source_report": str(report_path), "count": len(saved), "contact_sheet": str(sheet), "frames": summary_rows}, indent=2),
        encoding="utf-8",
    )

    print("OUT_DIR", out_dir)
    print("FRAMES", len(saved))
    print("SHEET", sheet)
    print("SUMMARY", summary_path)


if __name__ == "__main__":
    main()
