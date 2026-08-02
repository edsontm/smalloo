from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from scripts.ramfd_water_knn_experiment import (
    _build_water_bank,
    _collect_water_examples_around_box,
    _filter_candidates_by_water,
    _water_rgb_around_bbox,
    _nearest_proto,
)
from src.mmb_complete import _ramfd_components, _read_gray_frame
from src.viso_evaluation import load_coco_annotations


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


def _match_pred_gt(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> Tuple[List[bool], List[bool]]:
    gt_xyxy = [_to_xyxy(b) for b in gt_boxes]
    pred_xyxy = [_to_xyxy(b) for b in pred_boxes]

    gt_used = [False] * len(gt_xyxy)
    pred_tp = [False] * len(pred_xyxy)
    gt_matched = [False] * len(gt_xyxy)

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
            gt_matched[best] = True
            pred_tp[pi] = True

    return pred_tp, gt_matched


def _draw_fp_fn_frame(
    image_path: Path,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_tp: List[bool],
    gt_matched: List[bool],
    draw_fp: bool,
    draw_fn: bool,
    title: str,
    out_path: Path,
) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Always draw GT reference in yellow.
    for gt in gt_boxes:
        x, y, w, h = gt
        draw.rectangle((x, y, x + w, y + h), outline=(255, 220, 0), width=2)

    marked_count = 0
    if draw_fp:
        for idx, (bbox, is_tp) in enumerate(zip(pred_boxes, pred_tp), start=1):
            if is_tp:
                continue
            x, y, w, h = bbox
            marked_count += 1
            for t in range(4):
                draw.rectangle((x - t, y - t, x + w + t, y + h + t), outline=(255, 40, 40), width=2)
            tx = max(0, int(x))
            ty = max(0, int(y) - 16)
            draw.rectangle((tx, ty, min(image.width - 1, tx + 44), ty + 14), fill=(255, 40, 40))
            draw.text((tx + 4, ty + 2), f"FP{idx}", fill=(255, 255, 255))

    if draw_fn:
        for idx, (bbox, is_match) in enumerate(zip(gt_boxes, gt_matched), start=1):
            if is_match:
                continue
            x, y, w, h = bbox
            marked_count += 1
            for t in range(4):
                draw.rectangle((x - t, y - t, x + w + t, y + h + t), outline=(255, 0, 255), width=2)
            tx = max(0, int(x))
            ty = max(0, int(y) - 16)
            draw.rectangle((tx, ty, min(image.width - 1, tx + 44), ty + 14), fill=(255, 0, 255))
            draw.text((tx + 4, ty + 2), f"FN{idx}", fill=(255, 255, 255))

    draw.rectangle((0, 0, min(image.width - 1, 980), 24), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return marked_count


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


def _load_config(report_path: Path) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    dataset = payload.get("dataset", {})
    water = payload.get("water_bank", {})
    ramfd = payload.get("ramfd", {})

    return {
        "ann_path": Path(str(dataset.get("annotations", ""))),
        "image_dir": Path(str(dataset.get("image_dir", ""))),
        "max_images": int(dataset.get("max_images", 120)),
        "train_ratio": float(dataset.get("train_ratio", 0.7)),
        "water_margin": int(water.get("water_margin", 18)),
        "water_train_margins": [int(x) for x in water.get("water_train_margins", [12, 18, 26])],
        "water_train_jitter": int(water.get("water_train_jitter", 6)),
        "water_train_use_all_gt": bool(water.get("water_train_use_all_gt", False)),
        "new_cluster_distance": float(water.get("new_cluster_distance", 22.0)),
        "test_water_max_dist": float(water.get("test_water_max_dist", 18.0)),
        "nearest_gap_min": float(water.get("nearest_gap_min", 0.0)),
        "use_frame_relative_water_threshold": bool(water.get("use_frame_relative_water_threshold", False)),
        "frame_relative_margin": float(water.get("frame_relative_margin", 26.0)),
        "max_water_candidates_per_frame": int(water.get("max_water_candidates_per_frame", 0)),
        "expected_fp": int(payload.get("metrics", {}).get("water_filtered_ramfd", {}).get("fp", -1)),
        "expected_fn": int(payload.get("metrics", {}).get("water_filtered_ramfd", {}).get("fn", -1)),
        "ramfd_k": float(ramfd.get("k", 4.0)),
        "ramfd_min_area": int(ramfd.get("min_area", 5)),
        "ramfd_max_area": int(ramfd.get("max_area", 220)),
        "ramfd_min_aspect_ratio": float(ramfd.get("min_aspect_ratio", 0.4)),
        "ramfd_max_aspect_ratio": float(ramfd.get("max_aspect_ratio", 10.0)),
        "ramfd_clutter_active_ratio": float(ramfd.get("clutter_active_ratio", 0.95)),
        "ramfd_clutter_dilate_kernel": int(ramfd.get("clutter_dilate_kernel", 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FP/FN highlights from RAMFD+water-KNN report")
    parser.add_argument("--report-json", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["fp", "fn", "both"], default="both")
    parser.add_argument("--out-tag", type=str, default="")
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--ramfd-k", type=float, default=None)
    parser.add_argument("--ramfd-min-area", type=int, default=None)
    parser.add_argument("--ramfd-max-area", type=int, default=None)
    parser.add_argument("--ramfd-min-aspect-ratio", type=float, default=None)
    parser.add_argument("--ramfd-max-aspect-ratio", type=float, default=None)
    parser.add_argument("--ramfd-clutter-active-ratio", type=float, default=None)
    parser.add_argument("--ramfd-clutter-dilate-kernel", type=int, default=None)
    parser.add_argument("--test-water-max-dist", type=float, default=None)
    parser.add_argument("--nearest-gap-min", type=float, default=None)
    parser.add_argument("--frame-relative-margin", type=float, default=None)
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    cfg = _load_config(report_path)

    if args.ramfd_k is not None:
        cfg["ramfd_k"] = float(args.ramfd_k)
    if args.ramfd_min_area is not None:
        cfg["ramfd_min_area"] = int(args.ramfd_min_area)
    if args.ramfd_max_area is not None:
        cfg["ramfd_max_area"] = int(args.ramfd_max_area)
    if args.ramfd_min_aspect_ratio is not None:
        cfg["ramfd_min_aspect_ratio"] = float(args.ramfd_min_aspect_ratio)
    if args.ramfd_max_aspect_ratio is not None:
        cfg["ramfd_max_aspect_ratio"] = float(args.ramfd_max_aspect_ratio)
    if args.ramfd_clutter_active_ratio is not None:
        cfg["ramfd_clutter_active_ratio"] = float(args.ramfd_clutter_active_ratio)
    if args.ramfd_clutter_dilate_kernel is not None:
        cfg["ramfd_clutter_dilate_kernel"] = int(args.ramfd_clutter_dilate_kernel)
    if args.test_water_max_dist is not None:
        cfg["test_water_max_dist"] = float(args.test_water_max_dist)
    if args.nearest_gap_min is not None:
        cfg["nearest_gap_min"] = float(args.nearest_gap_min)
    if args.frame_relative_margin is not None:
        cfg["frame_relative_margin"] = float(args.frame_relative_margin)

    ann_path = cfg["ann_path"]
    image_dir = cfg["image_dir"]
    max_images = int(cfg["max_images"])
    train_ratio = float(cfg["train_ratio"])

    run_tag = args.out_tag.strip() if args.out_tag.strip() else report_path.parent.name
    out_dir = report_path.parent.parent / f"{run_tag}_fp_fn_highlight"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_coco_annotations(ann_path)
    images = sorted(payload["images"], key=lambda x: str(x.get("file_name", "")))[:max_images]
    selected = {int(i["id"]) for i in images}
    anns = [a for a in payload["annotations"] if int(a.get("image_id", -1)) in selected]
    category_id = int(anns[0].get("category_id", 0)) if anns else 0

    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        if int(a.get("category_id", -1)) != category_id:
            continue
        anns_by_image.setdefault(int(a["image_id"]), []).append(a)

    usable: List[Dict[str, Any]] = []
    gray_frames: List[np.ndarray] = []
    rgb_frames: List[np.ndarray] = []
    for img in images:
        p = image_dir / str(img.get("file_name", ""))
        if not p.exists():
            continue
        usable.append(img)
        gray_frames.append(_read_gray_frame(p))
        rgb_frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))

    n = len(usable)
    if n < 8:
        raise RuntimeError("Not enough usable images")

    train_n = int(max(4, min(n - 4, round(n * train_ratio))))
    train_idx = list(range(train_n))
    test_idx = list(range(train_n, n))

    train_samples: List[np.ndarray] = []
    for i in train_idx:
        image_id = int(usable[i]["id"])
        gt_boxes = anns_by_image.get(image_id, [])
        if not gt_boxes:
            continue
        chosen = gt_boxes if bool(cfg["water_train_use_all_gt"]) else [gt_boxes[0]]
        for gt in chosen:
            bbox = list(map(float, gt["bbox"]))
            samples = _collect_water_examples_around_box(
                rgb_image=rgb_frames[i],
                bbox=bbox,
                margins=[int(x) for x in cfg["water_train_margins"]],
                jitter_px=int(cfg["water_train_jitter"]),
            )
            train_samples.extend(samples)

    min_usage = max(3, int(math.ceil(0.01 * len(train_idx))))
    bank = _build_water_bank(
        train_samples=train_samples,
        new_cluster_distance=float(cfg["new_cluster_distance"]),
        min_usage=min_usage,
    )

    per_frame = _ramfd_components(
        gray_frames,
        k=float(cfg["ramfd_k"]),
        min_area=int(cfg["ramfd_min_area"]),
        max_area=int(cfg["ramfd_max_area"]),
        min_aspect_ratio=float(cfg["ramfd_min_aspect_ratio"]),
        max_aspect_ratio=float(cfg["ramfd_max_aspect_ratio"]),
        clutter_active_ratio=float(cfg["ramfd_clutter_active_ratio"]),
        clutter_dilate_kernel=int(cfg["ramfd_clutter_dilate_kernel"]),
        use_tile_adaptive_threshold=False,
        tile_size=64,
        clutter_require_static_edge=False,
        clutter_edge_percentile=75.0,
        use_dark_water_gate=False,
        water_intensity_percentile=60.0,
    )

    want_fp = args.mode in ("fp", "both")
    want_fn = args.mode in ("fn", "both")

    fp_paths: List[Path] = []
    fn_paths: List[Path] = []
    rows: List[Dict[str, Any]] = []
    total_fp = 0
    total_fn = 0

    for i in test_idx:
        image_id = int(usable[i]["id"])
        file_name = str(usable[i]["file_name"])
        gt_items = anns_by_image.get(image_id, [])
        gt_boxes = [list(map(float, g["bbox"])) for g in gt_items]

        comps = per_frame.get(i, [])
        water_candidates: List[Dict[str, Any]] = []
        for c in comps:
            box = [float(c.x), float(c.y), float(c.w), float(c.h)]
            water_sample = _water_rgb_around_bbox(rgb_frames[i], bbox=box, margin=int(cfg["water_margin"]))
            if water_sample is None:
                continue
            best_idx, dist, second_dist = _nearest_proto(water_sample, bank)
            if best_idx < 0:
                continue
            water_candidates.append(
                {
                    "box": box,
                    "best_idx": int(best_idx),
                    "dist": float(dist),
                    "second_dist": float(second_dist),
                    "local_thr": float(cfg["test_water_max_dist"]),
                }
            )

        kept_boxes, _ = _filter_candidates_by_water(
            candidates=water_candidates,
            nearest_gap_min=float(cfg["nearest_gap_min"]),
            use_frame_relative_threshold=bool(cfg["use_frame_relative_water_threshold"]),
            frame_relative_margin=float(cfg["frame_relative_margin"]),
            max_candidates_per_frame=int(cfg["max_water_candidates_per_frame"]),
        )

        pred_tp, gt_matched = _match_pred_gt(gt_boxes, kept_boxes)
        fp_count = int(sum(1 for is_tp in pred_tp if not is_tp))
        fn_count = int(sum(1 for is_match in gt_matched if not is_match))
        total_fp += fp_count
        total_fn += fn_count

        frame_row = {
            "frame_index": i,
            "file_name": file_name,
            "gt": len(gt_boxes),
            "pred_kept": len(kept_boxes),
            "fp": fp_count,
            "fn": fn_count,
        }
        rows.append(frame_row)

        img_path = image_dir / file_name
        if want_fp and fp_count > 0:
            out_path_fp = out_dir / "fp_frames" / f"{i:03d}_{Path(file_name).stem}.png"
            _draw_fp_fn_frame(
                image_path=img_path,
                gt_boxes=gt_boxes,
                pred_boxes=kept_boxes,
                pred_tp=pred_tp,
                gt_matched=gt_matched,
                draw_fp=True,
                draw_fn=False,
                title=f"idx={i} FP={fp_count} FN={fn_count} file={file_name}",
                out_path=out_path_fp,
            )
            fp_paths.append(out_path_fp)

        if want_fn and fn_count > 0:
            out_path_fn = out_dir / "fn_frames" / f"{i:03d}_{Path(file_name).stem}.png"
            _draw_fp_fn_frame(
                image_path=img_path,
                gt_boxes=gt_boxes,
                pred_boxes=kept_boxes,
                pred_tp=pred_tp,
                gt_matched=gt_matched,
                draw_fp=False,
                draw_fn=True,
                title=f"idx={i} FP={fp_count} FN={fn_count} file={file_name}",
                out_path=out_path_fn,
            )
            fn_paths.append(out_path_fn)

    fp_sheet = None
    fn_sheet = None
    if want_fp and fp_paths:
        fp_sheet = out_dir / "contact_sheet_fp_only.png"
        _contact_sheet(fp_paths, fp_sheet, cols=int(args.sheet_cols))
    if want_fn and fn_paths:
        fn_sheet = out_dir / "contact_sheet_fn_only.png"
        _contact_sheet(fn_paths, fn_sheet, cols=int(args.sheet_cols))

    summary = {
        "source_report": str(report_path),
        "mode": str(args.mode),
        "expected_from_report": {
            "fp": int(cfg["expected_fp"]),
            "fn": int(cfg["expected_fn"]),
        },
        "recomputed": {
            "fp": int(total_fp),
            "fn": int(total_fn),
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "fp_frames_dir": str(out_dir / "fp_frames"),
            "fn_frames_dir": str(out_dir / "fn_frames"),
            "contact_sheet_fp_only": str(fp_sheet) if fp_sheet is not None else None,
            "contact_sheet_fn_only": str(fn_sheet) if fn_sheet is not None else None,
        },
        "frame_stats": rows,
    }

    summary_path = out_dir / "fp_fn_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OUT_DIR", out_dir)
    print("SUMMARY", summary_path)
    print("TOTAL_FP", total_fp)
    print("TOTAL_FN", total_fn)
    print("FP_SHEET", fp_sheet)
    print("FN_SHEET", fn_sheet)


if __name__ == "__main__":
    main()
