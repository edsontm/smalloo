from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.mmb_complete import Component, _amfd_components, _ramfd_components, _read_gray_frame, _trajectory_filter
from src.viso_evaluation import evaluate_viso_detection, load_coco_annotations


@dataclass(frozen=True)
class FrameStats:
    tp: int
    fp: int
    fn: int


def to_xyxy(bbox_xywh: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x + w), float(y + h)


def overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def match_overlap(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> Tuple[FrameStats, List[bool]]:
    gt_xyxy = [to_xyxy(b) for b in gt_boxes]
    pred_xyxy = [to_xyxy(b) for b in pred_boxes]

    matched_gt = [False] * len(gt_xyxy)
    matched_pred = [False] * len(pred_xyxy)

    for p_idx, p in enumerate(pred_xyxy):
        best_i = -1
        best_area = 0.0
        for g_idx, g in enumerate(gt_xyxy):
            if matched_gt[g_idx]:
                continue
            area = overlap(p, g)
            if area > best_area:
                best_area = area
                best_i = g_idx
        if best_i >= 0 and best_area > 0.0:
            matched_gt[best_i] = True
            matched_pred[p_idx] = True

    tp = int(sum(matched_pred))
    fp = int(len(pred_xyxy) - tp)
    fn = int(len(gt_xyxy) - sum(matched_gt))
    return FrameStats(tp=tp, fp=fp, fn=fn), matched_pred


def draw_frame(
    image_path: Path,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_is_tp: List[bool],
    frame_stats: FrameStats,
    title: str,
    out_path: Path,
) -> None:
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    for gt in gt_boxes:
        x, y, w, h = gt
        draw.rectangle((x, y, x + w, y + h), outline=(255, 215, 0), width=2)

    for pred, is_tp in zip(pred_boxes, pred_is_tp):
        x, y, w, h = pred
        color = (32, 220, 80) if is_tp else (235, 64, 52)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)

    draw.rectangle((0, 0, min(im.width, 980), 24), fill=(0, 0, 0))
    draw.text((6, 5), f"{title} | TP={frame_stats.tp} FP={frame_stats.fp} FN={frame_stats.fn}", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def make_contact_sheet(image_paths: List[Path], out_path: Path, cols: int = 5) -> None:
    if not image_paths:
        return
    thumbs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = thumbs[0].size
    rows = int(np.ceil(len(thumbs) / max(1, cols)))
    gap = 8

    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (24, 24, 24))
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = gap + c * (w + gap)
        y = gap + r * (h + gap)
        sheet.paste(thumb, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def detect_variant(variant: str, frames: List[np.ndarray], params: Dict[str, Any]) -> Dict[int, List[Any]]:
    if variant == "amfd_baseline":
        return _amfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
        )
    if variant == "ramfd_v1":
        return _ramfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=False,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )
    if variant == "ramfd_v2":
        return _ramfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=True,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )
    if variant == "ramfd_v3":
        return _ramfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=True,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )
    if variant == "ramfd_v4":
        return _ramfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=False,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )
    if variant == "ramfd_v5":
        base = _ramfd_components(
            frames,
            k=float(params["k"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=False,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )
        return _trajectory_filter(
            base,
            pipeline_len=int(params["pf_len"]),
            radius=float(params["pf_radius"]),
            min_occurrences=int(params["pf_min_occ"]),
            recover_partial_tracks=bool(params["pf_recover"]),
        )
    if variant == "ramfd_v6":
        # Branch A: conservative detector with very low-FP operating point.
        conservative = _ramfd_components(
            frames,
            k=float(params["k_conservative"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=False,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )

        # Branch B: recovery detector (more sensitive), later constrained by motion context.
        recovery = _amfd_components(
            frames,
            k=float(params["k_recovery"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
        )

        fused: Dict[int, List[Component]] = {idx: [] for idx in range(len(frames))}

        anchor_history: List[Tuple[float, float]] = []
        size_history: List[Tuple[int, int]] = []
        radius = float(params["recovery_radius"])

        for t in range(len(frames)):
            cons = conservative.get(t, [])
            if cons:
                fused[t] = list(cons)
                for c in cons:
                    anchor_history.append((c.cx, c.cy))
                    size_history.append((c.w, c.h))
                if len(anchor_history) > 8:
                    anchor_history = anchor_history[-8:]
                if len(size_history) > 8:
                    size_history = size_history[-8:]
                continue

            rec = recovery.get(t, [])
            if not rec or not anchor_history:
                continue

            if len(anchor_history) >= 2:
                vx = anchor_history[-1][0] - anchor_history[-2][0]
                vy = anchor_history[-1][1] - anchor_history[-2][1]
            else:
                vx, vy = 0.0, 0.0

            px = anchor_history[-1][0] + vx
            py = anchor_history[-1][1] + vy

            if size_history:
                med_w = int(round(float(np.median([w for w, _ in size_history]))))
                med_h = int(round(float(np.median([h for _, h in size_history]))))
            else:
                med_w, med_h = 0, 0

            best = None
            best_cost = float("inf")
            for cand in rec:
                d = float(np.hypot(cand.cx - px, cand.cy - py))
                if d > radius:
                    continue
                if med_w > 0 and med_h > 0:
                    size_cost = abs(cand.w - med_w) + abs(cand.h - med_h)
                else:
                    size_cost = 0.0
                cost = d + 0.25 * float(size_cost)
                if cost < best_cost:
                    best_cost = cost
                    best = cand

            if best is not None:
                fused[t] = [best]
                anchor_history.append((best.cx, best.cy))
                size_history.append((best.w, best.h))
                if len(anchor_history) > 8:
                    anchor_history = anchor_history[-8:]
                if len(size_history) > 8:
                    size_history = size_history[-8:]

        # Final decision: keep only temporally consistent trajectories.
        return _trajectory_filter(
            fused,
            pipeline_len=int(params["pf_len"]),
            radius=float(params["pf_radius"]),
            min_occurrences=int(params["pf_min_occ"]),
            recover_partial_tracks=bool(params["pf_recover"]),
        )
    if variant == "ramfd_v7":
        # Branch A: conservative detector with low FP.
        conservative = _ramfd_components(
            frames,
            k=float(params["k_conservative"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
            clutter_active_ratio=float(params["clutter_ratio"]),
            clutter_dilate_kernel=int(params["clutter_kernel"]),
            use_tile_adaptive_threshold=False,
            tile_size=int(params["tile_size"]),
            clutter_require_static_edge=bool(params["edge_gate"]),
            clutter_edge_percentile=float(params["edge_percentile"]),
            use_dark_water_gate=bool(params["water_gate"]),
            water_intensity_percentile=float(params["water_percentile"]),
        )

        # Branch B: sensitive recovery branch.
        recovery = _amfd_components(
            frames,
            k=float(params["k_recovery"]),
            min_area=int(params["min_area"]),
            max_area=int(params["max_area"]),
            min_aspect_ratio=float(params["min_ar"]),
            max_aspect_ratio=float(params["max_ar"]),
        )

        fused: Dict[int, List[Component]] = {idx: [] for idx in range(len(frames))}

        anchor_history: List[Tuple[float, float]] = []
        size_history: List[Tuple[int, int]] = []
        confidence = 0

        recovery_radius = float(params["recovery_radius"])
        max_recovery_per_frame = int(params["max_recovery_per_frame"])
        min_conf_for_recovery = int(params["min_conf_for_recovery"])
        conf_gain_conservative = int(params["conf_gain_conservative"])
        conf_gain_recovery = int(params["conf_gain_recovery"])
        conf_decay = int(params["conf_decay"])

        for t in range(len(frames)):
            cons = conservative.get(t, [])
            if cons:
                fused[t] = list(cons)
                confidence += conf_gain_conservative
                for c in cons:
                    anchor_history.append((c.cx, c.cy))
                    size_history.append((c.w, c.h))
                if len(anchor_history) > 10:
                    anchor_history = anchor_history[-10:]
                if len(size_history) > 10:
                    size_history = size_history[-10:]
                continue

            confidence = max(0, confidence - conf_decay)
            rec = recovery.get(t, [])
            if not rec or not anchor_history:
                continue

            # Require accumulated temporal confidence before activating recovery branch.
            if confidence < min_conf_for_recovery:
                continue

            if len(anchor_history) >= 2:
                vx = anchor_history[-1][0] - anchor_history[-2][0]
                vy = anchor_history[-1][1] - anchor_history[-2][1]
            else:
                vx, vy = 0.0, 0.0

            px = anchor_history[-1][0] + vx
            py = anchor_history[-1][1] + vy

            if size_history:
                med_w = int(round(float(np.median([w for w, _ in size_history]))))
                med_h = int(round(float(np.median([h for _, h in size_history]))))
            else:
                med_w, med_h = 0, 0

            scored: List[Tuple[float, Component]] = []
            for cand in rec:
                d = float(np.hypot(cand.cx - px, cand.cy - py))
                if d > recovery_radius:
                    continue
                if med_w > 0 and med_h > 0:
                    size_cost = abs(cand.w - med_w) + abs(cand.h - med_h)
                else:
                    size_cost = 0.0
                cost = d + (0.20 * float(size_cost))
                scored.append((cost, cand))

            if not scored:
                continue

            scored.sort(key=lambda x: x[0])
            accepted = [c for _, c in scored[: max(1, max_recovery_per_frame)]]
            fused[t] = accepted
            confidence += conf_gain_recovery

            # Use best accepted candidate as anchor update.
            best = accepted[0]
            anchor_history.append((best.cx, best.cy))
            size_history.append((best.w, best.h))
            if len(anchor_history) > 10:
                anchor_history = anchor_history[-10:]
            if len(size_history) > 10:
                size_history = size_history[-10:]

        return _trajectory_filter(
            fused,
            pipeline_len=int(params["pf_len"]),
            radius=float(params["pf_radius"]),
            min_occurrences=int(params["pf_min_occ"]),
            recover_partial_tracks=bool(params["pf_recover"]),
        )
    raise ValueError(variant)


def build_iteration_plan() -> List[Dict[str, Any]]:
    return [
        {
            "iteration": 1,
            "variant": "amfd_baseline",
            "algorithm_change": "Baseline AMFD (referencia)",
            "reason": "Estabelecer baseline visual e quantitativo.",
            "params": {
                "k": 4.0,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.35,
                "clutter_kernel": 3,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": False,
                "water_percentile": 55.0,
            },
        },
        {
            "iteration": 2,
            "variant": "ramfd_v1",
            "algorithm_change": "R-AMFD v1: supressao de clutter persistente no tempo",
            "reason": "FPs recorrentes aparecem sempre nas mesmas estruturas estaticas do porto.",
            "params": {
                "k": 4.0,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.35,
                "clutter_kernel": 3,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": False,
                "water_percentile": 55.0,
            },
        },
        {
            "iteration": 3,
            "variant": "ramfd_v2",
            "algorithm_change": "R-AMFD v2: v1 + limiar adaptativo por blocos",
            "reason": "Reduzir sensibilidade a hot-spots locais mantendo resposta no alvo movel.",
            "params": {
                "k": 3.8,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.30,
                "clutter_kernel": 3,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": False,
                "water_percentile": 55.0,
            },
        },
        {
            "iteration": 4,
            "variant": "ramfd_v3",
            "algorithm_change": "R-AMFD v3: v2 + supressao persistente limitada a borda estatica",
            "reason": "v1/v2 removeram tambem o alvo; limitar a supressao para estruturas estaticas de alto gradiente.",
            "params": {
                "k": 3.8,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.30,
                "clutter_kernel": 3,
                "tile_size": 64,
                "edge_gate": True,
                "edge_percentile": 72.0,
                "water_gate": False,
                "water_percentile": 55.0,
            },
        },
        {
            "iteration": 5,
            "variant": "ramfd_v4",
            "algorithm_change": "R-AMFD v4: contexto de agua escura para suprimir clutter em terra",
            "reason": "FP dominante ocorre em estruturas brilhantes do porto; alvo fica em agua escura.",
            "params": {
                "k": 4.0,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.95,
                "clutter_kernel": 1,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": True,
                "water_percentile": 60.0,
            },
        },
        {
            "iteration": 6,
            "variant": "ramfd_v5",
            "algorithm_change": "R-AMFD v5: v4 + recuperacao temporal de trilha curta",
            "reason": "v4 reduziu quase todo ruído, mas recall caiu; recuperar detecções faltantes por consistencia temporal.",
            "params": {
                "k": 4.0,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.95,
                "clutter_kernel": 1,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": True,
                "water_percentile": 60.0,
                "pf_len": 5,
                "pf_radius": 8.0,
                "pf_min_occ": 2,
                "pf_recover": True,
            },
        },
        {
            "iteration": 7,
            "variant": "ramfd_v6",
            "algorithm_change": "R-AMFD v6: fusao de dois ramos + consistencia temporal",
            "reason": "Combinar um ramo conservador (baixo FP) com recuperacao guiada pelo ultimo alvo e filtrar por trilha consistente.",
            "params": {
                "k_conservative": 4.0,
                "k_recovery": 3.2,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.95,
                "clutter_kernel": 1,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": True,
                "water_percentile": 60.0,
                "recovery_radius": 18.0,
                "pf_len": 5,
                "pf_radius": 9.0,
                "pf_min_occ": 2,
                "pf_recover": True,
            },
        },
        {
            "iteration": 8,
            "variant": "ramfd_v7",
            "algorithm_change": "R-AMFD v7: janela maior + ate 2 candidatos + confianca temporal acumulada",
            "reason": "Aumentar recall com recuperacao controlada por contexto temporal e reduzir falsos resgates isolados.",
            "params": {
                "k_conservative": 4.0,
                "k_recovery": 3.0,
                "min_area": 5,
                "max_area": 220,
                "min_ar": 0.4,
                "max_ar": 10.0,
                "clutter_ratio": 0.95,
                "clutter_kernel": 1,
                "tile_size": 64,
                "edge_gate": False,
                "edge_percentile": 75.0,
                "water_gate": True,
                "water_percentile": 60.0,
                "recovery_radius": 26.0,
                "max_recovery_per_frame": 2,
                "min_conf_for_recovery": 2,
                "conf_gain_conservative": 2,
                "conf_gain_recovery": 1,
                "conf_decay": 1,
                "pf_len": 5,
                "pf_radius": 10.0,
                "pf_min_occ": 2,
                "pf_recover": True,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental algorithm test: AMFD -> R-AMFD")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--tag", type=str, default="ship20_ramfd")
    parser.add_argument("--sheet-cols", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ann_path = root / "VISO" / "coco" / "ship" / "Annotations" / "instances_train2017.json"
    image_dir = root / "VISO" / "coco" / "ship" / "train2017"
    out_root = root / "research" / "experiments" / "v9_mmb_ship_bayesian_validation_tuning" / "artifacts" / "debug" / f"{args.tag}_incremental"
    out_root.mkdir(parents=True, exist_ok=True)

    payload = load_coco_annotations(ann_path)
    images = sorted(payload["images"], key=lambda x: str(x.get("file_name", "")))[: int(args.max_images)]
    selected_ids = {int(i["id"]) for i in images}
    annotations = [a for a in payload["annotations"] if int(a.get("image_id", -1)) in selected_ids]

    category_id = int(annotations[0].get("category_id", 0)) if annotations else 0

    frames: List[np.ndarray] = []
    usable_images: List[Dict[str, Any]] = []
    for img in images:
        p = image_dir / str(img.get("file_name", ""))
        if p.exists():
            frames.append(_read_gray_frame(p))
            usable_images.append(img)

    gt_by_image: Dict[int, List[List[float]]] = {}
    for ann in annotations:
        if int(ann.get("category_id", -1)) != category_id:
            continue
        image_id = int(ann["image_id"])
        gt_by_image.setdefault(image_id, []).append(list(map(float, ann["bbox"])))

    plan = build_iteration_plan()
    report: Dict[str, Any] = {
        "tag": args.tag,
        "max_images": int(args.max_images),
        "images_loaded": len(usable_images),
        "iterations": [],
    }

    for cfg in plan:
        iteration = int(cfg["iteration"])
        variant = str(cfg["variant"])
        params = dict(cfg["params"])

        per_frame = detect_variant(variant=variant, frames=frames, params=params)

        predictions: List[Dict[str, Any]] = []
        counts: List[int] = []
        for idx, comps in per_frame.items():
            counts.append(len(comps))
            if idx < 0 or idx >= len(usable_images):
                continue
            image_id = int(usable_images[idx]["id"])
            for c in comps:
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(c.x), float(c.y), float(c.w), float(c.h)],
                        "score": 0.95,
                    }
                )

        metrics = evaluate_viso_detection(gt_annotations=annotations, pred_annotations=predictions)

        iter_dir = out_root / f"iter_{iteration:02d}_{variant}"
        frames_dir = iter_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_rows: List[Dict[str, Any]] = []
        render_paths: List[Path] = []

        for idx, img in enumerate(usable_images):
            image_id = int(img["id"])
            file_name = str(img["file_name"])
            gt_boxes = gt_by_image.get(image_id, [])
            comps = per_frame.get(idx, [])
            pred_boxes = [[float(c.x), float(c.y), float(c.w), float(c.h)] for c in comps]
            fs, pred_tp = match_overlap(gt_boxes=gt_boxes, pred_boxes=pred_boxes)

            out_img = frames_dir / f"{idx:03d}_{Path(file_name).stem}.png"
            draw_frame(
                image_path=image_dir / file_name,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_is_tp=pred_tp,
                frame_stats=fs,
                title=f"Iter{iteration} {variant} frame={idx:03d}",
                out_path=out_img,
            )
            render_paths.append(out_img)
            frame_rows.append(
                {
                    "frame_index": idx,
                    "image_id": image_id,
                    "file_name": file_name,
                    "tp": fs.tp,
                    "fp": fs.fp,
                    "fn": fs.fn,
                    "pred_count": len(pred_boxes),
                }
            )

        contact_sheet = iter_dir / "contact_sheet.png"
        make_contact_sheet(render_paths, contact_sheet, cols=int(args.sheet_cols))

        worst_fp = sorted(frame_rows, key=lambda r: (r["fp"], -r["tp"]), reverse=True)[:5]
        iter_report = {
            "iteration": iteration,
            "variant": variant,
            "algorithm_change": cfg["algorithm_change"],
            "reason": cfg["reason"],
            "params": params,
            "metrics": {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "predictions": int(len(predictions)),
            },
            "amfd_stats": {
                "components_total": int(sum(counts)),
                "components_mean_per_frame": float(np.mean(counts) if counts else 0.0),
                "components_p95_per_frame": float(np.percentile(counts, 95) if counts else 0.0),
            },
            "contact_sheet": str(contact_sheet),
            "frames_dir": str(frames_dir),
            "worst_fp_frames": worst_fp,
            "frame_stats": frame_rows,
        }
        report["iterations"].append(iter_report)
        (iter_dir / "summary.json").write_text(json.dumps(iter_report, indent=2), encoding="utf-8")

    report_path = out_root / "summary_all_iterations.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("OUTPUT_DIR", out_root)
    print("REPORT", report_path)


if __name__ == "__main__":
    main()
