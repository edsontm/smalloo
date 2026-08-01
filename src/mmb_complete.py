from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.mmb.robust_matrix_completion import RobustMatrixCompletion, RobustMatrixCompletionConfig


@dataclass
class Component:
    frame_index: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


def _read_gray_frame(image_path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as image:
        gray = image.convert('L')
        return np.asarray(gray, dtype=np.float32)


def _dilate(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    radius = kernel_size // 2
    h, w = binary.shape
    padded = np.pad(binary, radius, mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for y in range(h):
        ys = y
        for x in range(w):
            xs = x
            window = padded[ys : ys + kernel_size, xs : xs + kernel_size]
            out[y, x] = 1 if np.any(window > 0) else 0
    return out


def _erode(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    radius = kernel_size // 2
    h, w = binary.shape
    padded = np.pad(binary, radius, mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for y in range(h):
        ys = y
        for x in range(w):
            xs = x
            window = padded[ys : ys + kernel_size, xs : xs + kernel_size]
            out[y, x] = 1 if np.all(window > 0) else 0
    return out


def _morphology(binary: np.ndarray) -> np.ndarray:
    # Opening then closing to suppress speckle noise and fill tiny holes.
    opened = _dilate(_erode(binary, kernel_size=3), kernel_size=3)
    closed = _erode(_dilate(opened, kernel_size=3), kernel_size=3)
    return closed


def _connected_components(binary: np.ndarray, frame_index: int) -> List[Component]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    components: List[Component] = []

    for y0 in range(h):
        for x0 in range(w):
            if binary[y0, x0] == 0 or visited[y0, x0] != 0:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = 1
            pixels: List[Tuple[int, int]] = []

            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if binary[ny, nx] == 0 or visited[ny, nx] != 0:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            area = len(pixels)
            components.append(
                Component(
                    frame_index=frame_index,
                    x=int(min_x),
                    y=int(min_y),
                    w=int(bw),
                    h=int(bh),
                    area=int(area),
                    cx=float(sum(xs) / area),
                    cy=float(sum(ys) / area),
                )
            )

    return components


def _filter_components(
    components: Iterable[Component],
    min_area: int = 5,
    max_area: int = 80,
    min_aspect_ratio: float = 1.0,
    max_aspect_ratio: float = 6.0,
) -> List[Component]:
    kept: List[Component] = []
    for comp in components:
        if comp.area < min_area or comp.area > max_area:
            continue
        aspect_ratio = float(comp.w) / float(max(comp.h, 1))
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        kept.append(comp)
    return kept


def _components_to_mask(shape: Tuple[int, int], components: Iterable[Component]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for comp in components:
        x1 = max(0, comp.x)
        y1 = max(0, comp.y)
        x2 = min(w, comp.x + comp.w)
        y2 = min(h, comp.y + comp.h)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 1
    return mask


def _masked_mean_std(values: np.ndarray, mask: np.ndarray | None) -> Tuple[float, float]:
    if mask is None:
        return float(values.mean()), float(values.std())
    selected = values[mask > 0]
    if selected.size == 0:
        return float(values.mean()), float(values.std())
    return float(selected.mean()), float(selected.std())


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> Tuple[float, float, float]:
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(mov)
    cps = f_ref * np.conj(f_mov)
    cps /= np.maximum(np.abs(cps), 1e-8)
    corr = np.fft.ifft2(cps)
    corr_abs = np.abs(corr)
    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    dy = float(peak[0])
    dx = float(peak[1])
    h, w = ref.shape
    if dy > h // 2:
        dy -= float(h)
    if dx > w // 2:
        dx -= float(w)
    return dx, dy, float(corr_abs[peak])


def _affine_warp_with_mask(
    frame: np.ndarray,
    dx: float,
    dy: float,
    angle_deg: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    h, w = frame.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    inv_s = 1.0 / max(float(scale), 1e-6)

    a = cos_t * inv_s
    b = sin_t * inv_s
    c = cx - (a * (cx + dx)) - (b * (cy + dy))
    d = -sin_t * inv_s
    e = cos_t * inv_s
    f = cy - (d * (cx + dx)) - (e * (cy + dy))

    img = Image.fromarray(np.asarray(np.clip(frame, 0, 255), dtype=np.float32), mode='F')
    warped_img = img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR, fillcolor=0)
    warped = np.asarray(warped_img, dtype=np.float32)

    mask_img = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode='L')
    warped_mask_img = mask_img.transform(
        (w, h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.NEAREST,
        fillcolor=0,
    )
    warped_mask = (np.asarray(warped_mask_img, dtype=np.uint8) > 0).astype(np.uint8)
    return warped, warped_mask


def _normalize_radiometry(
    frame: np.ndarray,
    ref_mean: float,
    ref_std: float,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    mean_v, std_v = _masked_mean_std(frame, valid_mask)
    std_v = max(std_v, 1e-6)
    normalized = ((frame - mean_v) / std_v) * max(ref_std, 1e-6) + ref_mean
    return np.asarray(np.clip(normalized, 0.0, 255.0), dtype=np.float32)


def _stabilize_frames(
    frames: List[np.ndarray],
    max_step_shift: float = 20.0,
    use_affine: bool = False,
    affine_max_angle_deg: float = 1.0,
    affine_max_scale_delta: float = 0.01,
    radiometric_normalize: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not frames:
        return [], []

    stabilized: List[np.ndarray] = [frames[0]]
    masks: List[np.ndarray] = [np.ones(frames[0].shape, dtype=np.uint8)]
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    cumulative_angle = 0.0
    cumulative_scale = 1.0
    ref_mean, ref_std = _masked_mean_std(frames[0], masks[0])

    for idx in range(1, len(frames)):
        current = frames[idx]
        previous_stable = stabilized[idx - 1]

        best_dx = 0.0
        best_dy = 0.0
        best_angle_rel = 0.0
        best_scale_rel = 1.0

        if use_affine:
            angle_candidates = [-affine_max_angle_deg, 0.0, affine_max_angle_deg]
            scale_candidates = [1.0 - affine_max_scale_delta, 1.0, 1.0 + affine_max_scale_delta]
            best_peak = -1.0
            for angle_rel in angle_candidates:
                for scale_rel in scale_candidates:
                    candidate, _ = _affine_warp_with_mask(current, dx=0.0, dy=0.0, angle_deg=angle_rel, scale=scale_rel)
                    dx_rel, dy_rel, peak = _phase_correlation_shift(previous_stable, candidate)
                    if abs(dx_rel) > max_step_shift or abs(dy_rel) > max_step_shift:
                        continue
                    if peak > best_peak:
                        best_peak = peak
                        best_dx = dx_rel
                        best_dy = dy_rel
                        best_angle_rel = angle_rel
                        best_scale_rel = scale_rel
        else:
            dx_rel, dy_rel, _ = _phase_correlation_shift(frames[idx - 1], current)
            if abs(dx_rel) <= max_step_shift and abs(dy_rel) <= max_step_shift:
                best_dx = dx_rel
                best_dy = dy_rel

        cumulative_dx += best_dx
        cumulative_dy += best_dy
        if use_affine:
            cumulative_angle += best_angle_rel
            cumulative_scale *= best_scale_rel

        shifted, valid_mask = _affine_warp_with_mask(
            current,
            dx=cumulative_dx,
            dy=cumulative_dy,
            angle_deg=cumulative_angle,
            scale=cumulative_scale,
        )
        if radiometric_normalize:
            shifted = _normalize_radiometry(shifted, ref_mean, ref_std, valid_mask)
        stabilized.append(shifted.astype(np.float32))
        masks.append(valid_mask)
    return stabilized, masks


def _amfd_components(
    frames: List[np.ndarray],
    k: float = 4.0,
    min_area: int = 5,
    max_area: int = 80,
    min_aspect_ratio: float = 1.0,
    max_aspect_ratio: float = 6.0,
    valid_masks: List[np.ndarray] | None = None,
) -> Dict[int, List[Component]]:
    per_frame: Dict[int, List[Component]] = {idx: [] for idx in range(len(frames))}
    if len(frames) < 3:
        return per_frame

    for t in range(1, len(frames) - 1):
        i_prev = frames[t - 1]
        i_curr = frames[t]
        i_next = frames[t + 1]

        dt1 = np.abs(i_curr - i_prev)
        dt2 = np.abs(i_next - i_prev)
        dt3 = np.abs(i_next - i_curr)
        response = (dt1 + dt2 + dt3) / 3.0

        valid = None
        if valid_masks is not None and len(valid_masks) > t + 1:
            valid = (valid_masks[t - 1] & valid_masks[t] & valid_masks[t + 1]).astype(np.uint8)

        mean_v, std_v = _masked_mean_std(response, valid)
        threshold = float(mean_v + (k * std_v))
        binary = (response >= threshold).astype(np.uint8)
        if valid is not None:
            binary = (binary * valid).astype(np.uint8)
        binary = _morphology(binary)

        comps = _filter_components(
            _connected_components(binary, frame_index=t),
            min_area=min_area,
            max_area=max_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        per_frame[t] = comps
    return per_frame


def _lrmc_components(
    frames: List[np.ndarray],
    amfd_components: Dict[int, List[Component]],
    l: int = 4,
    k: float = 2.5,
    min_area: int = 5,
    max_area: int = 80,
    min_aspect_ratio: float = 1.0,
    max_aspect_ratio: float = 6.0,
    valid_masks: List[np.ndarray] | None = None,
    frame_frequency: float = 10.0,
) -> Dict[int, List[Component]]:
    per_frame: Dict[int, List[Component]] = {idx: [] for idx in range(len(frames))}
    if len(frames) < 2:
        return per_frame

    completion = RobustMatrixCompletion(
        RobustMatrixCompletionConfig(
            lambda_value=None,
            max_iter=60,
            tol=1e-5,
        )
    )

    # Paper operational setting: N = M / (L * f), where L=4 and f=10 by default.
    # This implies processing frames in contiguous observation groups of size (L*f).
    group_len = max(2, int(round(float(l) * max(float(frame_frequency), 1.0))))
    total_frames = len(frames)
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(total_frames, start_idx + group_len)
        window = frames[start_idx:end_idx]
        if len(window) < 2:
            start_idx = end_idx
            continue

        decomposition = completion.decompose(window)
        for local_idx, sparse_frame in enumerate(decomposition.sparse_frames):
            t = start_idx + local_idx
            fg = np.abs(sparse_frame).astype(np.float32)

            valid = None
            if valid_masks is not None and len(valid_masks) > t:
                valid = valid_masks[t]

            mean_v, std_v = _masked_mean_std(fg, valid)
            threshold = float(mean_v + (k * std_v))
            binary = (fg >= threshold).astype(np.uint8)
            if valid is not None:
                binary = (binary * valid).astype(np.uint8)

            # Use AMFD ROI to suppress LRMC false alarms when background changes.
            roi_mask = _components_to_mask(binary.shape, amfd_components.get(t, []))
            if int(roi_mask.sum()) > 0:
                binary = (binary * roi_mask).astype(np.uint8)

            binary = _morphology(binary)

            comps = _filter_components(
                _connected_components(binary, frame_index=t),
                min_area=min_area,
                max_area=max_area,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
            )
            per_frame[t] = comps

        start_idx = end_idx

    return per_frame


def _merge_components(amfd: Dict[int, List[Component]], lrmc: Dict[int, List[Component]]) -> Dict[int, List[Component]]:
    merged: Dict[int, List[Component]] = {}
    frame_indices = sorted(set(amfd.keys()) | set(lrmc.keys()))
    for frame_index in frame_indices:
        merged[frame_index] = list(amfd.get(frame_index, [])) + list(lrmc.get(frame_index, []))
    return merged


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    max_cost = float(cost.max() if cost.size else 0.0)
    padded = np.full((n, n), max_cost + 1e6, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: List[Tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_rows and j <= n_cols:
            assignments.append((i - 1, j - 1))
    return assignments


def _trajectory_filter(
    components_by_frame: Dict[int, List[Component]],
    pipeline_len: int = 5,
    radius: float = 7.0,
    min_occurrences: int = 3,
    recover_partial_tracks: bool = True,
) -> Dict[int, List[Component]]:
    confirmed: Dict[int, List[Component]] = {idx: [] for idx in components_by_frame}
    frame_indices = sorted(components_by_frame.keys())

    for frame_index in frame_indices:
        seeds = components_by_frame.get(frame_index, [])
        if not seeds:
            continue

        tracks: List[List[Component]] = [[seed] for seed in seeds]
        track_points: List[Dict[int, Component]] = [{frame_index: seed} for seed in seeds]
        occurrences = [1 for _ in seeds]
        active_positions = [(seed.cx, seed.cy) for seed in seeds]

        for delta in range(1, pipeline_len + 1):
            next_index = frame_index + delta
            candidates = components_by_frame.get(next_index, [])
            if not candidates:
                continue

            cost = np.full((len(tracks), len(candidates)), fill_value=1e9, dtype=np.float64)
            for track_idx, (ax, ay) in enumerate(active_positions):
                for cand_idx, cand in enumerate(candidates):
                    dx = abs(ax - cand.cx)
                    dy = abs(ay - cand.cy)
                    if dx < radius and dy < radius:
                        cost[track_idx, cand_idx] = float(np.hypot(dx, dy))

            for track_idx, cand_idx in _hungarian(cost):
                if cost[track_idx, cand_idx] >= 1e8:
                    continue
                cand = candidates[cand_idx]
                tracks[track_idx].append(cand)
                track_points[track_idx][next_index] = cand
                occurrences[track_idx] += 1
                active_positions[track_idx] = (cand.cx, cand.cy)

        for track_idx, (track, occ) in enumerate(zip(tracks, occurrences)):
            if occ >= min_occurrences:
                for item in track:
                    confirmed[item.frame_index].append(item)

                # Paper behavior: if h == 3 or 4 in a 5-frame pipeline, recover missing detections.
                if recover_partial_tracks and pipeline_len == 5 and occ in (3, 4):
                    points = track_points[track_idx]
                    all_frames = [frame_index + d for d in range(0, pipeline_len + 1)]
                    present = sorted(points.keys())
                    if len(present) >= 2:
                        ref_w = int(round(np.median([c.w for c in track])))
                        ref_h = int(round(np.median([c.h for c in track])))
                        ref_area = int(round(np.median([c.area for c in track])))
                        for missing_frame in all_frames:
                            if missing_frame in points:
                                continue
                            prev_candidates = [f for f in present if f < missing_frame]
                            next_candidates = [f for f in present if f > missing_frame]
                            if not prev_candidates or not next_candidates:
                                continue
                            f0 = prev_candidates[-1]
                            f1 = next_candidates[0]
                            c0 = points[f0]
                            c1 = points[f1]
                            if f1 == f0:
                                continue
                            gap = f1 - f0
                            if gap > 3:
                                continue
                            anchor_distance = float(np.hypot(c1.cx - c0.cx, c1.cy - c0.cy))
                            if anchor_distance > (radius * 2.0):
                                continue
                            ratio = float(missing_frame - f0) / float(f1 - f0)
                            cx = (1.0 - ratio) * c0.cx + (ratio * c1.cx)
                            cy = (1.0 - ratio) * c0.cy + (ratio * c1.cy)
                            prev_step = float(np.hypot(cx - c0.cx, cy - c0.cy))
                            next_step = float(np.hypot(c1.cx - cx, c1.cy - cy))
                            if prev_step > radius or next_step > radius:
                                continue
                            x = int(round(cx - (ref_w / 2.0)))
                            y = int(round(cy - (ref_h / 2.0)))
                            recovered = Component(
                                frame_index=missing_frame,
                                x=x,
                                y=y,
                                w=ref_w,
                                h=ref_h,
                                area=ref_area,
                                cx=float(cx),
                                cy=float(cy),
                            )
                            confirmed[missing_frame].append(recovered)

    # Deduplicate by box coordinates per frame.
    deduped: Dict[int, List[Component]] = {}
    for frame_index, comps in confirmed.items():
        seen = set()
        kept: List[Component] = []
        for comp in comps:
            key = (comp.x, comp.y, comp.w, comp.h)
            if key in seen:
                continue
            seen.add(key)
            kept.append(comp)
        deduped[frame_index] = kept
    return deduped


def run_complete_mmb(
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    image_dir: Path,
    max_images: int | None,
    intervention: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    intervention = intervention or {}
    sorted_images = sorted(images, key=lambda item: str(item.get('file_name', '')))
    if max_images is not None:
        sorted_images = sorted_images[:max_images]

    image_ids = [int(item['id']) for item in sorted_images]
    selected_set = set(image_ids)
    gt = [ann for ann in annotations if int(ann.get('image_id', -1)) in selected_set]

    frames: List[np.ndarray] = []
    usable_images: List[Dict[str, Any]] = []
    missing_images: List[int] = []
    for image in sorted_images:
        image_id = int(image['id'])
        image_path = image_dir / str(image.get('file_name', ''))
        if not image_path.exists():
            missing_images.append(image_id)
            continue
        frames.append(_read_gray_frame(image_path))
        usable_images.append(image)

    if not frames:
        return {
            'ground_truth': gt,
            'predictions': [],
            'evaluated_image_ids': image_ids,
            'images_missing_from_disk': missing_images,
            'algorithm': {
                'name': 'mmb_complete',
                'frames_loaded': 0,
            },
        }

    amfd_k = float(intervention.get('amfd_k', 4.0))
    lrmc_l = int(intervention.get('lrmc_l', 4))
    lrmc_k = float(intervention.get('lrmc_k', 2.5))
    pf_len = int(intervention.get('pf_length', 5))
    pf_radius = float(intervention.get('pf_radius', 7.0))
    pf_min_occ = int(intervention.get('pf_min_occurrences', 3))
    lrmc_frame_frequency = float(intervention.get('lrmc_frame_frequency', 10.0))
    pf_recover_partial_tracks = bool(intervention.get('pf_recover_partial_tracks', True))
    min_area = int(intervention.get('component_min_area', 5))
    max_area = int(intervention.get('component_max_area', 80))
    min_aspect_ratio = float(intervention.get('component_min_aspect_ratio', 1.0))
    max_aspect_ratio = float(intervention.get('component_max_aspect_ratio', 6.0))
    stabilize_motion = bool(intervention.get('stabilize_motion', False))
    stabilize_max_step_shift = float(intervention.get('stabilize_max_step_shift', 20.0))
    stabilize_affine = bool(intervention.get('stabilize_affine', False))
    stabilize_affine_max_angle_deg = float(intervention.get('stabilize_affine_max_angle_deg', 1.0))
    stabilize_affine_max_scale_delta = float(intervention.get('stabilize_affine_max_scale_delta', 0.01))
    radiometric_normalize = bool(intervention.get('radiometric_normalize', False))
    score = float(intervention.get('score', 0.95))

    # Process contiguous shape-consistent segments independently.
    segments: List[Tuple[int, int]] = []
    start = 0
    for idx in range(1, len(frames)):
        if frames[idx].shape != frames[idx - 1].shape:
            segments.append((start, idx))
            start = idx
    segments.append((start, len(frames)))

    confirmed_global: Dict[int, List[Component]] = {idx: [] for idx in range(len(frames))}
    for seg_start, seg_end in segments:
        segment_frames = frames[seg_start:seg_end]
        if not segment_frames:
            continue

        segment_masks = None
        if stabilize_motion:
            segment_frames, segment_masks = _stabilize_frames(
                segment_frames,
                max_step_shift=stabilize_max_step_shift,
                use_affine=stabilize_affine,
                affine_max_angle_deg=stabilize_affine_max_angle_deg,
                affine_max_scale_delta=stabilize_affine_max_scale_delta,
                radiometric_normalize=radiometric_normalize,
            )

        amfd = _amfd_components(
            segment_frames,
            k=amfd_k,
            min_area=min_area,
            max_area=max_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            valid_masks=segment_masks,
        )
        lrmc = _lrmc_components(
            segment_frames,
            amfd_components=amfd,
            l=lrmc_l,
            k=lrmc_k,
            min_area=min_area,
            max_area=max_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            valid_masks=segment_masks,
            frame_frequency=lrmc_frame_frequency,
        )
        merged = _merge_components(amfd, lrmc)
        confirmed_local = _trajectory_filter(
            merged,
            pipeline_len=pf_len,
            radius=pf_radius,
            min_occurrences=pf_min_occ,
            recover_partial_tracks=pf_recover_partial_tracks,
        )
        for local_idx, comps in confirmed_local.items():
            global_idx = seg_start + local_idx
            shifted = [
                Component(
                    frame_index=global_idx,
                    x=comp.x,
                    y=comp.y,
                    w=comp.w,
                    h=comp.h,
                    area=comp.area,
                    cx=comp.cx,
                    cy=comp.cy,
                )
                for comp in comps
            ]
            confirmed_global[global_idx].extend(shifted)

    default_category_id = 0
    if gt:
        default_category_id = int(gt[0].get('category_id', 0))

    predictions: List[Dict[str, Any]] = []
    # `confirmed_global` indexes correspond to the order of `usable_images`.
    for frame_index, comps in confirmed_global.items():
        if frame_index < 0 or frame_index >= len(usable_images):
            continue
        image_id = int(usable_images[frame_index]['id'])
        for comp in comps:
            predictions.append(
                {
                    'image_id': image_id,
                    'category_id': default_category_id,
                    'bbox': [float(comp.x), float(comp.y), float(comp.w), float(comp.h)],
                    'score': score,
                }
            )

    return {
        'ground_truth': gt,
        'predictions': predictions,
        'evaluated_image_ids': [int(item['id']) for item in usable_images],
        'images_missing_from_disk': missing_images,
        'algorithm': {
            'name': 'mmb_complete',
            'frames_loaded': len(frames),
            'segments': len(segments),
            'amfd_k': amfd_k,
            'lrmc_l': lrmc_l,
            'lrmc_k': lrmc_k,
            'pf_length': pf_len,
            'pf_radius': pf_radius,
            'pf_min_occurrences': pf_min_occ,
            'pf_recover_partial_tracks': pf_recover_partial_tracks,
            'lrmc_frame_frequency': lrmc_frame_frequency,
            'component_min_area': min_area,
            'component_max_area': max_area,
            'component_min_aspect_ratio': min_aspect_ratio,
            'component_max_aspect_ratio': max_aspect_ratio,
            'stabilize_motion': stabilize_motion,
            'stabilize_max_step_shift': stabilize_max_step_shift,
            'stabilize_affine': stabilize_affine,
            'stabilize_affine_max_angle_deg': stabilize_affine_max_angle_deg,
            'stabilize_affine_max_scale_delta': stabilize_affine_max_scale_delta,
            'radiometric_normalize': radiometric_normalize,
        },
    }
