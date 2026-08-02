from __future__ import annotations

import numpy as np

from scripts.ramfd_water_knn_experiment import (
    _component_from_box,
    _collect_water_examples_around_box,
    _filter_candidates_by_water,
    _parse_int_csv,
    _point_in_triangle,
    _select_tip_component,
    _wake_triangle_points,
)


def test_filter_candidates_by_water_base_threshold_only() -> None:
    candidates = [
        {"box": [0.0, 0.0, 10.0, 10.0], "dist": 8.0, "second_dist": 20.0, "local_thr": 12.0},
        {"box": [20.0, 20.0, 10.0, 10.0], "dist": 11.5, "second_dist": 30.0, "local_thr": 12.0},
        {"box": [40.0, 40.0, 10.0, 10.0], "dist": 15.0, "second_dist": 25.0, "local_thr": 12.0},
    ]

    kept, diag = _filter_candidates_by_water(
        candidates=candidates,
        nearest_gap_min=0.0,
        use_frame_relative_threshold=False,
        frame_relative_margin=26.0,
        max_candidates_per_frame=0,
    )

    assert len(kept) == 2
    assert diag["relative_thr"] == float("inf")


def test_filter_candidates_by_water_frame_relative_tightens_tail() -> None:
    candidates = [
        {"box": [0.0, 0.0, 10.0, 10.0], "dist": 4.0, "second_dist": 20.0, "local_thr": 100.0},
        {"box": [20.0, 20.0, 10.0, 10.0], "dist": 18.0, "second_dist": 35.0, "local_thr": 100.0},
        {"box": [40.0, 40.0, 10.0, 10.0], "dist": 34.0, "second_dist": 55.0, "local_thr": 100.0},
    ]

    kept, diag = _filter_candidates_by_water(
        candidates=candidates,
        nearest_gap_min=0.0,
        use_frame_relative_threshold=True,
        frame_relative_margin=12.0,
        max_candidates_per_frame=0,
    )

    assert len(kept) == 2
    assert diag["relative_thr"] == 16.0


def test_filter_candidates_by_water_topk_cap() -> None:
    candidates = [
        {"box": [0.0, 0.0, 10.0, 10.0], "dist": 10.0, "second_dist": 30.0, "local_thr": 100.0},
        {"box": [20.0, 20.0, 10.0, 10.0], "dist": 9.0, "second_dist": 25.0, "local_thr": 100.0},
        {"box": [40.0, 40.0, 10.0, 10.0], "dist": 6.0, "second_dist": 22.0, "local_thr": 100.0},
    ]

    kept, _ = _filter_candidates_by_water(
        candidates=candidates,
        nearest_gap_min=0.0,
        use_frame_relative_threshold=False,
        frame_relative_margin=26.0,
        max_candidates_per_frame=2,
    )

    assert len(kept) == 2
    assert kept[0] == [40.0, 40.0, 10.0, 10.0]
    assert kept[1] == [20.0, 20.0, 10.0, 10.0]


def test_parse_int_csv_parses_and_ignores_empty_tokens() -> None:
    assert _parse_int_csv("12, 18,,26") == [12, 18, 26]


def test_collect_water_examples_around_box_generates_multiple_samples() -> None:
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:, :] = np.array([40, 60, 80], dtype=np.uint8)
    samples = _collect_water_examples_around_box(
        rgb_image=rgb,
        bbox=[20.0, 20.0, 20.0, 20.0],
        margins=[12, 18],
        jitter_px=6,
    )

    # 2 margins * (center + 8 offsets)
    assert len(samples) == 18
    for sample in samples:
        assert sample.shape == (3,)


def test_point_in_triangle_detects_inside() -> None:
    tri = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert _point_in_triangle(1.0, 1.0, tri)
    assert not _point_in_triangle(9.0, 9.0, tri)


def test_select_tip_component_prefers_forward_projection() -> None:
    prev_prev = _component_from_box(0, [10.0, 10.0, 8.0, 8.0])
    prev = _component_from_box(1, [20.0, 10.0, 8.0, 8.0])
    c_front = _component_from_box(2, [32.0, 10.0, 8.0, 8.0])
    c_back = _component_from_box(2, [16.0, 10.0, 8.0, 8.0])
    tip, motion = _select_tip_component([c_back, c_front], prev_tip=prev, prev_prev_tip=prev_prev)
    assert tip is c_front
    assert motion is not None


def test_wake_triangle_points_returns_three_vertices() -> None:
    tri = _wake_triangle_points(
        tip_cx=20.0,
        tip_cy=20.0,
        dir_x=1.0,
        dir_y=0.0,
        wake_length=10.0,
        wake_half_width=5.0,
    )
    assert len(tri) == 3
