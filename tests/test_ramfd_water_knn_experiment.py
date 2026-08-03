from __future__ import annotations

import numpy as np

from scripts.ramfd_water_knn_experiment import (
    _apply_color_jitter_to_feature,
    _component_from_box,
    _collect_water_examples_around_box,
    _feature_vector_around_bbox,
    _filter_candidates_by_water,
    _legend_labels,
    _parse_int_csv,
    _point_in_triangle,
    _prepare_run_output_dir,
    _select_tip_component,
    _small_target_detector_boxes,
    _summarize_gt_loss_from_split_results,
    _tiny_object_fallback_boxes,
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


def test_legend_labels_include_counts() -> None:
    assert _legend_labels(tp_count=2, fp_count=3, fn_count=1) == ["TP (2)", "FP (3)", "FN (1)"]


def test_small_target_detector_boxes_return_small_boxes() -> None:
    gray_frames = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
    gray_frames[0][8, 8] = 30.0
    gray_frames[1][8, 8] = 60.0
    gray_frames[2][8, 8] = 30.0

    boxes = _small_target_detector_boxes(gray_frames=gray_frames, frame_idx=1, max_candidates=2)

    assert boxes
    assert len(boxes[0]) == 4
    assert boxes[0][2] <= 24 and boxes[0][3] <= 24
    assert boxes[0][2] >= 6 and boxes[0][3] >= 6


def test_small_target_detector_boxes_use_appearance_when_motion_is_weak() -> None:
    gray_frames = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
    gray_frames[0][8, 8] = 80.0
    gray_frames[1][8, 8] = 80.0
    gray_frames[2][8, 8] = 80.0

    boxes = _small_target_detector_boxes(gray_frames=gray_frames, frame_idx=1, max_candidates=2)

    assert boxes
    assert any(abs(box[0] - 8.0) <= 2.0 and abs(box[1] - 8.0) <= 2.0 for box in boxes)


def test_prepare_run_output_dir_creates_timestamped_summary(tmp_path) -> None:
    run_dir = _prepare_run_output_dir(
        base_dir=tmp_path / "artifacts",
        tag="demo_run",
        max_images=12,
        description="Test description",
    )

    assert run_dir.exists()
    assert run_dir.name.startswith("20")
    assert (run_dir / "README.md").exists()
    summary = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "demo_run" in summary
    assert "Test description" in summary


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


def test_apply_color_jitter_to_feature_returns_bounded_variants() -> None:
    base = np.asarray([40.0, 60.0, 80.0], dtype=np.float32)
    variants = _apply_color_jitter_to_feature(base, feature_mode="rgb_mean", jitter_value=5.0)

    assert len(variants) == 8
    for variant in variants:
        assert variant.shape == (3,)
        assert float(variant.min()) >= 0.0
        assert float(variant.max()) <= 255.0


def test_collect_water_examples_around_box_color_jitter_expands_bank() -> None:
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:, :] = np.array([40, 60, 80], dtype=np.uint8)
    samples = _collect_water_examples_around_box(
        rgb_image=rgb,
        bbox=[20.0, 20.0, 20.0, 20.0],
        margins=[12],
        jitter_px=0,
        feature_mode="rgb_mean",
        color_jitter=5.0,
    )

    assert len(samples) == 9


def test_feature_vector_around_bbox_rgb_mean_std_edge_is_richer() -> None:
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:, :40] = np.array([30, 60, 90], dtype=np.uint8)
    rgb[:, 40:] = np.array([120, 140, 160], dtype=np.uint8)

    feature = _feature_vector_around_bbox(
        rgb_image=rgb,
        bbox=[20.0, 20.0, 20.0, 20.0],
        margin=10,
        feature_mode="rgb_mean_std_edge",
    )

    assert feature is not None
    assert feature.shape == (9,)
    assert float(feature[-1]) >= 0.0


def test_filter_candidates_by_water_negative_bank_rejects_land_like_candidate() -> None:
    candidates = [
        {"box": [0.0, 0.0, 10.0, 10.0], "dist": 6.0, "second_dist": 20.0, "neg_dist": 6.5, "local_thr": 12.0},
        {"box": [20.0, 20.0, 10.0, 10.0], "dist": 6.0, "second_dist": 20.0, "neg_dist": 12.0, "local_thr": 12.0},
    ]

    kept, _ = _filter_candidates_by_water(
        candidates=candidates,
        nearest_gap_min=0.0,
        use_frame_relative_threshold=False,
        frame_relative_margin=26.0,
        max_candidates_per_frame=0,
        use_negative_bank=True,
        negative_margin=1.0,
    )

    assert kept == [[20.0, 20.0, 10.0, 10.0]]


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


def test_summarize_gt_loss_from_split_results_uses_frame_stats() -> None:
    split_results = {
        "train": {
            "frame_stats": [
                {"file_name": "a.jpg", "gt_lost": 1, "gt_lost_by_stage": {"ramfd": 1}},
                {"file_name": "b.jpg", "gt_lost": 0, "gt_lost_by_stage": {}},
            ]
        },
        "validation": {"frame_stats": []},
        "test": {"frame_stats": [{"file_name": "c.jpg", "gt_lost": 2, "gt_lost_by_stage": {"water": 1, "trajectory": 1}}]},
    }

    summary = _summarize_gt_loss_from_split_results(split_results)

    assert summary["stage_counts"] == {"ramfd": 1, "water": 1, "trajectory": 1}
    assert summary["examples"][0]["file_name"] == "a.jpg"
    assert summary["examples"][1]["file_name"] == "c.jpg"
