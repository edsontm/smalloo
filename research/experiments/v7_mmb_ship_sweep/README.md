# V7 MMB Ship Sweep

- Slug: `v7_mmb_ship_sweep`
- Status: Completed
- Dataset profile: `viso` / `ship`
- Input mode: image directory from `VISO/coco/ship/test2017`
- Primary variable: foreground threshold sweep

## Objective

Run a lightweight classical MMB sweep on the VISO ship split using the new multi-frame-difference + robust matrix completion pipeline.

## Reproducibility

- Command: `python scripts/run_mmb_ship_sweep.py`
- Frame sample: first 32 sorted ship frames
- Thresholds tested: `0.85`, `0.90`, `0.95`
- Temporal windows: `(1, 2, 3)`
- Robust matrix completion: `max_iter=60`, `tol=1e-5`

## Outcome

All three thresholds produced zero detections on the 32-frame sample, so the overlap-based VISO metrics were all zero:

- precision: `0.0`
- recall: `0.0`
- F1: `0.0`
- AP: `0.0`
- mAP: `0.0`
- TP: `0`
- FP: `0`
- FN: `32`

This is a negative result and indicates the current foreground thresholding is too strict for the sampled ship subset, or the RPCA output still needs calibration before a full reproduction attempt.

## Artifacts

- Summary: `research/experiments/v7_mmb_ship_sweep/artifacts/results/sweep_summary.json`
- Per-threshold outputs: `research/experiments/v7_mmb_ship_sweep/artifacts/results/threshold_*/`