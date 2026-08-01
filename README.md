# smalloo

AI-native research and engineering repository for small object detection in satellite videos.

## Start Here

- Repository contract: AGENTS.md
- Contributor guide: CONTRIBUTING.md
- AI workflow documentation: docs/ai/project-overview.md

## Single Source of Truth

Detailed policies are centralized in docs/ai.
Tool-specific files (Claude, Copilot, and others) should reference AGENTS.md and docs/ai instead of redefining rules.

## Core Commands

- Validate experiment:
	- python3 scripts/validate_experiment.py --slug <slug> --dataset-profile devsample
- Materialize deterministic runs:
	- python3 scripts/materialize_runs.py --slug <slug> --dataset-profile devsample
- Smoke train:
	- python3 scripts/train_experiment.py --slug <slug> --dataset-profile devsample --trainer smoke --smoke-steps 2
- Tests:
	- python3 -m unittest discover -s tests

## Documentation Map

- docs/ai: canonical engineering and research process
- docs/architecture: architecture overview and ADRs
- docs/experiments: experiment documentation templates
- docs/reports: reporting conventions

## MMB Baseline (Complete Classical Pipeline)

This repository now includes a full non-deep-learning Motion Modeling Baseline (MMB) implementation aligned with the VISO paper's classical motion-modeling direction.

Implementation modules are in src/mmb:

- registration.py: frame registration (ORB+affine with deterministic fallback)
- background.py: temporal median, running average, and robust PCA (IALM)
- foreground.py: frame-background differencing + normalization + threshold + morphology
- detection.py: connected-component motion detection and bounding boxes
- tracking.py: multi-object tracking with Kalman prediction + Hungarian assignment
- pipeline.py: end-to-end execution and artifact export
- metrics.py: detection and tracking metrics helpers
- visualization.py: detection/track overlays and video rendering

### Mathematical Formulation

The implementation follows the paper's temporal-motion strategy:

- Registration removes global motion before local motion analysis.
- Background decomposition uses low-rank + sparse separation:
	X = L + S
- Foreground is extracted from absolute difference:
	D_t = |I_t - B_t|
- AMFD-inspired differencing is implemented in the classical MMB mode used by src/viso_mmb.py.
- Tracking uses a constant-velocity state model x = [x, y, vx, vy] with assignment over frame-wise detections.

### Reproducible Experiment Run

Run the standalone MMB experiment pipeline:

- python3 experiments/run_mmb.py --config configs/mmb.yaml --video path/to/video.mp4

Generated outputs under results/mmb include:

- detections.json
- tracks.json
- visualization.mp4
- metrics.json
- experiment_results.csv

### Tests

MMB-specific tests:

- tests/test_registration.py
- tests/test_background.py
- tests/test_foreground.py
- tests/test_detection.py
- tests/test_tracking.py
- tests/test_pipeline.py
