# Implementation Notes

## Changes

- `configs/datasets/viso.json`
- `configs/datasets/devsample.json`
- `configs/experiments/v1_mmb_baseline_reproduction.json`
- `src/experiment_config.py`
- `scripts/validate_experiment.py`
- `scripts/materialize_runs.py`
- `tests/test_experiment_config.py`

## Risks

- The complete MMB runtime is now active for v1, but metric sensitivity to temporal continuity remains high.
- The original MMB preprocessing contract may still contain hidden defaults not fully captured in public text.
- Reported metrics still require reproducible full-dataset execution evidence per seed.

## Runtime contract for v1

- `configs/experiments/v1_mmb_baseline_reproduction.json` now fixes `intervention.inference_mode` to `complete`.
- The intervention is pinned to AMFD + LRMC + PF defaults aligned with the paper behavior.
- Non-paper stabilization knobs are disabled for v1 (`stabilize_motion=false`, `stabilize_affine=false`, `radiometric_normalize=false`).

## Validation plan

- Unit tests
- Integration tests
- End-to-end tests

Current validation entrypoints:

- `python3 -m unittest tests/test_experiment_config.py`
- `python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`
- `python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile viso`
