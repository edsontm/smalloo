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

- The repository still lacks the actual MMB training and evaluation implementation.
- The original MMB preprocessing contract may contain hidden defaults not yet captured here.
- Reported metrics cannot be claimed until the runtime implementation is added and executed on the full dataset.

## Validation plan

- Unit tests
- Integration tests
- End-to-end tests

Current validation entrypoints:

- `python3 -m unittest tests/test_experiment_config.py`
- `python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`
- `python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile viso`
