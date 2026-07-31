# Implementation Notes

## Changes

- Registration module implementation, once available under `src/`
- `configs/experiments/v1_better_registration_audit.json`
- Shared validation and run-materialization scripts

## Risks

- The proposed registration change may silently alter preprocessing assumptions.
- Runtime cost may increase enough to make gains unattractive.
- Registration metrics may improve without helping AP.

## Validation plan

- Unit tests
- Integration tests
- End-to-end tests

Current validation entrypoints:

- `python3 scripts/validate_experiment.py --slug v1_better_registration_audit --dataset-profile devsample`
- `python3 scripts/materialize_runs.py --slug v1_better_registration_audit --dataset-profile devsample`
