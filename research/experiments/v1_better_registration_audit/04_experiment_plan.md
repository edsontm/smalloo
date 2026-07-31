# Experiment Plan

## Baselines

- Repository baseline: reproduced MMB baseline from `v1_mmb_baseline_reproduction`
- Current best internal model: baseline result unless this experiment beats it credibly
- Best published model: fill after literature review with matched protocol

## Controlled variable

The only changed variable is the registration stage.

Detector weights, preprocessing, splits, seed list, thresholds, and evaluation logic should remain fixed.

## Reproducibility assets

- Config path: `configs/experiments/v1_better_registration_audit.json`
- Dataset profiles: `configs/datasets/viso.json`, `configs/datasets/devsample.json`
- Seed list: `101, 202, 303, 404, 505`
- Environment definition: must match the baseline environment once imported
- Data preprocessing notes: identical to the baseline reproduction
- Checkpoint naming plan: `v1_better_registration_audit_<dataset-profile>_seed<seed>`

## Execution plan

1. Validate the inherited scaffold:
	`python3 scripts/validate_experiment.py --slug v1_better_registration_audit --dataset-profile devsample`
2. Freeze smoke-test manifests:
	`python3 scripts/materialize_runs.py --slug v1_better_registration_audit --dataset-profile devsample`
3. After implementation, run five seeds on `VISO` with the same seeds as the baseline.
4. Compare against `v1_mmb_baseline_reproduction` using AP, false positives, temporal consistency, and registration error.
