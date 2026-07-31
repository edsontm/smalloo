# Experiment Plan

## Baselines

- Repository baseline: `MMB` reproduction defined by `configs/experiments/mmb-baseline-reproduction.json`
- Current best internal model: none yet; this experiment creates the first internal reference
- Best published model: fill after literature review, using the same evaluation protocol whenever possible

## Controlled variable

No modeling variable is changed. This is a pure reproduction experiment.

The only allowed changes are infrastructure changes required to freeze the benchmark protocol.

## Reproducibility assets

- Config path: `configs/experiments/mmb-baseline-reproduction.json`
- Dataset profiles: `configs/datasets/viso.json`, `configs/datasets/devsample.json`
- Seed list: `101, 202, 303, 404, 505`
- Environment definition: to be added once the MMB runtime stack is imported into the repository
- Data preprocessing notes: keep the current COCO annotation files unchanged; no augmentation changes are allowed during reproduction
- Checkpoint naming plan: `mmb-baseline-reproduction_<dataset-profile>_seed<seed>`

## Execution plan

1. Smoke validation:
	`python3 scripts/validate_experiment.py --slug mmb-baseline-reproduction --dataset-profile devsample`
2. Smoke run-manifest generation:
	`python3 scripts/materialize_runs.py --slug mmb-baseline-reproduction --dataset-profile devsample`
3. Full-dataset validation:
	`python3 scripts/validate_experiment.py --slug mmb-baseline-reproduction --dataset-profile viso`
4. Full run-manifest generation:
	`python3 scripts/materialize_runs.py --slug mmb-baseline-reproduction --dataset-profile viso`
5. When MMB code is added under `src/`, bind training and evaluation entrypoints to the frozen manifests instead of changing dataset or seed definitions.
