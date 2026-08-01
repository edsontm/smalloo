# Experiment Plan

## Baselines

- Repository baseline: `MMB` reproduction defined by `configs/experiments/v1_mmb_baseline_reproduction.json`
- Current best internal model: none yet; this experiment creates the first internal reference
- Best published model: fill after literature review, using the same evaluation protocol whenever possible

## Controlled variable

No modeling variable is changed. This is a pure reproduction experiment.

The only allowed changes are infrastructure changes required to freeze the benchmark protocol.

## Reproducibility assets

- Config path: `configs/experiments/v1_mmb_baseline_reproduction.json`
- Dataset profiles: `configs/datasets/viso.json`, `configs/datasets/devsample.json`
- Seed list: `101, 202, 303, 404, 505`
- Environment definition: to be added once the MMB runtime stack is imported into the repository
- Data preprocessing notes: keep the current COCO annotation files unchanged; no augmentation changes are allowed during reproduction
- Checkpoint naming plan: `v1_mmb_baseline_reproduction_<dataset-profile>_seed<seed>`

## Execution plan

1. Smoke validation:
	`python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`
2. Smoke run-manifest generation:
	`python3 scripts/materialize_runs.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`
3. Full-dataset validation:
	`python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile viso`
4. Full run-manifest generation:
	`python3 scripts/materialize_runs.py --slug v1_mmb_baseline_reproduction --dataset-profile viso`
5. Run baseline evaluation from frozen manifests using complete MMB mode:
	`python3 scripts/train_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample --trainer mmb --smoke-steps 1`
6. Run reported evaluation on VISO using the same frozen manifests and intervention:
	`python3 scripts/train_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile viso --trainer mmb --smoke-steps 1`

## Paper-fidelity runtime contract

- Inference mode is fixed to `complete` in the experiment config.
- The intervention uses AMFD + LRMC + PF with paper-aligned defaults.
- Component filtering is fixed to paper bounds (`5..80` area and `1.0..6.0` aspect ratio).
