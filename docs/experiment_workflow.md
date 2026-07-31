# Experiment Workflow

## Bootstrap

Run:

```bash
python3 scripts/init_experiment.py --experiment-slug v1_my_idea --title "V1 My Idea"
```

This creates the base research tree if needed and scaffolds a new experiment under `research/experiments/`.

## Naming rule

Every applied idea should become a versioned experiment.

Use names like:

- `v1_better_registration`
- `v2_better_registration`
- `v1_temporal_consistency`

Do not overwrite old experiment folders when an idea evolves. Create a new version and link it to the previous one.

## Validation and run manifests

Default rule: always run on `devsample` first. Only move to `VISO` after the devsample validation and smoke run pass.

Use:

```bash
python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample
python3 scripts/materialize_runs.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample
```

`validate_experiment.py` checks scaffold completeness and dataset layout.
`materialize_runs.py` creates deterministic run manifests under the experiment's `artifacts/runs/` folder.

## Minimum completion criteria

- The eight pre-implementation questions are answered.
- Literature review records strengths, weaknesses, and research gaps.
- Benchmark compares against repository baseline and current best model.
- Ablation isolates one variable per study.
- Statistical validation reports confidence intervals and paired tests.
- Decision file states accepted or rejected and why.
