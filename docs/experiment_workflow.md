# Experiment Workflow

## Bootstrap

Run:

```bash
python3 scripts/init_experiment.py --experiment-slug my-experiment --title "My Experiment"
```

This creates the base research tree if needed and scaffolds a new experiment under `research/experiments/`.

## Validation and run manifests

Use:

```bash
python3 scripts/validate_experiment.py --slug mmb-baseline-reproduction --dataset-profile devsample
python3 scripts/materialize_runs.py --slug mmb-baseline-reproduction --dataset-profile devsample
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
