# Reproducing the RAMFD + Water-KNN Results

This guide provides a concrete workflow to recreate the main results reported in the paper from a fresh clone of this repository.

## 1. Environment setup

Create and activate a Python environment:

```bash
git clone <repository-url>
cd smalloo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

If the environment already exists, simply activate it and reinstall in editable mode:

```bash
source .venv/bin/activate
python3 -m pip install -e .
```

## 2. Data assumptions

The experiment expects the VISO-style COCO annotations and image directories to be available under the repository tree:

- VISO/coco/ship/Annotations/instances_train2017.json
- VISO/coco/ship/train2017
- VISO/coco/ship/Annotations/instances_test2017.json
- VISO/coco/ship/test2017

If your local checkout uses a different layout, pass the paths explicitly via the CLI flags.

## 3. Reproduce the main result

Run the main experiment with the same configuration used for the reported run:

```bash
smalloo-ramfd-water-knn \
  --max-images 120 \
  --tag reproducibility_main \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2
```

This will generate a timestamped artifact directory under:

- research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/

The output includes:

- report.json
- report.md
- README.md
- optional frame visualizations and contact sheets

## 4. Reproduce the ablations

The paper also includes ablation runs. Run them separately:

```bash
smalloo-ramfd-water-knn \
  --max-images 120 \
  --tag ablation_no_traj_no_wake \
  --use-single-tip-selection false \
  --use-trajectory-tip-filter false \
  --use-wake-triangle-filter false
```

```bash
smalloo-ramfd-water-knn \
  --max-images 120 \
  --tag ablation_traj_no_wake \
  --use-single-tip-selection true \
  --use-trajectory-tip-filter true \
  --use-wake-triangle-filter false
```

These produce separate folders that can be compared with the main run.

## 5. Check the reported metrics

After each run, inspect the generated report JSON:

```bash
python3 - <<'PY'
import json
from pathlib import Path
report = Path("research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug")
latest = sorted(report.glob("**/report.json"))[-1]
with latest.open() as f:
    data = json.load(f)
print(data["metrics"]["water_filtered_ramfd"])
PY
```

The main reported values should be close to:

- precision: 0.9478
- recall: 0.9083
- f1: 0.9277

## 6. Verify the paper artifacts

The paper source is in:

- research/papers/aaai_ramfd_water_knn_paper.tex

A compiled PDF can be produced with LaTeX if the local environment supports it.

## 7. Expected output structure

Each run should create a folder with the following files:

- README.md
- report.json
- report.md
- optional frame-rendered diagnostics

These artifacts are the main evidence for the reported results.
