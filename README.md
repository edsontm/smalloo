# smalloo

Research and engineering repository for small-vessel detection in satellite videos, with emphasis on a classical and interpretable pipeline based on motion, water color, and trajectory filtering.

## Main idea

The central proposal is to combine:

- candidate detection via RAMFD;
- filtering by water-color and water-environment similarity;
- trajectory- and wake-guided selection to reduce false positives;
- a robust fallback rule when trajectory-based selection is unreliable.

In short, the method aims to improve precision without losing the ability to recover real vessels in difficult scenarios.

## Main results

The results reported for the full method in the current experiment were:

- Precision: 0.9478
- Recall: 0.9083
- F1: 0.9277

For reference, the compared baseline achieved:

- Precision: 0.8538
- Recall: 0.9250
- F1: 0.8880

In addition, ablations of the method showed a sharp drop when critical components were removed, confirming the role of the water filter and trajectory-based selection.

## How to install

From a clone of the repository:

```bash
cd smalloo
python3 -m pip install .
```

Or, for development:

```bash
cd smalloo
python3 -m pip install -e .
```

## How to run

The main experiment can be run with:

```bash
smalloo-ramfd-water-knn --max-images 120
```

Optionally, you can pass arguments such as:

```bash
smalloo-ramfd-water-knn --max-images 120 --tag my_run --annotations-path path/to/annotations.json --image-dir path/to/images
```

## Important structure

- [scripts/ramfd_water_knn_experiment.py](scripts/ramfd_water_knn_experiment.py): main experiment pipeline.
- [src/mmb_complete.py](src/mmb_complete.py): implementation of the RAMFD and trajectory-filtering stages.
- [src/viso_evaluation.py](src/viso_evaluation.py): evaluation and metrics in a format compatible with the VISO protocol.
- [research/papers/aaai_ramfd_water_knn_paper.tex](research/papers/aaai_ramfd_water_knn_paper.tex): LaTeX version of the paper.

## Reproducing the paper results

To recreate the main results from a fresh clone of the repository, follow this workflow:

### 1. Prepare the environment

```bash
git clone <repository-url>
cd smalloo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

### 2. Check the data

The experiment expects the data in VISO/COCO format under the repository tree:

- [VISO/coco/ship/Annotations/instances_train2017.json](VISO/coco/ship/Annotations/instances_train2017.json)
- [VISO/coco/ship/train2017](VISO/coco/ship/train2017)
- [VISO/coco/ship/Annotations/instances_test2017.json](VISO/coco/ship/Annotations/instances_test2017.json)
- [VISO/coco/ship/test2017](VISO/coco/ship/test2017)

If your local layout differs, pass the paths explicitly on the command line.

### 3. Run the main result

```bash
smalloo-ramfd-water-knn \
  --max-images 120 \
  --tag reproducibility_main \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2
```

Artifacts will be generated inside:

- [research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug](research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug)

### 4. Run the ablations

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

### 5. Check the metrics

After each run, inspect the generated report.json file in the run folder. The main result expected is close to:

- Precision: 0.9478
- Recall: 0.9083
- F1: 0.9277

### Development and tests

To run the tests:

```bash
python3 -m unittest discover -s tests
```

For repository documentation and internal rules, see:

- [AGENTS.md](AGENTS.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/ai/project-overview.md](docs/ai/project-overview.md)
