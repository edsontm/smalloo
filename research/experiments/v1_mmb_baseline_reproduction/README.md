# MMB Baseline Reproduction

## Metadata

- Slug: `v1_mmb_baseline_reproduction`
- Created: `2026-07-31`
- Status: Planning
- Scientific contribution scope: One hypothesis only
- Baseline target: `MMB`
- Primary dataset slice: `VISO/coco/ship`
- Smoke-test dataset slice: `devsample/coco/ship`

## Objective

Establish a reproducible MMB baseline on the ship subset of VISO before any method changes are attempted.

The main outcome is not an accuracy gain. The outcome is a stable benchmark reference with fixed seeds, fixed preprocessing, and a fixed reporting protocol that future hypotheses can compare against.

## Deliverables

- Reproducible code and config
- Tests
- Benchmark results
- Ablation results
- Statistical validation
- Final decision

## Immediate runbook

1. Validate the devsample layout with `python3 scripts/validate_experiment.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`.
2. Generate deterministic per-seed manifests with `python3 scripts/materialize_runs.py --slug v1_mmb_baseline_reproduction --dataset-profile devsample`.
3. Repeat validation and manifest generation on `VISO` before reported runs.
4. Attach the future MMB training and evaluation entrypoints to the frozen manifests rather than changing the protocol.
