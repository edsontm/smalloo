# Better Registration Audit

## Metadata

- Slug: `better-registration-audit`
- Created: `2026-07-31`
- Status: Planning
- Scientific contribution scope: One hypothesis only
- Baseline dependency: `mmb-baseline-reproduction`
- Controlled variable: registration stage only

## Objective

Test whether improving frame-to-frame registration alone reduces false positives and improves temporal consistency on VISO ship.

This experiment must inherit the dataset handling, seeds, and reporting contract established by the baseline reproduction.

## Deliverables

- Reproducible code and config
- Tests
- Benchmark results
- Ablation results
- Statistical validation
- Final decision

## Immediate runbook

1. Confirm the MMB baseline reproduction protocol is frozen.
2. Validate this experiment against `devsample`.
3. Freeze per-seed manifests with the same seed list as the baseline.
4. Change only the registration stage during implementation.
