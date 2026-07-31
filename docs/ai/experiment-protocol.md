# Experiment Protocol

All research experiments must follow this protocol.

## Required Sections

Each experiment must define:
- Objective
- Hypothesis
- Implementation scope
- Datasets and profiles
- Metrics
- Baselines
- Ablation studies
- Statistical significance plan
- Reproducibility metadata
- Conclusion and decision

## Variable Isolation

- Change one primary variable per experiment version.
- Keep preprocessing and evaluation protocol fixed unless explicitly studied.

## Reproducibility Requirements

Every result must be traceable to:
- experiment slug
- configuration files
- dataset profile
- random seeds
- runtime profile
- command entrypoints
- generated artifacts

## Baseline and Comparison Rules

- Compare against repository baseline and stated target baseline.
- Use the same split and metric protocol for fair comparison.

## Statistical Validation

- Report confidence intervals when applicable.
- Use paired statistical tests for direct baseline comparisons.
- Record negative results and failure analysis.

## Acceptance Criteria

No experiment is successful without reproducible evidence and protocol-compliant reporting.
