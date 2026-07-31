# Statistical Validation

## Repeated runs

Planned seeds: `101, 202, 303, 404, 505`

Planned number of reported runs: `5`

Hardware: fill once runtime implementation is available.

## Statistical tests

- Confidence intervals
- Paired statistical tests
- Effect sizes

The key comparison is paired baseline-vs-registration-candidate performance under the same seeds.

## Interpretation

An observed gain only counts if it beats seed variance from the reproduced baseline and improves at least one alignment-sensitive metric, not just AP.
