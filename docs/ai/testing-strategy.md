# Testing Strategy

## Objectives

- Prevent regressions.
- Validate correctness of configuration and runtime logic.
- Support reproducible experimentation.

## Test Pyramid

- Unit tests: validate isolated functions and modules.
- Integration tests: validate module interactions and CLI workflows.
- End-to-end tests: validate experiment flow on devsample.
- Regression tests: lock behavior for fixed bugs.

## Benchmarks and Performance Checks

- Separate benchmark tasks from functional tests.
- Track runtime-related changes with controlled settings.

## Coverage Expectations

- Target high coverage for core logic modules.
- Prioritize coverage on configuration, runtime selection, and experiment orchestration.

## Deterministic Testing

- Set explicit seeds where stochastic behavior exists.
- Avoid dependence on external mutable state.
- Keep test fixtures stable and lightweight.

## Minimum Validation Before Merge

- Unit tests pass.
- Integration tests for changed paths pass.
- devsample validation and materialization pass for affected experiment slugs.
