# Definition of Done

A task is done only when all applicable items are complete.

## Code

- Implementation is correct, minimal, and maintainable.
- Configuration and runtime contracts are respected.

## Tests

- Relevant unit and integration tests are added or updated.
- Tests pass in the expected environment.

## Documentation

- Canonical documentation is updated in docs/ai when policy or workflow changes.
- Experiment-facing docs are updated when experiment behavior changes.

## Validation

- Validation and run-materialization commands succeed for affected experiment slugs.
- Evidence artifacts are generated and stored in expected paths.

## Review

- PR review feedback is addressed.
- Risks, assumptions, and trade-offs are documented.

## Performance and Security

- Performance impact is documented for critical paths.
- Security checks for secrets and unsafe input handling are satisfied.

## Reproducibility

- Results can be reproduced from committed code, config, and commands.
