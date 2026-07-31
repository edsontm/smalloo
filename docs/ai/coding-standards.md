# Coding Standards

## Naming

- Use descriptive names for modules, functions, and variables.
- Follow experiment slug format v<version>_<idea_slug>.

## Formatting

- Keep formatting consistent within each file.
- Prefer readability over compactness.
- Avoid unrelated formatting changes in functional PRs.

## Comments and Documentation

- Add comments only when intent is non-obvious.
- Keep docstrings concise and factual.
- Update docs when behavior or contracts change.

## Logging and Errors

- Use structured, informative messages.
- Fail fast on invalid configuration.
- Surface actionable error context for dataset and runtime issues.

## Exceptions

- Raise specific exceptions where possible.
- Avoid broad exception suppression.
- Keep error handling close to boundaries where recovery is possible.

## Typing

- Use type hints for public functions and interfaces.
- Keep function signatures explicit.

## Performance

- Optimize after correctness and reproducibility are ensured.
- Profile bottlenecks before changing architecture.

## Security

- Never commit secrets.
- Validate external inputs and paths.
- Avoid executing shell commands from untrusted data.
