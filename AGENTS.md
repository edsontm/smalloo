# Repository Contract for Human and AI Contributors

This repository is AI-native and agent-compatible by design.

## Scope

This contract applies to:
- Human contributors
- Claude Code
- GitHub Copilot
- Cursor
- Codex
- OpenHands
- Future coding agents

## Single Source of Truth

- This file defines the repository contract.
- Detailed rules live under docs/ai.
- Tool-specific bootstrap files must reference this contract and must not redefine policy.

## Mandatory Read Order

1. AGENTS.md
2. docs/ai/project-overview.md
3. docs/ai/repository-conventions.md
4. docs/ai/development-workflow.md
5. docs/ai/coding-standards.md
6. docs/ai/testing-strategy.md
7. docs/ai/experiment-protocol.md
8. docs/ai/definition-of-done.md

## Engineering Philosophy

- Optimize for scientific rigor, reproducibility, and maintainability.
- Prefer explicit decisions over implicit behavior.
- Minimize hidden state and undocumented assumptions.
- Treat failed experiments as valuable outputs.

Reference: docs/ai/project-overview.md, docs/ai/experiment-protocol.md

## Coding Standards

- Follow project naming, typing, and error-handling rules.
- Keep modules focused and testable.
- Avoid dead code and hidden configuration.

Reference: docs/ai/coding-standards.md

## Development Workflow

- One pull request should carry one primary contribution.
- Run validation and tests before opening or merging a PR.
- Use versioned experiment slugs for research changes.

Reference: docs/ai/development-workflow.md, docs/ai/repository-conventions.md

## Architecture Principles

- Respect layering and module boundaries.
- Depend on interfaces at boundaries.
- Keep runtime configuration centralized.

Reference: docs/ai/architecture.md

## Testing Requirements

- Add or update unit, integration, and regression tests for behavior changes.
- Keep tests deterministic whenever feasible.

Reference: docs/ai/testing-strategy.md

## Documentation Policy

- Update docs in the same PR as behavior changes.
- Do not duplicate policy across multiple files.
- Link to canonical docs instead of rewriting them.

Reference: docs/ai/repository-conventions.md

## Security Policy

- Never commit secrets or credentials.
- Validate untrusted inputs and file paths.
- Record security-impacting changes in review notes.

Reference: docs/ai/coding-standards.md, docs/ai/definition-of-done.md

## Review Policy

- Reviews must check correctness, tests, reproducibility, and documentation.
- Research PRs must check variable isolation and baseline consistency.

Reference: docs/ai/development-workflow.md, docs/ai/experiment-protocol.md

## Definition of Done

A task is done only when code, tests, docs, validation evidence, and review criteria are satisfied.

Reference: docs/ai/definition-of-done.md

## Reporting Requirements

- Keep experiment reports reproducible and traceable to config, seed, and runtime.
- Store artifacts and summary reports in repository-defined locations.

Reference: docs/ai/experiment-protocol.md, docs/reports/README.md
