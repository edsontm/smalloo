# Prompt Patterns for AI Agents

Use these reusable prompts to improve consistency across AI tools.

## Bug Fixing

Goal:
- Reproduce, isolate root cause, implement minimal fix, add regression test.

Prompt Template:
- Reproduce this bug using existing tests or commands.
- Identify root cause with file-level evidence.
- Implement the smallest safe fix.
- Add or update regression tests.
- Summarize changed files and verification steps.

## Feature Implementation

Goal:
- Add one focused capability with tests and docs.

Prompt Template:
- Implement feature X with acceptance criteria Y.
- Keep interfaces backward compatible when possible.
- Add tests and update docs/ai if process or policy changes.

## Refactoring

Goal:
- Improve structure without behavior changes.

Prompt Template:
- Refactor module X for readability and testability.
- Preserve behavior and public contracts.
- Prove equivalence with tests.

## Code Review

Goal:
- Find correctness and regression risks first.

Prompt Template:
- Review changes with focus on bugs, regressions, missing tests, and reproducibility gaps.
- List findings by severity and file.

## Architecture Review

Goal:
- Check layering and dependency discipline.

Prompt Template:
- Evaluate whether change respects docs/ai/architecture.md.
- Flag boundary violations and propose alternatives.

## Research Implementation

Goal:
- Implement one research variable change with controlled protocol.

Prompt Template:
- Implement experiment slug vN_name with one controlled variable.
- Keep baseline protocol fixed.
- Generate artifacts required by docs/ai/experiment-protocol.md.

## Experiment Execution

Goal:
- Run deterministic devsample-first workflow.

Prompt Template:
- Execute validate, materialize, and smoke training for slug X on devsample.
- Save outputs under research/experiments/X/artifacts/results.
- Summarize runtime profile, seeds, and completion status.
