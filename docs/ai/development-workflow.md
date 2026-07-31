# Development Workflow

## Contribution Types

- Feature development
- Bug fixing
- Refactoring
- Research experiment implementation
- Documentation updates

## Standard Flow

1. Confirm scope and acceptance criteria.
2. Create or update experiment slug when research behavior changes.
3. Implement minimal, focused changes.
4. Run validation and tests.
5. Update documentation and reports.
6. Open PR with evidence.

## Feature Development

- Keep one primary contribution per PR.
- Prefer small, reviewable commits.
- Keep public interfaces stable unless change is justified.

## Bug Fixing

- Reproduce first.
- Add regression test when feasible.
- Document root cause and verification steps in PR.

## Branching and Pull Requests

- Use short-lived branches.
- Reference experiment slug(s) in PR when applicable.
- Include commands run and artifact paths.

## Release and Merge Discipline

- Do not merge without required tests and documentation updates.
- For research-impacting changes, include protocol-compliant evidence.

## Required Commands Before PR

- python3 scripts/validate_experiment.py --slug <slug> --dataset-profile devsample
- python3 scripts/materialize_runs.py --slug <slug> --dataset-profile devsample
- python3 -m unittest discover -s tests
