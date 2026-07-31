# smalloo

AI-native research and engineering repository for small object detection in satellite videos.

## Start Here

- Repository contract: AGENTS.md
- Contributor guide: CONTRIBUTING.md
- AI workflow documentation: docs/ai/project-overview.md

## Single Source of Truth

Detailed policies are centralized in docs/ai.
Tool-specific files (Claude, Copilot, and others) should reference AGENTS.md and docs/ai instead of redefining rules.

## Core Commands

- Validate experiment:
	- python3 scripts/validate_experiment.py --slug <slug> --dataset-profile devsample
- Materialize deterministic runs:
	- python3 scripts/materialize_runs.py --slug <slug> --dataset-profile devsample
- Smoke train:
	- python3 scripts/train_experiment.py --slug <slug> --dataset-profile devsample --trainer smoke --smoke-steps 2
- Tests:
	- python3 -m unittest discover -s tests

## Documentation Map

- docs/ai: canonical engineering and research process
- docs/architecture: architecture overview and ADRs
- docs/experiments: experiment documentation templates
- docs/reports: reporting conventions
