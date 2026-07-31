# Project Overview

## Purpose

Smalloo is a research and engineering repository for small object detection in satellite video datasets, with emphasis on reproducible experimentation and evidence-driven improvements.

## Main Technologies

- Python 3
- PyTorch runtime profiles (MPS, CUDA, CPU)
- JSON-based experiment and dataset configuration
- Markdown-based research artifacts and reports

## Architecture Summary

- Runtime and config logic in src
- Automation and orchestration in scripts
- Dataset and experiment configs in configs
- Research lifecycle artifacts in research
- Tests in tests
- Governance and process documentation in docs/ai and AGENTS.md

## Common Commands

- Validate experiment scaffold:
  - python3 scripts/validate_experiment.py --slug <slug> --dataset-profile devsample
- Materialize deterministic run manifests:
  - python3 scripts/materialize_runs.py --slug <slug> --dataset-profile devsample
- Run smoke training:
  - python3 scripts/train_experiment.py --slug <slug> --dataset-profile devsample --trainer smoke --smoke-steps 2
- Run tests:
  - python3 -m unittest discover -s tests

## Repository Layout

- src: core runtime, configuration, and training logic
- scripts: CLI workflows for validation, run materialization, and training
- configs: experiment and dataset profiles
- research: ideas, experiments, reports, and knowledge capture
- docs: process and architecture documentation
- tests: automated test suite

## Read Next

- AGENTS.md
- docs/ai/repository-conventions.md
- docs/ai/development-workflow.md
