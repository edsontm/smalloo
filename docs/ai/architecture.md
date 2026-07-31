# Architecture

## Architectural Style

- Layered, script-driven research architecture.
- Deterministic workflow orchestration around configuration contracts.

## Layering

- Interface layer: scripts for CLI workflows.
- Application layer: experiment configuration and training orchestration.
- Infrastructure layer: runtime detection, filesystem I/O, and artifact persistence.

## Dependency Rules

- scripts may depend on src.
- src modules should avoid depending on scripts.
- tests may depend on src and script entrypoints.
- configs and research artifacts are data inputs, not code dependencies.

## Module Boundaries

- Configuration boundary: src/experiment_config.py loads and validates data contracts.
- Runtime boundary: src/runtime_profile.py selects accelerator behavior.
- Training boundary: src/trainer_registry.py resolves trainer implementations.

## Interfaces and Contracts

- CLI entrypoints accept explicit flags for slug and dataset profile.
- Run manifests are serialized JSON artifacts for reproducibility.
- Runtime contract is centralized and exposed to training paths.

## Inversion of Control

- Trainers are selected via registry rather than hardcoded imports in orchestration scripts.
- Runtime behavior is injected through profile resolution, not duplicated checks.

## Design Principles

- Keep experiment logic data-driven through configs.
- Keep side effects explicit and artifact outputs stable.
- Prefer extension by adding new trainers or config profiles over editing unrelated code.
