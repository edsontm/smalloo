# Repository Conventions

## Directory Organization

- Keep runtime code in src.
- Keep executable workflows in scripts.
- Keep configuration in configs.
- Keep experiment and research evidence in research.
- Keep policy and process docs in docs/ai.

## Naming Conventions

- Experiment slug format: v<version>_<idea_slug>
  - Example: v1_mmb_baseline_reproduction
- Do not overwrite prior experiment versions.
- Keep file names descriptive and stable.

## Generated Files

- Run manifests: research/experiments/<slug>/artifacts/runs
- Run outputs and summaries: research/experiments/<slug>/artifacts/results
- Consolidated reports: research/reports

Generated outputs should be reproducible from committed code and config.

## Protected and Canonical Files

- AGENTS.md is the repository contract.
- docs/ai/* are canonical detailed policies.
- Tool-specific instruction files must reference canonical docs.

## Datasets and Large Files

- Raw datasets are not expected to be committed unless explicitly required.
- Use dataset profiles in configs/datasets for path indirection.
- Keep deterministic sub-samples under agreed repository paths.

## Artifacts and Reproducibility

- Every reported result must be traceable to:
  - experiment slug
  - dataset profile
  - seed
  - runtime profile
  - command entrypoint
