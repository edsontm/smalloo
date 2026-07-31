# Research Workspace

This workspace follows the repository contract in AGENTS.md and the canonical process docs in docs/ai.

## Core Rules

- One pull request should carry one scientific contribution.
- Every experiment must preserve reproducibility assets inside the repository.
- Negative results are first-class outputs and should be recorded, not discarded.
- Every applied idea should use a versioned experiment slug such as v1_better_registration or v2_better_registration.

## Main Folders

- `ideas/`: ranked backlog and rough concepts.
- `literature/`: paper summaries, implementation notes, and gap analyses.
- `experiments/`: experiment-specific working directories.
- `reports/`: finalized experiment reports.
- `ablations/`: targeted variable-isolation studies.
- `negative_results/`: archived failed ideas with failure analysis.
- `blog/`: lab notebook style posts for every experiment.
- `figures/`, `tables/`, `presentations/`: publication assets.

## Standard Lifecycle

1. Problem statement
2. Literature review
3. Research gap analysis
4. Hypothesis
5. Expected contribution
6. Implementation
7. Unit, integration, and end-to-end tests
8. Benchmark
9. Ablation
10. Statistical validation
11. Discussion
12. Blog entry
13. Decision: accepted or rejected

Use `python3 scripts/init_experiment.py --experiment-slug <slug> --title <title>` to create a new experiment scaffold.

Prefer versioned slugs for all new experiment folders and reports.
