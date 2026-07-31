from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Dict


ROOT = Path('/Users/edsontm/dev/smalloo')

BASE_DIRECTORIES = [
    'research/ideas',
    'research/proposals',
    'research/literature',
    'research/experiments',
    'research/reports',
    'research/ablations',
    'research/negative_results',
    'research/accepted_methods',
    'research/rejected_methods',
    'research/blog',
    'research/figures',
    'research/tables',
    'research/presentations',
    'src',
    'tests',
    'configs',
    'docs',
]


BASE_FILES = {
    'research/README.md': """# Research Workspace

This workspace mirrors the experiment lifecycle defined in `.github/copilot-instructions.md`.

## Core Rules

- One pull request should carry one scientific contribution.
- Every experiment must preserve reproducibility assets inside the repository.
- Negative results are first-class outputs and should be recorded, not discarded.
- Applied ideas should use explicit versioned experiment names such as `v1_better_registration`, `v2_better_registration`, or `v1_temporal_consistency`.

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

Prefer versioned slugs such as `v1_better_registration` instead of ambiguous names.
""",
    'research/knowledge.md': """# Knowledge Base

Track durable findings from experiments here.

## Sections

### Confirmed Findings

### Negative Results

### Dataset Issues

### Implementation Pitfalls

### Reviewer Concerns

### Next Hypotheses
""",
    'research/ideas/backlog.md': """# Idea Backlog

Rank ideas by expected scientific impact, not engineering convenience.

| Title | Description | Novelty | Cost | Risk | Expected AP Gain | Publication Potential | Related Papers | Dependencies | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Example: Temporal consistency loss audit | Verify whether temporal consistency remains beneficial after stronger registration. | 6 | Medium | Medium | +0.5 AP | Workshop | TBD | Baseline benchmark | Reading |
""",
    'docs/experiment_workflow.md': """# Experiment Workflow

## Bootstrap

Run:

```bash
python3 scripts/init_experiment.py --experiment-slug v1_my_idea --title "V1 My Idea"
```

This creates the base research tree if needed and scaffolds a new experiment under `research/experiments/`.

## Naming rule

Use versioned experiment slugs for every applied idea.

Examples:

- `v1_better_registration`
- `v2_better_registration`
- `v1_temporal_consistency`

If an idea changes substantially, create a new version instead of overwriting the previous experiment.

## Minimum completion criteria

- The eight pre-implementation questions are answered.
- Literature review records strengths, weaknesses, and research gaps.
- Benchmark compares against repository baseline and current best model.
- Ablation isolates one variable per study.
- Statistical validation reports confidence intervals and paired tests.
- Decision file states accepted or rejected and why.
""",
}


EXPERIMENT_FILES: Dict[str, str] = {
    'README.md': """# {title}

## Metadata

- Slug: `{slug}`
- Created: `{created}`
- Status: Planning
- Scientific contribution scope: One hypothesis only
- Version lineage: record parent experiment if this is `v2_...` or later

## Objective

State the exact scientific question this experiment is meant to answer.

## Deliverables

- Reproducible code and config
- Tests
- Benchmark results
- Ablation results
- Statistical validation
- Final decision
""",
    '01_problem_statement.md': """# Problem Statement

## Problem

What problem are we solving?

## Failure of current SOTA

Why does the current SOTA fail on this problem?

## Scope boundaries

What is explicitly in scope and out of scope for this experiment?
""",
    '02_research_hypothesis.md': """# Research Hypothesis

## Required questions

1. What problem are we solving?
2. Why does the current SOTA fail?
3. Why should this idea work?
4. Which paper inspired it?
5. What is the novelty?
6. What is the expected gain?
7. What are the risks?
8. What experiments are required?

## Main hypothesis

State one falsifiable hypothesis.

## Expected contribution

Describe the expected scientific contribution, not just implementation output.
""",
    '03_literature_review.md': """# Literature Review

## Reviewed papers

For each paper record:

- Citation
- Core idea
- Strengths
- Weaknesses
- Research gaps
- Classification: Incremental / Interesting / High Potential / Game Changer
- Implementation notes
- Combination opportunities
""",
    '04_experiment_plan.md': """# Experiment Plan

## Baselines

- Repository baseline
- Current best internal model
- Best published model

## Controlled variable

Describe the single variable changed by this experiment.

## Reproducibility assets

- Config path
- Seed list
- Environment definition
- Data preprocessing notes
- Checkpoint naming plan

## Execution plan

List the exact train, eval, and analysis runs.
""",
    '05_implementation_notes.md': """# Implementation Notes

## Changes

List the code paths, configs, and interfaces touched.

## Risks

List implementation risks that could invalidate the experiment.

## Validation plan

- Unit tests
- Integration tests
- End-to-end tests
""",
    '06_benchmark.md': """# Benchmark

## Comparison set

- Original paper
- Repository baseline
- Current best model
- Best published model

## Metrics

- Precision
- Recall
- F1
- Average Precision
- FPS
- GPU memory
- Parameters
- Training time
- Inference time
- Registration error
- Temporal consistency
- False positives
- False negatives
- Confidence interval
- Paired statistical test

## Results table

Fill with exact values and links to artifacts.
""",
    '07_ablation.md': """# Ablation Study

## Variable isolation

Each row should isolate one factor only.

| Ablation | Changed variable | Control | Result | Interpretation |
| --- | --- | --- | --- | --- |
""",
    '08_statistical_validation.md': """# Statistical Validation

## Repeated runs

Document seeds, number of runs, and hardware.

## Statistical tests

- Confidence intervals
- Paired statistical tests
- Effect sizes

## Interpretation

State whether the gain is statistically credible.
""",
    '09_discussion.md': """# Discussion

## Why it worked or failed

Explain the mechanism, not just the outcome.

## Failure cases

List the main failure modes with evidence.

## Lessons learned

Record findings that should influence future experiments.
""",
    '10_blog_post.md': """# Blog Draft

## Background

## Motivation

## Hypothesis

## Method

## Figures and visualizations

## Results

## Why it worked

## Why it failed

## Lessons learned

## Future ideas
""",
    '11_decision.md': """# Decision

## Outcome

Accepted or Rejected

## Reason

State the evidence supporting the decision.

## Next action

- Merge into accepted methods
- Archive in rejected methods
- Extend with a follow-up experiment
""",
    'artifacts/README.md': """# Artifacts

Store references to:

- configs
- checkpoints
- metrics exports
- figures
- tables
- environment definition
- package versions
- seeds
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bootstrap research structure and optional experiment scaffold.')
    parser.add_argument('--experiment-slug', help='Folder name for a new experiment scaffold.')
    parser.add_argument('--title', help='Human-readable experiment title.')
    parser.add_argument('--force', action='store_true', help='Overwrite scaffold files if they already exist.')
    return parser.parse_args()


def write_text(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    path.write_text(content)


def bootstrap_base(force: bool) -> None:
    for relative_dir in BASE_DIRECTORIES:
        (ROOT / relative_dir).mkdir(parents=True, exist_ok=True)
    for relative_path, content in BASE_FILES.items():
        write_text(ROOT / relative_path, content, force)


def scaffold_experiment(slug: str, title: str, force: bool) -> Path:
    experiment_dir = ROOT / 'research' / 'experiments' / slug
    created = date.today().isoformat()
    replacements = {
        'slug': slug,
        'title': title,
        'created': created,
    }
    for relative_path, template in EXPERIMENT_FILES.items():
        write_text(experiment_dir / relative_path, template.format(**replacements), force)
    return experiment_dir


def main() -> None:
    args = parse_args()
    bootstrap_base(force=args.force)

    if not args.experiment_slug:
        print('Bootstrapped research structure.')
        return

    experiment_title = args.title or args.experiment_slug.replace('-', ' ').title()
    experiment_dir = scaffold_experiment(args.experiment_slug, experiment_title, args.force)
    print(f'Created experiment scaffold at {experiment_dir.relative_to(ROOT)}')


if __name__ == '__main__':
    main()