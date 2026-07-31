# Idea Backlog

Rank ideas by expected scientific impact, not engineering convenience.

| Title | Description | Novelty | Cost | Risk | Expected AP Gain | Publication Potential | Related Papers | Dependencies | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 MMB baseline reproduction | Reproduce the current MMB baseline on VISO with full configs, seeds, tests, and benchmark protocol before any new hypothesis. Slug: `v1_mmb_baseline_reproduction`. | 3 | Medium | Low | 0.0 AP | Internal prerequisite | MMB baseline paper and repo | Dataset wiring and evaluation pipeline | Planning |
| V1 better registration audit | Replace or strengthen registration only, keeping the detector and evaluation protocol fixed, to test whether motion alignment is the bottleneck on VISO ship. Slug: `v1_better_registration_audit`. | 7 | Medium | Medium | +0.5 to +1.5 AP | Conference-tier if gains hold | Registration and satellite video detection papers | V1 MMB baseline reproduction | Planning |
| Example: Temporal consistency loss audit | Verify whether temporal consistency remains beneficial after stronger registration. | 6 | Medium | Medium | +0.5 AP | Workshop | TBD | Baseline benchmark | Reading |
