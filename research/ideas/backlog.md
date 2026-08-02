# Idea Backlog

Rank ideas by expected scientific impact, not engineering convenience.

Backlog-first policy (canonical): every new idea must be registered here before implementation; after execution, update outcome and evidence.

| Title | Description | Dependencies | Status | Outcome | Evidence |
| --- | --- | --- | --- | --- | --- |
| V1 MMB baseline reproduction | Reproduce baseline MMB on VISO with reproducible configs and reports. | Dataset/eval pipeline | Implemented | Worked as reproducible baseline; limited detection quality on ships. | research/experiments/v1_mmb_baseline_reproduction |
| V1 better registration audit | Test stronger alignment while keeping detector/eval fixed. | V1 baseline | Implemented | Mixed/limited gains; not enough to solve high FP regime alone. | research/experiments/v1_better_registration_audit |
| RAMFD permissive recall mode | Relax RAMFD and water thresholds to maximize recall. | RAMFD + water filter | Implemented | Worked for recall increases but caused severe FP explosion. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/recall_quick_b40_40frames/report.json |
| Adaptive prototype threshold | Per-prototype adaptive distance threshold (percentile + sigma cap). | Water prototype bank | Implemented | Partial impact; insufficient alone under heavy clutter frames. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/ship_water_knn_multimodal_adaptive_120frames/report.json |
| Frame-relative water threshold | Keep candidates near best in-frame water match to cut distance tail. | Water filter stage | Implemented | Worked well to reduce FP while preserving recall in short calibration runs. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/recall_quick_b40_knn_examples_40frames/report.json |
| Multi-sample water KNN training | Use all GT boxes + multi-margin + jitter samples to enrich water bank. | Water KNN bank | Implemented | Strong improvement in FP reduction for hard frames (36/37/38). | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/recall_quick_b40_knn_examples_40frames/report.json |
| Max candidates per frame cap | Hard cap retained candidates after water filter (e.g., cap=3). | Frame-relative threshold | Implemented | Small extra FP reduction with similar recall in 120-frame validation. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/recall_knn_examples_120_cap3_120frames/report.json |
| Trajectory tip-only selection | Apply trajectory filtering and keep only trajectory tip prediction each frame. | Water-filtered candidates | Implemented | Great FP suppression but reduced recall notably in validation. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/traj_tip_wake_40_40frames/report.json |
| Wake triangle suppression | Model boat wake as backward triangle from motion tip and suppress detections in wake region. | Trajectory direction estimate | Implemented | Effective in removing wake-like FP, but combined setting may over-prune true positives. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/traj_tip_wake_40_40frames/report.json |
| Train/eval disjoint protocol | Fit on train only, evaluate final once on disjoint test set (no leakage). | Stable tuned config | Implemented | Protocol succeeded; generalization still weak with current trajectory+wake settings. | research/experiments/v9_mmb_ship_bayesian_validation_tuning/artifacts/debug/final_trainval_tuned_test_100000frames/report.json |
| Dynamic wake geometry (pending) | Adjust wake length/width from recent motion stability and ship size variance. | Trajectory+wake pipeline | Proposed | Not tested yet. | TBD |
| Two-branch decision (pending) | Fuse conservative tip branch with recall branch and resolve by temporal consistency. | Current pipeline outputs | Proposed | Not tested yet. | TBD |
