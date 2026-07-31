# Research Hypothesis

## Required questions

1. What problem are we solving?
	We are testing whether registration quality is the main bottleneck limiting robust small object detection on VISO ship.
2. Why does the current SOTA fail?
	Weak alignment can inject motion artifacts that downstream modules interpret as targets, especially in cluttered or multi-object scenes.
3. Why should this idea work?
	Better registration should stabilize background modeling and temporal features without forcing detector changes.
4. Which paper inspired it?
	The immediate inspiration is the project's own list of high-priority directions, which explicitly names better registration as a current research direction.
5. What is the novelty?
	The novelty is a clean variable-isolated audit of registration on VISO ship under a frozen benchmark contract.
6. What is the expected gain?
	Expected gain is moderate: improved AP and lower false positives if registration is truly the bottleneck.
7. What are the risks?
	Registration improvements may not transfer to detection quality, or gains may disappear once seed variance is considered.
8. What experiments are required?
	Baseline reproduction, registration-only swap, ablations over registration settings, and statistical comparison against the baseline.

## Main hypothesis

If registration quality is improved while the detector and evaluation protocol remain fixed, then VISO ship will show lower false positives and better temporal consistency, with measurable AP gains over the reproduced MMB baseline.

## Expected contribution

This experiment determines whether registration deserves priority over detector changes in the research roadmap.
