# Research Hypothesis

## Required questions

1. What problem are we solving?
	We are solving baseline unreliability: the lab needs a reproducible MMB reference on VISO ship.
2. Why does the current SOTA fail?
	The main risk is not a known algorithmic failure but missing local reproducibility, which makes every downstream comparison scientifically weak.
3. Why should this idea work?
	Locking seeds, dataset paths, annotation files, and reporting steps removes major sources of variation before any new method is introduced.
4. Which paper inspired it?
	The baseline source is the MMB method that the project already names as the current baseline in project instructions.
5. What is the novelty?
	The novelty is procedural rather than architectural: this experiment creates the first repository-native benchmark contract for future method comparisons.
6. What is the expected gain?
	The expected gain is 0 AP improvement. The gain is lower variance and higher interpretability of future experiments.
7. What are the risks?
	Missing MMB implementation details, mismatch between original paper preprocessing and local dataset format, and hidden defaults that break reproducibility.
8. What experiments are required?
	Devsample validation, full-dataset validation, five-seed run planning, benchmark collection, and variance estimation.

## Main hypothesis

If MMB is reproduced on VISO ship with fixed preprocessing and seeds `[101, 202, 303, 404, 505]`, then the resulting benchmark variance will be low enough to support future one-hypothesis experiments with a stable baseline contract.

## Expected contribution

This experiment contributes the reference protocol that future experiments must inherit. It makes later positive and negative results scientifically interpretable.
