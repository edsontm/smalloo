# Discussion

## Why it worked or failed

If reproduction succeeds, the likely reason is that dataset layout, annotations, seeds, and evaluation protocol were all frozen before training code was introduced.

If it fails, the most likely causes are hidden preprocessing assumptions or an incomplete capture of the original MMB evaluation recipe.

## Failure cases

- Missing implementation details from the original MMB source
- Dataset format mismatch between the original pipeline and local COCO conversion
- Large run-to-run variance that prevents fair downstream comparisons

## Lessons learned

Any failure uncovered here should be documented before opening improvement experiments, otherwise later gains will remain uninterpretable.
