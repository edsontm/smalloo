# Problem Statement

## Problem

Satellite video detection is sensitive to misregistration. If frame alignment is weak, background subtraction and motion-driven cues become noisy, which can inflate false positives and harm tiny object localization.

## Failure of current SOTA

Current pipelines often mix registration changes with detector changes, making it difficult to isolate whether motion alignment is the real bottleneck.

On VISO ship, the larger `1345x451` images also contain more multi-object scenes, increasing the chance that registration noise cascades into unstable detections.

## Scope boundaries

In scope:

- replacing or strengthening registration only
- keeping detector, splits, and evaluation fixed
- measuring AP, temporal consistency, false positives, and registration error

Out of scope:

- detector architecture changes
- loss-function changes
- augmentation changes unrelated to registration
