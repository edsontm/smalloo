# Problem Statement

## Problem

We need a trusted reproduction of the current MMB baseline on VISO ship before testing any new research idea.

Without a locked baseline, any future gain can be caused by dataset wiring differences, preprocessing drift, or seed variance instead of the proposed method.

## Failure of current SOTA

The immediate failure mode is experimental, not algorithmic: the repository does not yet contain a reproducible internal baseline with fixed seeds, fixed split handling, and a single reporting protocol.

For small object detection in satellite video, this is especially risky because small implementation changes in registration, cropping, and thresholding can produce misleading AP shifts.

## Scope boundaries

In scope:

- baseline reproduction on `VISO/coco/ship`
- deterministic seed list
- devsample smoke-test path and full-dataset validation path
- benchmark and reporting protocol

Out of scope:

- any architectural changes to MMB
- new losses, new augmentations, or new preprocessing
- claiming SOTA improvement
