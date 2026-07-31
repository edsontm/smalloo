# Discussion

## Why it worked or failed

If the method works, the likely mechanism is cleaner motion alignment reducing background noise and false alarms.

If it fails, either registration was not the main bottleneck or the chosen replacement harmed downstream features.

## Failure cases

- Better alignment but no AP gain
- Better AP but unacceptable runtime or memory cost
- Gains limited to devsample and not reproduced on full VISO

## Lessons learned

If registration is not the bottleneck, future effort should move toward temporal attention, feature alignment, or adaptive motion modeling.
