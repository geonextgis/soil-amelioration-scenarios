---
description: Summarise calibration state — parameters, bounds, frozen set, history and convergence — for a crop
argument-hint: "[crop] [phenology|growth|both]"
---

Report the calibration state for `$ARGUMENTS` (default: `winter_wheat both`).

Gather:

```bash
python optimization/calibrate.py status  --crop <crop> --target phenology
python optimization/calibrate.py history --crop <crop> --target phenology
python optimization/calibrate.py status  --crop <crop> --target growth
python optimization/calibrate.py history --crop <crop> --target growth
```

Then summarise:

- **Where the workflow stands** — stage 1 in progress, promoted, handed off, or
  stage 2 running. `optimization/calibration/<crop>/growth/provenance.json` says
  which phenology result stage 2 was seeded from.
- **Frozen set** — intact or drifted, and the digest. A drift is a serious
  finding: report it prominently and do not paper over it.
- **Progress** — baseline objective, best objective, iterations completed,
  iterations since the last improvement, and what the stopping rule says. For the
  growth stage report the **LAI and yield losses separately** as well as the
  combined objective; a flat combined objective can hide one component improving
  while the other degrades.
- **What changed** — the parameter path from baseline to best, with the recorded
  reason for each step.
- **Next action** — the specific command.

For a deeper read of *why* the current error looks the way it does, delegate to
the **calibration-diagnostics** agent rather than re-deriving it here.

Note which crops have no calibration yet (no
`optimization/calibration/<crop>/<stage>/` directory) so the picture is complete.
