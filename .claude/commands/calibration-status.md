---
description: Summarise calibration state — parameters, bounds, frozen set, history and convergence — for a crop
argument-hint: "[crop] [lai|yield|both]"
---

Report the calibration state for `$ARGUMENTS` (default: `winter_wheat both`).

Gather:

```bash
python optimization/calibrate.py status  --crop <crop> --target lai
python optimization/calibrate.py history --crop <crop> --target lai
python optimization/calibrate.py status  --crop <crop> --target yield
python optimization/calibrate.py history --crop <crop> --target yield
```

Then summarise:

- **Frozen phenology** — intact or drifted, and the digest. A drift is a serious
  finding: report it prominently and do not paper over it.
- **Progress** — baseline objective, best objective, iterations completed,
  iterations since the last improvement, and what the stopping rule says.
- **What changed** — the parameter path from baseline to best, with the recorded
  reason for each step.
- **Where it stands** — still improving, converged, or oscillating.
- **Next action** — the specific command.

For a deeper read of *why* the current error looks the way it does, delegate to
the **calibration-diagnostics** agent rather than re-deriving it here.

Note which crops have no calibration yet (no
`optimization/calibration/<crop>__<target>/` directory) so the picture is
complete.
