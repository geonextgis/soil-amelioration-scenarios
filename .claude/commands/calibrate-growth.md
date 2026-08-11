---
description: Run the joint LAI + yield calibration loop for a crop (stage 2; Claude decides the parameters, no optimizer)
argument-hint: <crop> [--iterations N] [--locations N]
---

Run the **joint LAI + yield** calibration for **$ARGUMENTS** (default crop:
`winter_wheat`). Delegate to the `growth-calibrator` agent.

LAI and yield are calibrated together, in one loop, on one combined objective:
the parameters that set biomass also set leaf area, so calibrating them in
sequence means each stage undoes the other. One iteration runs two simulations
from the same `crop.xml` and scores both.

## Before you start

Stage 1 must be finished and handed over:

```bash
python optimization/calibrate.py status  --crop <crop> --target phenology
python optimization/calibrate.py handoff --crop <crop>          # if not already done
```

`handoff` copies the phenology-calibrated `crop.xml` into the growth run dirs and
re-anchors the freeze on it. If it says the production `crop.xml` does not carry
the calibrated phenology yet, tell the user — `promote --target phenology --yes`
comes first.

## The loop

1. `python optimization/calibrate.py status --crop <crop> --target growth`
2. Iteration 0 with no `--params` — the baseline objective to beat.
3. Then, each iteration: read the diagnostics, name the mechanism, change one
   parameter, and state what you expect to happen to **both** components.
4. Stop when the objective plateaus, when both components are inside the
   observation noise, or when the stopping rule fires.

Pre-flight with `--dry-run` (free, no simulation). Use `--locations 40` while
working out the machinery and the configured subset for the run that counts —
each iteration is two SLURM runs.

## Reporting

- best iteration and objective, plus the LAI and yield losses behind it
- the parameter path from baseline to best, with the reason for each step
- which component is limiting the objective now, and why
- what you believe still limits the fit

## Never

- Edit `crop.xml` or anything under `simplace/<crop>/data/` by hand.
- Run `calibrate.py promote` — that is the user's decision.
