---
description: Run the phenology calibration loop for a crop (stage 1, from scratch)
argument-hint: <crop> [--iterations N] [--locations N]
---

Run the **phenology** calibration for **$ARGUMENTS** (default crop:
`winter_wheat`). Delegate to the `phenology-calibrator` agent.

This is stage 1 and it runs from scratch. Its result is frozen for the joint
LAI + yield stage, so it has to be finished before stage 2 starts.

## The loop

1. `python optimization/calibrate.py status --crop <crop> --target phenology`
2. Iteration 0 with no `--params` — the baseline objective (mean of the flowering
   and maturity RMSE, in days).
3. Then, each iteration: read the diagnostics, name the mechanism, change one
   thermal-time parameter, state the expected effect, rerun.
4. Diagnose flowering from the **flowering** residual and `TSUM2` from the
   **duration** residual, never from the raw maturity date.
5. Stop when flowering and duration are both within a few days and no structure
   is left in the residual. Residuals under ~2 days are inside the observation
   network's resolution — say so rather than chasing them.

Pre-flight with `--dry-run`; it is free. Use `--locations 40` while iterating and
the configured subset for the run that counts.

## Reporting

- best iteration and objective; flowering / maturity / duration RMSE and bias
- the parameter path from baseline to best, with the reason for each step
- any remaining structure (year, season warmth, latitude) and what it implicates

## Handing over

Report these two commands and let the user run them:

```bash
python optimization/calibrate.py promote --crop <crop> --target phenology --yes
python optimization/calibrate.py handoff --crop <crop>
```

Then `/calibrate-growth <crop>` is stage 2.

## Never

- Edit `crop.xml` by hand. Everything goes through `calibrate.py`.
- Run `promote` or `handoff` yourself.
