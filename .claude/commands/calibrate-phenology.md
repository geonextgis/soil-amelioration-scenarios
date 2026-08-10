---
description: Validate the already-optimized phenology for a crop (or recalibrate, if explicitly asked)
argument-hint: <crop> [validate|recalibrate] [--locations N]
---

Run the phenology calibrator for **$ARGUMENTS** (default crop: `winter_wheat`,
default mode: `validate`).

Phenology is already optimized for all five crops here and both later stages
freeze it, so **validation is the default and recalibration must be asked for in
so many words**. If the arguments do not clearly say "recalibrate", validate.

Delegate to the `phenology-calibrator` agent.

## Validating

1. `python optimization/calibrate.py status --crop <crop> --target phenology`
2. Run it unchanged:
   `python optimization/calibrate.py run --crop <crop> --target phenology --reason "validation"`
3. Report flowering RMSE/bias, maturity RMSE/bias, the anthesis-to-maturity
   duration RMSE/bias, the residual structure (year, season warmth, latitude) and
   the attribution verdict.
4. State whether the optimized set still reproduces the observations. A residual
   under ~2 days is inside the observation network's resolution — say so rather
   than implying there is something to fix.

Change nothing. Do not pass `--allow-recalibration`.

## Recalibrating (only if explicitly requested)

Same loop as `/calibrate-lai`: one hypothesis, one parameter, a stated expected
effect, then compare. Add `--allow-recalibration` to every `run`. Diagnose
flowering from the flowering residual and `TSUM2` from the **duration** residual,
never from the raw maturity date.

The pre-change values are preserved automatically in
`optimization/calibration/<crop>__phenology/optimized_baseline.json` and can be
restored with `calibrate.py restore-optimized --crop <crop> --yes`.

## Either way

Never run `calibrate.py promote`. For phenology in particular, promoting would
invalidate the LAI and yield calibrations built on the frozen values — if a
recalibration succeeds, say plainly that both later stages must be redone.

Prefer `--locations 40` or so while iterating; the configured 150 is for the run
that counts.
