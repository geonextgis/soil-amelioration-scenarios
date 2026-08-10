---
description: Run the agentic yield calibration loop for a crop, starting from the calibrated LAI parameters
argument-hint: "[crop] [--locations N] [--max-iterations N]"
---

Calibrate **yield** for `$ARGUMENTS` (default crop: `winter_wheat`), starting
from the LAI-calibrated parameters.

Preconditions — check these before starting, and stop and say so if unmet:

1. The LAI calibration for this crop has completed iterations and a
   `best_crop.xml` (`python optimization/calibrate.py history --crop <crop> --target lai`).
2. `python optimization/calibrate.py handoff --crop <crop>` has been run, or run
   it now. It seeds the yield run dir from the LAI result and re-anchors the
   freeze snapshot.

Frozen for this target: the optimized phenology **and** the whole LAI parameter
set. There is no optimizer — you are the decision-maker.

Delegate the loop to the **yield-calibrator** agent. Its central task is
attribution: decide from the diagnostics whether the yield error is a **biomass**
problem (`RUETableRUE`, `KDIFTableK`), a **harvest-index** problem
(storage-organ partitioning, `TCNT`, `NMAXSO`), or a **stress-response** problem
(`NLUE`, `DVSNT`/`DVSNLT`) — and to move the parameter that matches, not the one
that happens to reduce the metric.

After any change to a parameter tagged `affects_lai`, it must run
`python optimization/calibrate.py verify-lai --crop <crop>`. A FAIL there means
the yield gain came out of the LAI calibration and must be given back.

`YieldAdjustRatio` is a reporting rescale, not a process parameter. It is the
last resort, for a residual constant bias only.

SIMPLACE runs take minutes to tens of minutes on SLURM — run them in the
background and wait for the notification.

When the loop stops, report:

- best iteration and objective, against the handoff baseline;
- each change, labelled as biomass / harvest-index / stress-response, with its
  justification;
- the final `verify-lai` verdict;
- the residual error and why it is not attributable;
- the promote commands, for the user to run if they choose:
  `python optimization/calibrate.py promote --crop <crop> --target yield` (diff),
  then `--yes` to write.

Do not promote anything yourself.
