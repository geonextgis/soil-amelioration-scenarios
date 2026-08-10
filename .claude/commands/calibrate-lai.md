---
description: Run the agentic LAI calibration loop for a crop (Claude decides the parameters; no optimizer)
argument-hint: "[crop] [--locations N] [--max-iterations N]"
---

Calibrate **leaf area development** for `$ARGUMENTS` (default crop:
`winter_wheat`) using the agentic calibration workflow in `optimization/`.

The optimized phenology parameters are **frozen** and must not change. There is
no sampler and no optimizer anywhere in this repository — you are the calibration
decision-maker. (The same loop can be driven by a local Ollama model instead:
`/calibrate-local <crop> lai`.)

Delegate the loop to the **lai-calibrator** agent, which owns the protocol and
the parameter-to-symptom mapping. Give it the crop and any `--locations` /
iteration budget from the arguments.

The loop it runs:

```
status  ->  run (baseline, iteration 0)  ->  read diagnostics + figures
        ->  hypothesis  ->  propose one parameter  ->  run  ->  compare with all
        previous iterations  ->  repeat until convergence or the stopping rule
```

Everything goes through `python optimization/calibrate.py`. Nothing edits
`crop.xml` directly, and nothing writes to `simplace/<crop>/data/crop/crop.xml`
— promotion is a separate, explicit step the user takes.

Each SIMPLACE run takes minutes to tens of minutes on SLURM, so run them in the
background and wait for the completion notification rather than polling.

When the loop stops, report back:

- the best iteration, its objective, and the objective at baseline;
- every parameter change from the baseline with the reason it was made;
- the residual error and whether it is attributable to anything still in the
  parameter space;
- where the ledger and figures are
  (`optimization/calibration/<crop>__lai/`);
- the next command:
  `python optimization/calibrate.py handoff --crop <crop>` to carry the
  calibrated LAI parameters into yield calibration, then `/calibrate-yield`.

Do not promote anything to production.
