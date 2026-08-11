---
description: Run the calibration loop with local Ollama agents instead of Claude
argument-hint: <crop> <phenology|growth> [--iterations N] [--locations N]
---

Drive the calibration with the **local LLM agents** in `optimization/agents/`,
which run entirely on this machine against a local Ollama server. Claude sets it
up and reports; the local model makes the parameter decisions.

Arguments: **$ARGUMENTS** (defaults: crop `winter_wheat`, target `growth`).

## Steps

1. **Check the setup first** — this fails on a machine without Ollama, and the
   error is the useful part:

   ```bash
   python optimization/agentic.py check
   ```

   If it reports the server unreachable or a model not pulled, show the user the
   output and stop. Do not fall back to calibrating it yourself unless they ask —
   they chose the local path deliberately.

2. **See what the model would do**, before spending cluster time on it:

   ```bash
   python optimization/agentic.py propose --crop <crop> --target <target>
   ```

   This asks for one decision and pre-flights it against the constraints. Nothing
   runs, nothing is written. Report the proposed change, the stated reason, and
   whether the pre-flight accepted it.

3. **Run the loop**:

   ```bash
   python optimization/agentic.py run --crop <crop> --target <target> --iterations 8
   ```

   Each iteration: the local model reads the parameters, bounds, diagnostics and
   full history, proposes one change, it is validated, SIMPLACE runs, the result
   is scored and appended to the same ledger `/calibrate-growth` writes to.
   A `growth` iteration runs SIMPLACE twice — once per view — and is scored on
   the combined LAI + yield objective.

4. **Review it** — a second local model, read-only, over the whole history:

   ```bash
   python optimization/agentic.py review --crop <crop> --target <target>
   ```

## Notes

- The guarantees are identical to the Claude path: the frozen phenology snapshot,
  the constraint engine, the append-only ledger and the isolated run dir all live
  in `calibrate.py`, which is the only way any agent reaches `crop.xml`.
- Order is `--target phenology`, then `calibrate.py promote --target phenology
  --yes` and `calibrate.py handoff`, then `--target growth`.
- Never run `calibrate.py promote`. That stays a human decision.
- A 14B model on a cold GPU can take a minute per decision, which is nothing next
  to the SLURM run. If the model is producing unusable JSON repeatedly, report the
  repair failures rather than switching to a bigger model on your own.
