# `optimization/` — SIMPLACE calibration

**There is no optimizer here.** No Optuna, no Bayesian search, no random search,
no sampler of any kind. Every parameter change comes from an agent that reads the
diagnostics, names the mechanism it thinks is responsible, proposes one change
with a stated reason, and is judged against every previous iteration.

Two agent runtimes drive the same machinery, and differ only in which model does
the reasoning:

| | who decides | entry point |
| --- | --- | --- |
| **Local agents** | a local LLM over Ollama, on this machine | `python optimization/agentic.py run --crop winter_wheat --target lai` |
| **Claude Code agents** | Claude, in this session | `/calibrate-lai`, `/calibrate-yield`, `/calibrate-phenology` |

Both go through `calibrate.py`, which is the **only** path to `crop.xml`. It
validates the proposal, verifies the phenology freeze against the written file,
runs SIMPLACE, scores it with the losses in `objectives.py`, and appends an
immutable record to the ledger. A hand-typed proposal gets exactly the same
treatment. That is the whole safety argument: the guarantees live below the
decision-maker, so it does not matter which one you use.

```
optimization/
  config.yaml              MODEL/RUN config: repo root, climate, SLURM, input
                           profiles, per-target run definitions, per-crop facts
  calibration.yaml         CALIBRATION config: parameter meanings and bounds,
                           constraints, freeze groups, stopping rules, llm block
  common.py                staging, crop.xml edits, SIMPLACE, reading outputs
  objectives.py            the three losses (phenology / LAI / yield)
  calib_common.py          space resolution, constraints, freeze guard, ledger
  calib_diagnostics.py     error attribution + figures, per target
  calibrate.py             one iteration — the CLI every agent drives
  agentic.py               the local-agent CLI
  agents/                  the local LLM agents
    llm.py                 Ollama client (stdlib only) + a mock for the tests
    base.py                the loop, the tool surface, context rendering
    phenology.py lai.py yield_.py analyst.py
    prompts/*.md           what each agent knows about its target
  test_calibrate.py        self-test, no cluster needed
  evaluate_run.ipynb       visual + metric evaluation of a finished run
  calibration/<crop>__<target>/   the ledger
```

Phenology has already been calibrated and promoted into each crop's `crop.xml`.
The LAI and yield targets treat those parameters as frozen and verify it on every
write; the phenology target itself is *protected* — see below.

## The order

```
Optimized phenology  ──  /calibrate-phenology validates it, changes nothing
        │                agentic.py validate-phenology does the same, no model needed
        ▼
   LAI calibration      /calibrate-lai      -> calibration/<crop>__lai/best_crop.xml
        │                                      calibrate.py handoff
        ▼
   Yield calibration    /calibrate-yield    -> calibration/<crop>__yield/best_crop.xml
        │                                      calibrate.py verify-lai
        ▼
   calibrate.py promote --yes  ->  simplace/<crop>/data/crop/crop.xml
```

## Local agents — `agentic.py`

Everything stays on the machine: the agent talks to a local Ollama server over
HTTP, gets a JSON proposal back, and hands it to `calibrate.py`.

```bash
python optimization/agentic.py check                                   # is Ollama ready?
python optimization/agentic.py validate-phenology --crop winter_wheat  # no model needed
python optimization/agentic.py propose --crop winter_wheat --target lai   # one decision, no run
python optimization/agentic.py run     --crop winter_wheat --target lai --iterations 8
python optimization/agentic.py step    --crop winter_wheat --target lai   # exactly one iteration
python optimization/agentic.py review  --crop winter_wheat --target lai   # read-only analysis
```

`check` is the first thing to run and the only one that fails informatively on a
machine without Ollama:

```
provider   ollama
host       http://localhost:11434
models     UNREACHABLE — cannot reach Ollama at http://localhost:11434
```

Configuration lives in `calibration.yaml → llm`: the host (overridden by
`OLLAMA_HOST`), the context size, the request timeout, how many repair attempts a
malformed proposal gets, and the model + temperature per agent. Any
instruction-tuned local model works; the defaults assume a 24–48 GB GPU and
should drop to `:7b` / `:8b` variants on smaller hardware.

**The client uses the standard library only** — no `ollama` package, no
`requests`. The cluster nodes have no outbound network, so a dependency that has
to be installed is a dependency that will not be there.

### What the model actually sees

Not a search space. The context is the current values with their bounds, the
biological `meaning` of every parameter it may touch, the frozen list in full,
the constraints, the diagnostics of the last completed iteration, and every
previous iteration with its reason and outcome. It replies with a small JSON
object naming one parameter, why, and what it expects to happen. Roughly 2.5k
tokens for an LAI iteration.

### The repair loop

A proposal is **pre-flighted** before any cluster time is spent on it —
`calibrate.py run --dry-run` validates bounds, table shape, partitioning closure
and blast radius, and writes nothing. If it fails, the exact violation text goes
back to the model and it tries again (`max_repair_attempts`, default 3). This is
what makes a 14B model usable here: a violated constraint is a far better
correction signal than any amount of prompt tuning, and it costs nothing.

If no valid proposal survives the repair attempts, the iteration is abandoned
rather than forced.

## Phenology is protected

The optimized phenology values are ground truth for both later stages, so the
phenology target refuses to write:

```bash
# validate — always allowed, changes nothing
python optimization/calibrate.py run --crop winter_wheat --target phenology \
    --reason "validation"

# recalibrate — only with an explicit flag
python optimization/calibrate.py run --crop winter_wheat --target phenology \
    --allow-recalibration --params '{"TSUM1": 1220}' --reason "..."
```

Before the first iteration the current values are preserved to
`calibration/<crop>__phenology/optimized_baseline.json` and diffed on every
iteration afterwards. `calibrate.py restore-optimized --crop <crop> --yes` puts
them back. The ledger is never rewritten, so the iterations that moved them stay
on record.

The agent exists for the cases that will come: a sixth crop, an extended
observation series, or simply confirming that the optimized set still reproduces
the observations after something else changed.

## What one iteration actually does

```
stage -> validate -> write crop.xml -> re-verify the freeze -> clear out/
      -> SIMPLACE -> read out/ -> loss -> diagnostics -> figures -> ledger
```

Calibration runs are isolated in `simplace/<crop>/runs_optim/calib_<target>/`,
built the same way `orchestration/generate.py` builds scenario runs (and reusing
its `project.proj.xml` templating):

- `data/crop/` is a **real copy** — `crop.xml` is rewritten every iteration, so it
  must not alias the production file. `simplace/<crop>/data/crop/crop.xml` is
  never modified except by `calibrate.py promote`.
- `data/soil/soil.csv`, `data/management/location.csv`,
  `data/management/fertilizer_<crop>.csv` are copies of the staged source tables
  (see *input profiles* below).
- `solution/`, `data/{slim,soilcnp,co2}` and the static management XMLs are
  symlinks to the crop's shared inputs.
- `project/project.csv` is generated: subset per config, then sorted
  location-contiguous and re-numbered.
- `config.yaml` is the `cluster:` block handed to `simplace_runner_cluster.py`.

Three things in there are load-bearing rather than cosmetic:

**Location-contiguous sorting.** SIMPLACE writes one output file per location
into a shared directory, so the cluster runner may only split work on location
boundaries. The baseline `project_<crop>.csv` appends each location's recent
years at the file tail, so an unsorted table would let two SLURM tasks handle the
same location and silently clobber each other's output.

**Clearing `out/` before each run.** Per-location output files persist, so a
location dropped by a failed task would otherwise contribute the *previous*
iteration's values to this iteration's loss.

**Waiting out the filesystem cache.** `out/<kind>/` is deleted locally and
recreated remotely by the compute nodes. On BeeGFS the login node can serve a
stale listing for a few seconds after `squeue` empties, so a single glob taken
the instant the jobs finish reports zero files for a run that in fact succeeded.
`common.await_outputs` polls with a forced re-read. Only fast runs land inside
that window, which is why it looks like it never mattered.

## Input profiles

`config.yaml → input_profiles` selects which per-location tables get staged
under the canonical names the solution reads:

- `production` — `soil.csv`, `location.csv`, `fertilizer_<crop>.csv`. Used by
  phenology and yield.
- `lai` — the `*_<crop>_LAI.csv` variants. The LAI calibration project table has
  its own point set *and its own weather grid*, distinct from the baseline table;
  only ~42% of its points exist in the production tables, so it needs the
  matching soil/location/fertilizer rows. Crops without `*_LAI` variants fall
  back to `production` with a printed note.

This is the one place the deprecated-looking `*_LAI.csv` inputs are still
correct: they are the LAI calibration point set, staged into an isolated run dir
under the canonical filenames. No solution is pointed at a `*_LAI.csv` path.

## Subsets

`calibration.yaml → targets.<target>.subset` trims the project table so an
iteration is cheap enough to reason about. (`config.yaml` carries a larger
default for the same targets; the calibration config wins.)

| target | project table | subset |
| --- | --- | --- |
| phenology | `project_<crop>.csv` | 150 locations, 1995–2022 |
| lai | `project_<crop>_LAI.csv` | 150 locations (table is already one season per point) |
| yield | `project_<crop>.csv` | 150 locations, 1995–2023 |

Locations are picked evenly across the sorted ID range, so the geographic spread
of the full set is preserved. `--locations N` overrides it per invocation — use
something small (30–50) while iterating on the machinery, and the configured
value for the run that counts. Set `n_locations: null` to calibrate on everything.

## The losses — `objectives.py`

| target | loss |
| --- | --- |
| phenology | mean of the flowering and maturity RMSE, in days (autumn bolting excluded) |
| lai | RMSE over DVS-bin × year means of observed vs simulated LAI |
| yield | mean of the yearly-mean RMSE and the state-mean RMSE, in t/ha |

Each is `process_result(run_spec) -> DataFrame` plus `loss_fn(frame) -> (loss,
metrics)` and nothing else. They are pure scoring functions; nothing in them
decides anything. Keeping them in one module is what lets the diagnostics, the
notebook and the ledger all agree on what a number means.

## Evaluating a run — `evaluate_run.ipynb`

Set `CROP` in the one configuration cell and run all cells. The notebook compares
simulated against observed **phenology, LAI and yield**, with comparison plots and
RMSE / MAE / R² / bias (plus nRMSE% and Nash–Sutcliffe efficiency) for each.

It finds the outputs itself. `common.discover_runs()` lists every directory under
`runs/` and `runs_optim/` that holds results; by default each variable is scored
against its own calibration run when that has output, and otherwise against the
most recently written scenario run. Set `SOURCE` to a label (`"DWD__S1"`,
`"optim:calib_lai"`) to pin all three to one run.

Loading reuses the same `process_result` functions the calibration uses, so the
notebook's numbers are the numbers the agents were scored on — the loss is not
re-implemented. Each variable's section is independent: a run with no LAI output
reports that and the other two sections still work.

Two things to know:

- **LAI from a scenario run is not like-for-like.** The LAI observations belong to
  their own point set and weather grid, so a scenario run matches only a fraction of
  them. The notebook warns when fewer than half the simulated locations carry
  observations; use the LAI calibration run for a real LAI evaluation.
- **Daily output is sampled.** A full run's daily files are tens of millions of rows;
  `MAX_LAI_LOCATIONS` (default 300) reads an evenly spaced subset. Raise it for a
  final evaluation, and note the LAI metrics move slightly with it.

---

# One iteration — `calibrate.py`

This is the layer every agent goes through, local or Claude, and the layer you
use directly when you want to make a change by hand. It reads a proposal, decides
whether it is allowed, runs it, scores it and records it. It contains no
reasoning of its own.

## The loop

```bash
python optimization/calibrate.py status --crop winter_wheat --target lai
# -> current values, per-crop bounds, the frozen set, the full history

python optimization/calibrate.py run --crop winter_wheat --target lai
# -> iteration 0: the baseline objective of the unchanged, phenology-optimized crop.xml

python optimization/calibrate.py run --crop winter_wheat --target lai \
    --params '{"SLATableSLA": {"3": 0.0138}}' \
    --reason      "anthesis DVS bin is 0.56 LAI short; the other bins are unbiased" \
    --hypothesis  "SLA at DVS 1.0 is too low, so the peak canopy is thin" \
    --reasoning   "…what was ruled out and why…" \
    --expected-effect "bin 1.0-1.25 bias toward zero; earlier bins unchanged"
```

One `run` = stage → validate → write `crop.xml` → re-verify the freeze →
SIMPLACE → score → diagnose → plot → append to the ledger. It ends with a
machine-readable `JSON {...}` line carrying the objective, whether it improved,
and whether the stopping rule has fired.

Other subcommands: `diagnose` (metrics and figures without simulating),
`history`, `show --iteration N`, `handoff`, `verify-lai`, `promote`,
`restore-optimized`.

Useful flags: `--locations N` (smaller subset → faster iteration), `--dry-run`
(validate a proposal and report the verdict as JSON; writes nothing at all, not
even to `rejected.jsonl` — this is the agents' pre-flight), `--skip-run`
(re-score what is in `out/`), `--force` (apply despite constraint violations;
recorded in the ledger), `--allow-recalibration` (permit a write to the protected
phenology target).

## Parameters — declared, not hard-coded

`calibration.yaml` declares each calibratable parameter with the biological
`meaning` the agent reasons from, what it `controls`, and how its bounds are
derived:

```yaml
RGRLAI:
  kind: scalar
  mode: relative          # bounds = current value x factor, then clipped
  factor: [0.5, 2.0]
  clip:   [0.0005, 0.06]  # hard physical limits
  controls: [early_growth, rise_rate]
  meaning: >-
    Maximum relative growth rate of LAI during the exponential phase … Raise it
    if simulated LAI lags early but the peak is right; lower it if the canopy
    closes too fast.
```

### Turning a parameter off

```yaml
      YieldAdjustRatio:
        enabled: false
        disabled_reason: >-
          declared in solution.sol.xml but consumed by no simcomponent —
          changing it cannot change any simulated value
        kind: scalar
        ...
```

`enabled: false` removes the parameter from the space entirely: a proposal naming
it is refused at parse time, it is never offered to an agent, and
`calibrate.py status` lists it with the reason. Deleting the block has the same
calibration effect and loses the finding — which is why the two inert parameters
are disabled rather than removed.

Three ways to stop a parameter moving, in increasing strength:

| Goal | How | Enforcement |
|---|---|---|
| Not calibrated in this study | `enabled: false` | refused at parse: *not a calibratable parameter* |
| Visible but immovable | `mode: absolute` with `low: X, high: X` | `bounds` violation |
| Provably never written | add to `frozen_groups`, list under the target's `frozen:` | `frozen` violation, plus the XML re-read after every iteration |

Bounds are **relative to each crop's own `crop.xml`**, so one declaration fits
all five crops even though their SLA tables have 8 (wheat), 5 (rapeseed),
3 (potato) nodes. Table parameters expand to one bound per element actually
present. A value that is structurally zero (leaf allocation after anthesis) stays
pinned at zero.

## What cannot happen

| Guard | Mechanism |
| --- | --- |
| Phenology is changed | Snapshot of the frozen set taken on first use; **re-read from the written XML after every change**. A drift aborts the iteration and rolls `crop.xml` back. |
| A parameter leaves its range | `check_within_bounds` on the flattened proposal, before the model runs |
| A biologically incoherent profile | Table shape rules: SLA smoothness and decline to maturity, leaf-death rate monotone in temperature, RUE non-increasing after anthesis, storage organs non-decreasing |
| Above-ground allocation stops summing to 1 | Leaves + stems + storage interpolated onto the union DVS grid and compared against the pre-change profile |
| Too many things change at once | ≤ 3 parameters and ≤ 4 individual values per iteration; ≤ 50 % move per value |
| The LAI calibration is undone by yield work | The whole LAI set is frozen for the yield target; parameters that still perturb LAI are tagged `affects_lai` and gated by `verify-lai` |
| The optimized phenology is overwritten | The phenology target is `protected:` — a write needs `--allow-recalibration`, and the pre-change values are preserved to `optimized_baseline.json` before the first iteration |
| History is lost | `ledger.jsonl` is append-only; every iteration keeps its own directory with `crop.xml`, the joined obs/sim pairs, metrics and figures |

A rejected proposal is written to `rejected.jsonl` and the model is not run. A
`--dry-run` pre-flight is not a proposal and is not recorded anywhere.

## The ledger

```
optimization/calibration/<crop>__<target>/
  state.json              pointer: iterations, best, frozen digest
  ledger.jsonl            append-only, one line per iteration
  rejected.jsonl          proposals that failed validation
  frozen_snapshot.json    the frozen parameters + digest
  optimized_baseline.json (protected targets) the already-optimized values,
                          preserved before the first iteration and restorable
  baseline_crop.xml       the starting point
  best_crop.xml           lowest objective so far
  current_crop.xml        last good state (used to roll back a failed iteration)
  provenance.json         (yield only) which LAI result it was seeded from
  lai_regression.jsonl    (yield only) verify-lai results
  history.png             objective per iteration, annotated with what changed
  iterations/iter_NNN/
      crop.xml            the exact parameters that produced this result
      proposal.json       what changed, the reason, hypothesis, full reasoning
      metrics.json        objective + loss metrics + the diagnostic bundle
      pairs.csv.gz        the joined observed/simulated frame — re-analysable
      season_shape.csv    per location-season curve features (LAI)
      diagnostics/*.png
      simplace.log
```

Each record carries: iteration, crop, target, timestamp, scope (project table,
rows, locations, subset, climate, years), the **full** parameter set, what
changed, the previous values, the objective, the simulated and observed metrics,
the diagnostics, the reason / hypothesis / agent reasoning / expected effect,
whether it improved, whether it is the new best, the delta against the previous
best, the frozen digest and whether the freeze held, any constraint violations,
the figures, and the elapsed time.

## Diagnostics — why not just RMSE

The objective stays the existing loss so the numbers remain comparable, but the
agent diagnoses on the decomposition:

**LAI** — bias per DVS bin (element *i* of the bin table maps onto element *i* of
`SLATableSLA`, which is what makes the bias directly actionable), then per
location-season: peak LAI, peak timing, peak stage, early-canopy level, rise
rate, plateau duration (days at ≥ 80 % of peak), decline rate — each as observed
vs simulated with the median difference and its IQR. Plus the residual structure:
by DVS, by month, by location, and the share of seasons whose residual never
changes sign. Observed and simulated statistics are always computed **on the
matched dates only**, so a sparse 8-day GLASS retrieval is never compared against
a daily statistic it cannot support.

**Yield** — the error split along `yield = AGBiomass x harvest index`: what the
model produced for each, what each would have to be to match the observations,
and whether those sit inside the agronomic ranges in `calibration.yaml →
crop_reference`. That is what separates "the crop did not make enough biomass"
(RUE, KDIF) from "it made the biomass and did not put it in the grain"
(partitioning, translocation). Plus by-year and by-state residuals, the residual
year trend, and correlations against `maxLAI`, `AGBiomass`, `TRANRF` and `NNI`.

**Phenology** — the flowering residual and the anthesis-to-maturity **duration**
residual, separately. That split is the whole diagnosis: maturity is dated *from*
anthesis, so a `TSUM1` error propagates into the raw maturity error unchanged and
diagnosing on it moves `TSUM2` to fix a `TSUM1` problem. The residual is then
regressed against season warmth (a slope means the temperature *response* is
wrong — `TEFFMX`, `TsumIncrementTableRate`) and against latitude (a gradient is
the photoperiod signature — `PhotoperiodTableFactor`).

## The agents

**Local** — `optimization/agents/`, run by `agentic.py`:

| agent | job |
| --- | --- |
| `PhenologyAgent` | validates the optimized set; recalibrates only in `--mode recalibrate` |
| `LAIAgent` | stage 1 — canopy development against GLASS-LAI |
| `YieldAgent` | stage 2 — district yields, LAI frozen; runs `verify-lai` afterwards |
| `AnalystAgent` | read-only review of a calibration in progress; proposes nothing |

Each is a thin class plus a prompt in `agents/prompts/`. The prompt is where the
target-specific knowledge lives — the symptom-to-parameter table, the rules that
cannot be broken, and when to stop. Adding an agent is a subclass and a prompt.

**Claude Code** — `.claude/agents/`: `crop-model-analyst` (read-only: what drives
what), `phenology-calibrator`, `lai-calibrator`, `yield-calibrator`,
`calibration-diagnostics`.

`.claude/commands/` — `/calibrate-phenology`, `/calibrate-lai`, `/calibrate-yield`,
`/calibration-status`, `/calibrate-local`.

## Self-test

```bash
python optimization/test_calibrate.py     # 50 checks, no cluster required
```

Covers config loading, per-crop space resolution for all three targets, every
constraint rule, the freeze guard, the phenology protection, proposal
normalisation, the ledger and stopping rules, and the diagnostics against
synthetic data with known level, phase and duration errors.

It also covers the local-agent layer without needing Ollama: the JSON extractor
against prose and fenced replies, the Ollama client against a **stand-in HTTP
server that implements `/api/tags` and `/api/chat` exactly as Ollama does** (which
is what verifies the wire format on a machine with no Ollama installed), and the
full agent loop — context rendering, reply parsing, the repair round-trip, the
freeze abort, self-stopping — against a scripted mock backend.

The mock is a test double, never a fallback. A missing Ollama at runtime is
reported as an error; nothing silently substitutes for the model.

---

## Adding a target or a crop

A target is:

1. a run definition in `config.yaml` (project table, input profile, `exp_name`),
2. a `process_result` / `loss_fn` pair in `objectives.py`, registered in `TARGETS`,
3. a `calibration.yaml` block: parameters with their `meaning`, constraints,
   freeze groups, stopping rules,
4. a branch in `calibrate.py::_evaluate` for its diagnostics,
5. a `CalibrationAgent` subclass and a prompt, if it should have a local agent.

Everything else — staging, XML editing, the freeze guard, validation, the ledger,
stopping — is inherited.

Crops need a `crops:` entry whose `crop_name` matches
`<parameter id="CropName">` in that crop's `data/crop/crop.xml`, and
`dm_fraction` if observed yield is not on a dry-matter basis (potato: observed
is fresh tuber weight, so `0.21`).
