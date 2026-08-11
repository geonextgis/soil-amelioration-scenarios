# `optimization/` — SIMPLACE calibration

**There is no optimizer here.** No Optuna, no Bayesian search, no random search,
no sampler of any kind. Every parameter change comes from an agent that reads the
diagnostics, names the mechanism it thinks is responsible, proposes one change
with a stated reason, and is judged against every previous iteration.

The workflow is **two stages**:

```
  stage 1                          stage 2
  ┌───────────────────┐            ┌──────────────────────────────────┐
  │ phenology         │  handoff   │ growth = LAI + yield, JOINTLY    │
  │ from scratch      │ ─────────► │ phenology frozen                 │
  │ TSUM1/TSUM2/…     │            │ one crop.xml, two simulations,   │
  └───────────────────┘            │ one combined objective           │
                                   └──────────────────────────────────┘
```

**Why LAI and yield are calibrated together.** Radiation use efficiency, light
interception and dry-matter partitioning set biomass *and* leaf area. Calibrated
in sequence, every accepted yield change silently rewrites the canopy — and
freezing the canopy to prevent that removes most of the yield levers with it. So
one iteration runs the model twice from the same `crop.xml` (the GLASS-LAI point
set and the district yield point set), scores both, and combines them into one
number the agent is judged on.

**Why phenology comes first and alone.** Everything downstream is dated off DVS.
A canopy or yield parameter tuned against a wrong development clock is tuned
against noise, so the clock is settled first and then frozen.

Two agent runtimes drive the same machinery, and differ only in which model does
the reasoning:

| | who decides | entry point |
| --- | --- | --- |
| **Local agents** | a local LLM over Ollama, on this machine | `python optimization/agentic.py run --crop winter_wheat --target growth` |
| **Claude Code agents** | Claude, in this session | `/calibrate-phenology`, `/calibrate-growth` |

Both go through `calibrate.py`, which is the **only** path to `crop.xml`. It
validates the proposal, verifies the freeze against the written file, runs
SIMPLACE, scores it with the losses in `objectives.py`, and appends an immutable
record to the ledger. A hand-typed proposal gets exactly the same treatment. That
is the whole safety argument: the guarantees live below the decision-maker, so it
does not matter which one you use.

```
optimization/
  config.yaml              MODEL/RUN config: repo root, climate, SLURM, input
                           profiles, the views of each stage, per-crop facts
  calibration.yaml         CALIBRATION config: parameter meanings and bounds,
                           objective weights, constraints, freeze groups,
                           stopping rules, llm block
  common.py                staging, crop.xml edits, SIMPLACE, reading outputs
  objectives.py            the three per-view losses + how they are combined
  evaluation.py            scoring one iteration: per view, then the objective
  calib_common.py          space resolution, constraints, freeze guard, ledger
  calib_diagnostics.py     error attribution + figures, per view
  calibrate.py             one iteration — the CLI every agent drives
  agentic.py               the local-agent CLI
  agents/                  the local LLM agents
    llm.py                 Ollama client (stdlib only) + a mock for the tests
    base.py                the loop, the tool surface, context rendering
    phenology.py growth.py analyst.py
    prompts/*.md           what each agent knows about its stage
  test_calibrate.py        self-test, no cluster needed
  evaluate_run.ipynb       visual + metric evaluation of a finished run
  calibration/<crop>/<stage>/     the ledger
```

---

# Running the workflow

Everything below is for one crop; substitute `--crop winter_rapeseed`,
`spring_barley`, `potato` or `maize` as needed.

## Stage 1 — phenology

```bash
# 1. where it stands: current values, per-crop bounds, the frozen set, history
python optimization/calibrate.py status --crop winter_wheat --target phenology

# 2. iteration 0 — the baseline objective to beat (no --params)
python optimization/calibrate.py run --crop winter_wheat --target phenology \
    --reason "baseline: the current thermal-time set, unchanged"

# 3. one change per iteration, with the reasoning recorded
python optimization/calibrate.py run --crop winter_wheat --target phenology \
    --params '{"TSUM1": 1220}' \
    --reason      "flowering is 6.2 d late with a large spread across years" \
    --hypothesis  "the emergence-to-anthesis thermal requirement is too high" \
    --reasoning   "ruled out TSUMEM: the bias IQR is 11 d, not a constant shift" \
    --expected-effect "flowering bias toward zero; duration unchanged"
```

Repeat step 3 until the stopping rule fires (it is reported at the end of every
iteration). The objective is the mean of the flowering and maturity RMSE, **in
days**; below ~6 days it is inside the resolution of the DWD phenology network.

Diagnose flowering from the **flowering** residual and `TSUM2` from the
**duration** residual — never from the raw maturity date, which carries the
flowering error unchanged.

## Between the stages

```bash
# 4. write the calibrated phenology into the production crop.xml
python optimization/calibrate.py promote --crop winter_wheat --target phenology --yes

# 5. seed stage 2 from it
python optimization/calibrate.py handoff --crop winter_wheat
```

`promote` shows the diff and does nothing without `--yes`; it keeps a
`crop.xml.pre_phenology_calibration` backup. `handoff` copies the phenology
`best_crop.xml` into both growth run dirs, re-anchors the freeze snapshot on it,
and records `provenance.json` saying which iteration it came from.

Do step 4 before step 5's stage-2 result is promoted: `promote --target growth`
refuses while the production file still carries an uncalibrated phenology (it
would mean promoting a candidate whose frozen parameters differ from production).

## Stage 2 — growth (LAI + yield, jointly)

```bash
# 6. state, including both views and the objective weights
python optimization/calibrate.py status --crop winter_wheat --target growth

# 7. iteration 0 — baseline
python optimization/calibrate.py run --crop winter_wheat --target growth \
    --reason "baseline from the phenology handoff"

# 8. one change per iteration
python optimization/calibrate.py run --crop winter_wheat --target growth \
    --params '{"SLATableSLA": {"3": 0.0138}}' \
    --reason      "anthesis DVS bin is 0.56 LAI short; the other bins are unbiased" \
    --hypothesis  "SLA at DVS 1.0 is too low, so the peak canopy is thin" \
    --reasoning   "ruled out RUE: biomass and HI are both inside the plausible range" \
    --expected-effect "LAI RMSE 0.50 -> ~0.45; yield unchanged to slightly up"

# 9. promote the result
python optimization/calibrate.py promote --crop winter_wheat --target growth --yes
```

Step 8 runs SIMPLACE **twice** — once per view. Both simulations use the same
`crop.xml`; `calibrate.py` mirrors the primary view's file into the other before
running, so the two components can never describe different parameter sets.

## Agent-driven instead of by hand

```bash
# Claude Code
/calibrate-phenology winter_wheat
/calibrate-growth    winter_wheat
/calibration-status  winter_wheat both

# local Ollama agents (nothing leaves the machine)
python optimization/agentic.py check                                        # is Ollama ready?
python optimization/agentic.py run     --crop winter_wheat --target phenology
python optimization/agentic.py run     --crop winter_wheat --target growth --iterations 8
python optimization/agentic.py step    --crop winter_wheat --target growth  # exactly one iteration
python optimization/agentic.py propose --crop winter_wheat --target growth  # one decision, no run
python optimization/agentic.py review  --crop winter_wheat --target growth  # read-only analysis
```

The agents run the same `calibrate.py` commands shown above. `promote` and
`handoff` are deliberately left to a person.

## Useful flags

| Flag | What it does |
| --- | --- |
| `--locations N` | override the subset size — use 30–50 while working out a change, the configured value for the run that counts |
| `--dry-run` | validate a proposal and print the verdict as JSON; writes nothing at all, not even to `rejected.jsonl`. This is the agents' pre-flight, and it is free |
| `--skip-run` | re-score whatever is already in `out/` instead of simulating |
| `--rebuild` | rebuild the run dirs from scratch (warns if the ledger is non-empty) |
| `--force` | apply despite constraint violations; recorded in the ledger |
| `--device local` | use `simplace_runner.py` instead of the SLURM driver |

Other subcommands: `diagnose` (metrics and figures without simulating),
`history`, `show --iteration N`, `restore-baseline` (put the stage's starting
`crop.xml` back into the run dirs; the ledger is untouched).

---

# How it works

## Stages, views and the combined objective

A **stage** (`phenology`, `growth`) is what you calibrate. A **view** is one
model run scored against one observation set. `config.yaml` declares the views:

```yaml
targets:
  growth:
    run_subdir: calib_growth
    views:
      lai:
        exp_name: CALIB_GROWTH_LAI
        project_csv: project/project_{crop}_LAI.csv
        inputs: lai
        subset: {n_locations: 400}
      yield:
        exp_name: CALIB_GROWTH_YIELD
        project_csv: project/project_{crop}.csv
        inputs: production
        subset: {n_locations: 400, year_start: 2000, year_end: 2023}
```

and `calibration.yaml` weighs them:

```yaml
targets:
  growth:
    objective:
      name: joint_lai_yield
      components:
        lai:   {weight: 0.5, scale: 0.15}   # DVS-binned LAI RMSE (m2/m2)
        yield: {weight: 0.5, scale: 0.50}   # mean of temporal + spatial RMSE (t/ha)
```

    objective = Σ weightᵢ · (lossᵢ / scaleᵢ) / Σ weightᵢ

`scale` is the loss at which a component counts as **1.0** — its own target. That
makes components with different units commensurable and gives the objective a
fixed meaning: **1.0 = both components at target on average**. Each component's
loss, scaled value and share of the objective is printed every iteration and
stored in the ledger, so a flat objective can never hide one component improving
while the other degrades.

The phenology stage has one view and `scale: 1.0`, so its objective is still the
raw RMSE in days.

Every view must have a weight and every weight must name a view — a mismatch is
refused at load time, because a view simulated but not scored (or scored but not
simulated) is silently wrong rather than loudly wrong.

## What one iteration actually does

```
stage every view -> validate -> write crop.xml -> re-verify the freeze
                 -> mirror crop.xml into the other views
                 -> for each view: clear out/, run SIMPLACE, read out/, loss, diagnostics, figures
                 -> combine -> ledger
```

Calibration runs are isolated in
`simplace/<crop>/runs_optim/<run_subdir>/<view>/`, built the same way
`orchestration/generate.py` builds scenario runs (and reusing its
`project.proj.xml` templating):

- `data/crop/` is a **real copy** — `crop.xml` is rewritten every iteration, so it
  must not alias the production file. `simplace/<crop>/data/crop/crop.xml` is
  never modified except by `calibrate.py promote`.
- `data/soil/soil.csv`, `data/management/location.csv`,
  `data/management/fertilizer_<crop>.csv` are copies of the staged source tables
  (see *input profiles* below).
- `solution/`, `data/{slim,soilcnp}` and the static management XMLs are symlinks
  to the crop's shared inputs; `data/co2/co2.csv` is staged.
- `project/project.csv` is generated: subset per config, then sorted
  location-contiguous and re-numbered.
- `config.yaml` is the `cluster:` block handed to `simplace_runner_cluster.py`.

Four things in there are load-bearing rather than cosmetic:

**One crop.xml per stage.** The first view owns the canonical file;
`calib_common.sync_crop_xml` mirrors it into the others before every simulation.
Without it the two components of the joint objective would describe different
parameter sets.

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

`config.yaml → input_profiles` selects which per-location tables get staged under
the canonical names the solution reads:

- `production` — `soil.csv`, `location.csv`, `fertilizer_<crop>.csv`. Used by the
  phenology stage and by the yield view.
- `lai` — the `*_<crop>_LAI.csv` variants. The LAI project table has its own point
  set *and its own weather grid*, distinct from the baseline table; only ~42 % of
  its points exist in the production tables, so it needs the matching
  soil/location/fertilizer rows. Crops without `*_LAI` variants fall back to
  `production` with a printed note.

This is also why the joint stage needs two runs rather than one: the two
observation sets do not live on the same points.

## Subsets

`config.yaml → targets.<target>.views.<view>.subset` trims the project table so an
iteration is cheap enough to reason about.

| stage | view | project table | subset |
| --- | --- | --- | --- |
| phenology | phenology | `project_<crop>.csv` | 400 locations, 1995–2022 |
| growth | lai | `project_<crop>_LAI.csv` | 400 locations (already one season per point) |
| growth | yield | `project_<crop>.csv` | 400 locations, 2000–2023 |

Locations are picked evenly across the sorted ID range, so the geographic spread
of the full set is preserved. `--locations N` overrides every view at once. Set
`n_locations: null` to calibrate on everything.

## The losses — `objectives.py`

| view | loss |
| --- | --- |
| phenology | mean of the flowering and maturity RMSE, in days (autumn bolting excluded) |
| lai | RMSE over DVS-bin × year means of observed vs simulated LAI |
| yield | mean of the yearly-mean RMSE and the state-mean RMSE, in t/ha |

Each is `process_result(run_spec) -> DataFrame` plus `loss_fn(frame) -> (loss,
metrics)` and nothing else, plus `combine()` for the weighted objective. They are
pure scoring functions; nothing in them decides anything. Keeping them in one
module is what lets the diagnostics, the notebook and the ledger all agree on what
a number means.

## Parameters — declared, not hard-coded

`calibration.yaml` declares each calibratable parameter with the biological
`meaning` the agent reasons from, what it `controls`, and how its bounds are
derived:

```yaml
RGRLAI:
  enabled: true
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

Bounds are **relative to each crop's own `crop.xml`**, so one declaration fits all
five crops even though their SLA tables have 8 (wheat), 5 (rapeseed), 3 (potato)
nodes. Table parameters expand to one bound per element actually present. A value
that is structurally zero (leaf allocation after anthesis) stays pinned at zero.

The stage-1 thermal-time parameters are the exception: they use `mode: absolute`
with physiological ranges, because that stage is calibrated from scratch and must
not be tethered to whatever is in the file today.

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
| Provably never written | add to `frozen_groups`, list under the stage's `frozen:` | `frozen` violation, plus the XML re-read after every iteration |

## What cannot happen

| Guard | Mechanism |
| --- | --- |
| Stage 1 is undone by stage 2 | The whole phenology set is frozen for `growth`; a snapshot is taken at handoff and **re-read from the written XML after every change**. A drift aborts the iteration and rolls `crop.xml` back. |
| The two views drift apart | The primary view's `crop.xml` is mirrored into every other view before each run, after every rollback, and on every restore |
| A yield change quietly wrecks the canopy | Both components are scored in the *same* iteration and both are recorded; there is no sequence in which one can be improved unobserved |
| A parameter leaves its range | `check_within_bounds` on the flattened proposal, before the model runs |
| A biologically incoherent profile | Table shape rules: SLA smoothness and decline to maturity, leaf-death rate monotone in temperature, RUE non-increasing after anthesis, storage organs non-decreasing |
| Above-ground allocation stops summing to 1 | Leaves + stems + storage interpolated onto the union DVS grid and compared against the pre-change profile; elements with no possible counterweight are pinned and reported as immovable |
| Too many things change at once | ≤ 3 parameters and ≤ 4 individual values per iteration (2 / 3 for phenology); ≤ 50 % move per value (25 % for phenology) |
| A calibration cannot be undone | `baseline_crop.xml` per stage + `restore-baseline`; `promote` keeps a `crop.xml.pre_<stage>_calibration` backup |
| History is lost | `ledger.jsonl` is append-only; every iteration keeps its own directory with `crop.xml`, the joined obs/sim pairs per view, metrics and figures |

A rejected proposal is written to `rejected.jsonl` and the model is not run. A
`--dry-run` pre-flight is not a proposal and is not recorded anywhere.

## The ledger

```
optimization/calibration/<crop>/<stage>/
  state.json              pointer: iterations, best, frozen digest, run dirs
  ledger.jsonl            append-only, one line per iteration
  rejected.jsonl          proposals that failed validation
  frozen_snapshot.json    the frozen parameters + digest
  baseline_crop.xml       the starting point (restore-baseline puts it back)
  best_crop.xml           lowest objective so far — what promote copies
  current_crop.xml        last good state (used to roll back a failed iteration)
  provenance.json         (growth) which phenology result it was seeded from
  history.png             objective per iteration, annotated with what changed
  iterations/iter_NNN/
      crop.xml            the exact parameters that produced this result
      proposal.json       what changed, the reason, hypothesis, full reasoning
      metrics.json        objective + per-component breakdown + metrics + diagnostics
      simplace_<view>.log
      <view>/pairs.csv.gz       the joined observed/simulated frame — re-analysable
      <view>/season_shape.csv   per location-season curve features (LAI)
      <view>/diagnostics/*.png
```

Each record carries: iteration, crop, stage, views, timestamp, scope per view
(project table, rows, locations, subset, climate), the **full** parameter set,
what changed, the previous values, the objective and its per-component breakdown,
the metrics, the diagnostics, the reason / hypothesis / agent reasoning / expected
effect, whether it improved, whether it is the new best, the delta against the
previous best, the frozen digest and whether the freeze held, any constraint
violations, the figures, and the elapsed time.

## Diagnostics — why not just RMSE

The objective is the loss, but the agent diagnoses on the decomposition:

**LAI** — bias per DVS bin (element *i* of the bin table maps onto element *i* of
`SLATableSLA`, which is what makes the bias directly actionable), then per
location-season: peak LAI, peak timing, peak stage, early-canopy level, rise rate,
plateau duration (days at ≥ 80 % of peak), decline rate — each as observed vs
simulated with the median difference and its IQR. Plus the residual structure: by
DVS, by month, by location, and the share of seasons whose residual never changes
sign. Observed and simulated statistics are always computed **on the matched dates
only**, so a sparse 8-day GLASS retrieval is never compared against a daily
statistic it cannot support.

**Yield** — the error split along `yield = AGBiomass x harvest index`: what the
model produced for each, what each would have to be to match the observations, and
whether those sit inside the agronomic ranges in `calibration.yaml →
crop_reference`. That is what separates "the crop did not make enough biomass"
(RUE, KDIF) from "it made the biomass and did not put it in the grain"
(partitioning, `FRTDM`, N translocation). Plus by-year and by-state residuals, the
residual year trend, correlations against `maxLAI`, `AGBiomass`, `TRANRF` and
`NNI`, and the translocation verdict — whether the `FRTDM` term currently helps or
overshoots.

**Phenology** — the flowering residual and the anthesis-to-maturity **duration**
residual, separately. That split is the whole diagnosis: maturity is dated *from*
anthesis, so a `TSUM1` error propagates into the raw maturity error unchanged and
diagnosing on it moves `TSUM2` to fix a `TSUM1` problem. The residual is then
regressed against season warmth (a slope means the temperature *response* is
wrong — `TEFFMX`, `TsumIncrementTableRate`) and against latitude (a gradient is the
photoperiod signature — `PhotoperiodTableFactor`).

## Local agents — `agentic.py`

Everything stays on the machine: the agent talks to a local Ollama server over
HTTP, gets a JSON proposal back, and hands it to `calibrate.py`.

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
instruction-tuned local model works; the defaults assume a 24–48 GB GPU and should
drop to `:7b` / `:8b` variants on smaller hardware.

**The client uses the standard library only** — no `ollama` package, no
`requests`. The cluster nodes have no outbound network, so a dependency that has
to be installed is a dependency that will not be there.

### What the model actually sees

Not a search space. The context is the current values with their bounds, the
biological `meaning` of every parameter it may touch, the objective components and
their weights, the frozen list in full, the constraints, the diagnostics of the
last completed iteration (one block per view), and every previous iteration with
its reason and outcome. It replies with a small JSON object naming one parameter,
why, and what it expects to happen to each component.

### The repair loop

A proposal is **pre-flighted** before any cluster time is spent on it —
`calibrate.py run --dry-run` validates bounds, table shape, partitioning closure
and blast radius, and writes nothing. If it fails, the exact violation text goes
back to the model and it tries again (`max_repair_attempts`). This is what makes a
small local model usable here: a violated constraint is a far better correction
signal than any amount of prompt tuning, and it costs nothing.

If no valid proposal survives the repair attempts, the iteration is abandoned
rather than forced.

## The agents

**Local** — `optimization/agents/`, run by `agentic.py`:

| agent | job |
| --- | --- |
| `PhenologyAgent` | stage 1 — thermal time against DWD observations |
| `GrowthAgent` | stage 2 — canopy and yield, jointly |
| `AnalystAgent` | read-only review of a calibration in progress; proposes nothing |

Each is a thin class plus a prompt in `agents/prompts/`. The prompt is where the
stage-specific knowledge lives — the symptom-to-parameter table, the rules that
cannot be broken, and when to stop. Adding an agent is a subclass and a prompt.

**Claude Code** — `.claude/agents/`: `crop-model-analyst` (read-only: what drives
what), `phenology-calibrator`, `growth-calibrator`, `calibration-diagnostics`.
`.claude/commands/` — `/calibrate-phenology`, `/calibrate-growth`,
`/calibration-status`, `/calibrate-local`.

## Evaluating a run — `evaluate_run.ipynb`

Set `CROP` in the one configuration cell and run all cells. The notebook compares
simulated against observed **phenology, LAI and yield**, with comparison plots and
RMSE / MAE / R² / bias (plus nRMSE % and Nash–Sutcliffe efficiency) for each.

It finds the outputs itself. `common.discover_runs()` lists every directory under
`runs/` and `runs_optim/` that holds results; by default each variable is scored
against its own calibration run when that has output, and otherwise against the
most recently written scenario run. Set `SOURCE` to a label (`"DWD__S1"`,
`"optim:lai"`) to pin all three to one run.

Loading reuses the same `process_result` functions the calibration uses, so the
notebook's numbers are the numbers the agents were scored on — the loss is not
re-implemented. Each variable's section is independent: a run with no LAI output
reports that and the other two sections still work.

Two things to know:

- **LAI from a scenario run is not like-for-like.** The LAI observations belong to
  their own point set and weather grid, so a scenario run matches only a fraction
  of them. The notebook warns when fewer than half the simulated locations carry
  observations; use the growth stage's `lai` view for a real LAI evaluation.
- **Daily output is sampled.** A full run's daily files are tens of millions of
  rows; `MAX_LAI_LOCATIONS` (default 300) reads an evenly spaced subset. Raise it
  for a final evaluation, and note the LAI metrics move slightly with it.

## Self-test

```bash
python optimization/test_calibrate.py     # 67 checks, no cluster required
```

Covers config loading, per-crop space resolution for both stages, the objective
combination arithmetic, every constraint rule, the freeze guard, the crop.xml
mirroring across views, proposal normalisation, the ledger and stopping rules, and
the diagnostics against synthetic data with known level, phase and duration errors.

It also covers the local-agent layer without needing Ollama: the JSON extractor
against prose and fenced replies, the Ollama client against a **stand-in HTTP
server that implements `/api/tags` and `/api/chat` exactly as Ollama does** (which
is what verifies the wire format on a machine with no Ollama installed), and the
full agent loop — context rendering, reply parsing, the repair round-trip, the
freeze abort, self-stopping — against a scripted mock backend.

The mock is a test double, never a fallback. A missing Ollama at runtime is
reported as an error; nothing silently substitutes for the model.

---

## Adding a view, a stage or a crop

A **view** is:

1. a run definition under `config.yaml → targets.<stage>.views`,
2. a `process_result` / `loss_fn` pair in `objectives.py`, registered in `VIEWS`,
3. an evaluator in `evaluation.py`, registered in `EVALUATORS`,
4. a weight and a scale under the stage's `objective.components`.

A **stage** is a `config.yaml` entry with its views, a `calibration.yaml` block
(parameters with their `meaning`, objective, constraints, freeze groups, stopping
rules), and — if it should have a local agent — a `CalibrationAgent` subclass and
a prompt.

Everything else — staging, XML editing, the freeze guard, validation, the ledger,
stopping — is inherited.

**Crops** need a `crops:` entry whose `crop_name` matches
`<parameter id="CropName">` in that crop's `data/crop/crop.xml`, and `dm_fraction`
if observed yield is not on a dry-matter basis (potato: observed is fresh tuber
weight, so `0.21`).

## Note on the previous layout

The calibration used to run as three separate stages (`phenology` → `lai` →
`yield`) with one run dir and one ledger each. Ledgers from that layout are still
on disk under `optimization/calibration/<crop>__<target>/`, and their run dirs
under `simplace/<crop>/runs_optim/{calib_phenology,calib_lai,calib_yield}/`
(alongside the new `calib_phenology/<view>/` and `calib_growth/<view>/`). Nothing
reads them any more; they are kept as history and can be deleted once you no
longer want the record. Objectives recorded there are **not** comparable with the
combined growth objective — compare the per-component losses instead.
