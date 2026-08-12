# Soil Amelioration Scenario Experiment Framework

Crop-model (SIMPLACE / LINTUL5) simulation framework for evaluating soil
amelioration scenarios across baseline, historical, and future climate in Germany.

Five crops: winter wheat, winter rapeseed, spring barley, potato, maize.
They are calibrated in two stages — phenology first, then LAI and yield jointly
(see *Calibration* below) — and this phase runs **large-scale scenario
simulations** with the calibrated crops.

> **Paths are portable — do not hard-code the repo location.** The repo root is
> derived from the file that needs it: `orchestration/generate.py` and
> `optimization/common.py` expose `resolve_repo_root()`, and the notebooks use an
> inline `_find_repo_root()` that walks up from the notebook's directory. Both
> honour the `SOIL_SCENARIOS_ROOT` environment variable, and both config files set
> `repo_root: auto`. Paths *outside* the repo (shared cluster stores) stay absolute
> but are env-overridable: `EXTERNAL_DATA_DIR`, `DWD_PHENOLOGY_DATA`,
> `DWD_CLIMATE_DIR`.
>
> An older checkout may still exist at
> `/beegfs/halder/GITHUB/RESEARCH/soil-amelioration-scenarios`. Because it is a
> valid-looking repo, a stale absolute `repo_root` there fails *silently* — runs
> read and write the wrong tree. Leave `repo_root: auto`.

## Repository Layout

```
data/
  raw/
    soil_scenarios/            # 5 amelioration scenarios (S1_BZE is the baseline)
      S1_BZE.csv  S2_BZE.csv  S3_BZE.csv  DL_BZE.csv  DLB_BZE.csv
    point_to_nearest_grid.csv  # PointID -> nearest climate grid cell mapping
    Site_Soil_BZE_WGS84.gpkg   # site geometries
    GLASS_LAI/                 # remote-sensing LAI reference
  external/  interim/  processed/
notebooks/                     # 00_exploration .. 04_evaluation
simplace/<crop>/               # one folder per crop (the simulation engine)
```

### Per-crop folder (`simplace/<crop>/`)
- `runs/<exp_id>/`           — generated experiment run dirs, one per (climate, soil);
  each holds its own `config.yaml` (+ `config_smoke.yaml`), `project/project.csv`,
  templated `project.proj.xml`, and staged `data/soil/soil.csv` + `data/co2/co2.csv`.
  Produced by `orchestration/generate.py` — do not hand-edit.
- `runs_optim/calib_<target>/` — isolated calibration run dirs (generated)
- `simplace_runner.py`       — local single-run driver
- `simplace_runner_cluster.py` — SLURM/Singularity batch driver (the cluster entrypoint)
- `solution/solution.sol.xml`  — SIMPLACE solution
- `project/project.proj.xml`   — SIMPLACE project definition (interfaces + header)
- `project/project_<crop>.csv` — main baseline project input table (one row per simulation)
- `data/`                    — model inputs (crop, soil, weather, management, slim, soilcnp)
- `data_observed/`           — observed phenology / LAI / yield for calibration
- `out/<EXP_NAME>/{daily,yearly}/` — simulation outputs

## Key Data Contracts

**`PointID` is the primary key** linking agricultural points → climate grid cells.
`data/raw/point_to_nearest_grid.csv` columns:
`PointID, NUTS_ID, NUTS_NAME, STATE_NAME, Latitude, Longitude, nearest_grid_id,
nearest_latitude, nearest_longitude, distance_deg` where `nearest_grid_id` is `C<col>R<row>`.

**Soil CSVs** (`soil_scenarios/*.csv`) are keyed by `location` (= PointID), with
6-layer profiles. `S1_BZE` is the baseline (the former `soil.csv` was identical and
was removed); `S2/S3/DL/DLB_BZE` are the amelioration variants. All share the same
schema, so they are interchangeable inputs.

**Project input CSV** (`project/project_<crop>.csv`, the main baseline project file,
`;`-delimited) header:
`projectid;simulationid;vColumn;vRow;vLocationID;vNUTS_ID;vNUTS_NAME;vSTATE_NAME;start_date;end_date;vIDPL`
- `vColumn`/`vRow` → weather grid cell; `vLocationID` = PointID.
- `vIDPL` = planting day-of-year (management lever, see scenarios below).

**Weather interface** (from `project.proj.xml`) reads:
`${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz`
`_DATADIR_` is bound to the climate source via `mount_data` in `config.yaml`.

## Climate Sources

- **DWD observations (baseline):**
  `/beegfs/common/data/climate/dwd/csvs/germany_ubn_1951-01-01_to_2024-08-30`
- **HYRAS bias-corrected CMIP (historical + future):**
  `/data01/FDS/muduchuru/Atmos/NEXGDDP_HYRAS_BC_CSV/<MODEL>/<SCENARIO>/<col>/...`
  - Models: `ACCESS-CM2, CanESM5, EC-Earth3, GFDL-ESM4, MIROC6` (5 GCMs), plus `OBS` (HYRAS).
  - Scenarios per GCM: `historical, ssp126, ssp370`.

Note the two sources differ in **layout and delimiter**: DWD is tab-delimited
gzip foldered by *row*; HYRAS is comma-delimited plain CSV foldered by *column*.
`generate.py` rewrites the `weatherfile` interface (divider + filename) per
climate, so the solution's own weather interface is always overridden.

**CO₂ forcing.** Every solution reads `data/co2/co2.csv`; the climate decides which
of the crop's `data/co2/*.csv` is staged there — same pattern as the soil scenario.
The resource is keyed `(CURRENT.YEAR, CURRENT.MONTH)` at DAILY frequency, so a
month with no row is a NullPointerException, not a gap.

| climate | staged file | span |
| --- | --- | --- |
| DWD, HYRAS OBS, all GCM `historical` | `co2_mm_historical.csv` | 1951-01 … 2026-06 |
| `ssp126` / `ssp370` | `co2_mm_ssp126_future.csv` / `co2_mm_ssp370_future.csv` | 2015 … 2500 |

`co2_mm_historical.csv` is **generated**, not raw: `co2_mm_observed.csv` is the
Mauna Loa record and starts 1958-03, seven years after the historical period does.
`orchestration/build_co2_historical.py` prepends 1951-01 … 1958-02 from the Law
Dome ice-core/firn spline (`data/external/law_dome_co2_spline_annual.csv`) with a
Mauna-Loa-derived seasonal cycle and a splice offset, and copies the observed
months through verbatim. Re-run it (`--check` to validate only) if the observed
record is updated. Calibration (`optimization/common.py`) still uses the raw
observed file — it runs on the DWD baseline from 1979, where the record is complete.

## Experiment Matrix (per soil condition)

| Type       | Period      | Climate            | Scenarios | Management (`vIDPL`)              |
| ---------- | ----------- | ------------------ | --------- | -------------------------------- |
| Baseline   | 1978–2024   | DWD observations   | 1         | dynamic, per observed year       |
| Historical | 1951–2014   | HYRAS OBS + 5 GCMs | 6         | median IDPL per district (NUTS)  |
| Future     | 2015–2100   | 5 GCMs × {SSP126, SSP370} | 10 | median IDPL per district (NUTS)  |
| **Total**  |             |                    | **17**    |                                  |

**17 climate scenarios × 5 soil scenarios = 85 experiments** per crop.

Periods above are the *nominal* spans in `orchestration/experiments.yaml`. The
actual windows written into each `project.csv` are narrower, for two reasons that
`generate.py` handles automatically:

- **Window length is per crop.** `winter_wheat` and `winter_rapeseed` are sown in
  autumn of year Y and harvested in Y+1, so a window is `Y-01-01 .. Y+1-12-31`;
  the three spring crops fit one calendar year. Length is read off the crop's own
  baseline CSV (`baseline_span_years`), never assumed. Getting this wrong is
  silent: a winter crop cut off at Dec 31 of the sowing year never reaches
  harvest, and the yearly output only fires on `HarvestManagement.DoHarvest`, so
  the run "succeeds" with header-only files.
- **Windows are clamped to the weather actually on disk** (`probe_coverage` +
  `clamp_to_coverage`). DWD stops on 2024-08-30, and CanESM5/GFDL-ESM4 ship files
  named `19510101_20141231` that really cover 1950-12-08 .. 2014-11-21. A window
  running past the end of its weather file dies with a NullPointerException.

> **CanESM5 and GFDL-ESM4 are on a shifted calendar.** Their data starts 24 days
> off nominal and ends 40 days off — a no-leap (365-day) model calendar re-stamped
> onto consecutive real dates, so the offset grows across the record and a
> day-of-year sowing rule lands progressively earlier in the real season.
> `generate.py` drops the unusable windows and prints a warning, but the seasonal
> misalignment is a property of the source data (owned by `muduchuru`) and cannot
> be fixed downstream. Confirm the intended handling before publishing these runs.

**Point set:** `point_to_nearest_grid.csv` carries 3099 PointIDs, but
`location.csv` and `fertilizer_<crop>.csv` only cover the **3086** in the baseline
project files. Generated experiments use those 3086 (`baseline_points`), so
hist/future stay comparable with the baseline.

## Running Simulations

Generate the run dirs, then submit:

```bash
python orchestration/generate.py --crop all --climate all --soil all
bash simplace/runs_submit/submit_<label>_<hash>.sh
```

The cluster driver reads the `cluster:` block of a generated run dir's config:

```bash
python simplace/<crop>/simplace_runner_cluster.py simplace/<crop>/runs/<exp_id>/config.yaml
```

Each runner blocks until its own SLURM jobs finish, so the generated submit script
drives `slurm.cluster_nodes // slurm.num_nodes` experiments at a time. Do not
background every line instead — that oversubscribes the partition and starts one
`squeue` poller per experiment. Smoke-test first with the run dir's
`config_smoke.yaml` (3 locations, 1 node, output namespaced `SMOKE_<exp_id>`).

It counts rows in `input_csv`, splits `[start_line, end_line]` across `num_nodes`
jobs (and `num_tasks_per_node` srun tasks each), generates a SLURM batch script per
job, runs SIMPLACE inside the Singularity image, and waits for completion via `squeue`.

Key `cluster:` keys to switch an experiment:
- `mount_data`   — climate source root (DWD vs. a GCM/scenario folder) → bound to `/data`
- `exp_name`     — output namespace (`out/<exp_name>/`)
- `input_csv`    — the project input table (selects period + `vIDPL` management)
- `solution` / `project`, `singularity_image`, `partition`, `walltime`,
  `num_nodes`, `num_tasks_per_node`, `start_line`/`end_line`.

## Workflow Goals for This Phase

Build a **human-readable (YAML) configuration/orchestration layer** that can:
1. Launch a single experiment, a subset, or all 85 at once.
2. Cleanly switch crop, soil scenario, climate model, climate scenario,
   simulation period, and `vIDPL` management strategy.
3. Generate the per-experiment project CSV (correct period, `vIDPL` rule, grid mapping)
   and the matching `cluster:` config before submission — reproducibly.

When generating project files: join points to grids via `point_to_nearest_grid.csv`,
set `start_date`/`end_date` per the period table, and set `vIDPL` per the management
rule (dynamic per-year for baseline; district-median for historical/future).

## Calibration

**Two stages**, in this order:

1. **`phenology`** — thermal time (`TSUM1`, `TSUM2`, `TEFFMX`, …), calibrated
   from scratch on its own. Everything downstream is dated off DVS, so the
   development clock is settled first and then frozen.
2. **`growth`** — **LAI and yield calibrated jointly**, with the stage-1
   phenology frozen. They are not separable: RUE, KDIF and the partitioning
   tables move biomass and leaf area at the same time, so calibrating them in
   sequence means each undoes the other. One iteration runs SIMPLACE **twice** —
   the GLASS-LAI point set and the district yield point set — from one
   `crop.xml`, and scores one combined objective (each component divided by its
   own target, so 1.0 = both at target on average).

**There is no optimizer.** No Optuna, no Bayesian search, no sampler. Every
parameter change comes from an agent that reads the diagnostics, names a
mechanism, and states what it expects to happen — for the growth stage, on
*both* components. Two agent runtimes drive the same machinery:

- **Local (Ollama)** — `python optimization/agentic.py run --crop <crop> --target growth`.
  Agents in `optimization/agents/` talk to a local model server; nothing leaves
  the machine. `agentic.py check` reports whether Ollama is reachable and the
  configured models are pulled.
- **Claude Code** — `/calibrate-phenology`, `/calibrate-growth`, driven by the
  agents in `.claude/agents/`.

Both go through `optimization/calibrate.py`, which is the only path to
`crop.xml`. It validates against the constraints in `calibration.yaml`, verifies
the freeze by re-reading the written XML, mirrors the file into every view, runs
SIMPLACE, scores with `optimization/objectives.py` +
`optimization/evaluation.py`, and appends to
`optimization/calibration/<crop>/<stage>/ledger.jsonl`.

```bash
python optimization/calibrate.py run     --crop <crop> --target phenology   # iterate
python optimization/calibrate.py promote --crop <crop> --target phenology --yes
python optimization/calibrate.py handoff --crop <crop>                      # seeds stage 2
python optimization/calibrate.py run     --crop <crop> --target growth      # iterate
python optimization/calibrate.py promote --crop <crop> --target growth --yes
```

Nothing writes to `simplace/<crop>/data/crop/crop.xml` except
`calibrate.py promote --yes`; do not edit it by hand. `restore-baseline` puts a
stage's starting parameters back if a calibration goes somewhere useless.

Self-test, no cluster needed: `python optimization/test_calibrate.py`.
See `optimization/README.md` for the full contract and the step-by-step runbook.
