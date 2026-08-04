# `optimization/` — single-trial SIMPLACE calibration

Three scripts, one per calibration target. **Each invocation runs exactly one
trial** — one parameter draw, one SIMPLACE run, one loss — then prints the drawn
parameters and the loss and exits.

```
optimization/
  config.yaml              parameter spaces, subsets, SLURM + Optuna settings
  common.py                shared plumbing (staging, XML edits, study, reporting)
  optimize_phenology.py    loss: mean RMSE of flowering + maturity DOY
  optimize_lai.py          loss: RMSE of DVS-binned LAI
  optimize_yield.py        loss: mean RMSE of yearly-mean + state-mean yield
  evaluate_run.ipynb       visual + metric evaluation of a finished run
  studies/                 <crop>__<target>.db   (Optuna/SQLite, created at runtime)
  results/<crop>__<target>/
      last.json            the most recent trial
      history.jsonl        one line per trial, append-only
      best_crop.xml        crop.xml of the best trial so far
      trial_<n>_crop.xml   crop.xml of each trial
      simplace_trial_<n>.log
```

## Run one trial

```bash
python optimization/optimize_phenology.py --crop winter_wheat
python optimization/optimize_lai.py       --crop winter_wheat
python optimization/optimize_yield.py     --crop winter_wheat
```

Useful flags: `--show-best` (report the best trial, run nothing), `--rebuild`
(rebuild the calibration run dir), `--skip-run` (re-score the outputs already in
`out/` without simulating — handy when iterating on a loss), `--device local`,
`--config`, `--crop`.

## Iterating (Claude Loop)

Trial history is persisted in `studies/<crop>__<target>.db`, so re-running the
same command *is* the iteration protocol: each invocation loads the previous
trials, asks the TPE sampler for the next parameter set, and appends its result.
Nothing needs to be threaded between runs.

```
/loop python optimization/optimize_phenology.py --crop winter_wheat
```

Each iteration ends with a machine-readable line the loop can read back:

```
JSON {"trial": 7, "loss": 6.4213, "parameters": {...}, "best_loss": 6.4213,
      "best_parameters": {...}, "is_new_best": true, "completed_trials": 8}
```

Stop when `is_new_best` has been `false` for a while, or when `best_loss` clears
whatever threshold the target needs.

The first trial of a fresh study is the **current `crop.xml` values**, so every
later trial has a reference loss to beat rather than an arbitrary first draw.

## Evaluating a run — `evaluate_run.ipynb`

Set `CROP` in the one configuration cell and run all cells. The notebook compares
simulated against observed **phenology, LAI and yield**, with comparison plots and
RMSE / MAE / R² / bias (plus nRMSE% and Nash–Sutcliffe efficiency) for each.

It finds the outputs itself. `common.discover_runs()` lists every directory under
`runs/` and `runs_optim/` that holds results; by default each variable is scored
against its own calibration run (`optim:<target>`) when that has output, and
otherwise against the most recently written scenario run. Set `SOURCE` to a label
(`"DWD__S1"`, `"optim:lai"`) to pin all three to one run.

Loading reuses the same `process_result` functions the calibration scripts use, so
the notebook's numbers are the numbers the optimizer scored — the loss is not
re-implemented. Each variable's section is independent: a run with no LAI output
reports that and the other two sections still work.

Two things to know:

- **LAI from a scenario run is not like-for-like.** The LAI observations belong to
  their own point set and weather grid, so a scenario run matches only a fraction of
  them. The notebook warns when fewer than half the simulated locations carry
  observations; use `SOURCE="optim:lai"` for a real LAI evaluation.
- **Daily output is sampled.** A full run's daily files are tens of millions of rows;
  `MAX_LAI_LOCATIONS` (default 300) reads an evenly spaced subset. Raise it for a
  final evaluation, and note the LAI metrics move slightly with it.

## What a trial actually does

```
draw params -> write crop.xml -> clear out/ -> run SIMPLACE -> read out/ -> loss -> tell study
```

Calibration runs are isolated in `simplace/<crop>/runs_optim/<target>/`, built
the same way `orchestration/generate.py` builds scenario runs (and reusing its
`project.proj.xml` templating):

- `data/crop/` is a **real copy** — `crop.xml` is rewritten every trial, so it
  must not alias the production file. `simplace/<crop>/data/crop/crop.xml` is
  never modified by these scripts.
- `data/soil/soil.csv`, `data/management/location.csv`,
  `data/management/fertilizer_<crop>.csv` are copies of the staged source tables
  (see *input profiles* below).
- `solution/`, `data/{slim,soilcnp,co2}` and the static management XMLs are
  symlinks to the crop's shared inputs.
- `project/project.csv` is generated: subset per `config.yaml`, then sorted
  location-contiguous and re-numbered.
- `config.yaml` is the `cluster:` block handed to `simplace_runner_cluster.py`.

Two things in there are load-bearing rather than cosmetic:

**Location-contiguous sorting.** SIMPLACE writes one output file per location
into a shared directory, so the cluster runner may only split work on location
boundaries. The baseline `project_<crop>.csv` appends each location's recent
years at the file tail, so an unsorted table would let two SLURM tasks handle
the same location and silently clobber each other's output.

**Clearing `out/` before each trial.** Per-location output files persist, so a
location dropped by a failed task would otherwise contribute the *previous*
trial's values to this trial's loss.

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

`config.yaml → targets.<target>.subset` trims the project table so a trial is
cheap enough to iterate on:

| target | project table | subset |
| --- | --- | --- |
| phenology | `project_<crop>.csv` | 300 locations, 1995–2022 |
| lai | `project_<crop>_LAI.csv` | 400 locations (table is already one season per point) |
| yield | `project_<crop>.csv` | 400 locations, 1995–2023 |

Locations are picked evenly across the sorted ID range, so the geographic spread
of the full set is preserved. Set `n_locations: null` (and the year bounds to
`null`) to calibrate on everything.

## Promoting a result

The scripts never write to the production crop file. When a study has converged:

```bash
python optimization/optimize_yield.py --crop winter_wheat --show-best
cp optimization/results/winter_wheat__yield/best_crop.xml \
   simplace/winter_wheat/data/crop/crop.xml
```

Diff before copying — `best_crop.xml` carries only that target's parameters
changed away from whatever `crop.xml` held when the study started.

## Adding a target or a crop

A target is a `config.yaml` block plus a script that defines `evaluate(spec) ->
(loss, metrics)` and calls `common.main(TARGET, __doc__, evaluate)`. Everything
else — staging, sampling, XML editing, the study, reporting — is inherited.

Crops need a `crops:` entry whose `crop_name` matches
`<parameter id="CropName">` in that crop's `data/crop/crop.xml`, and
`dm_fraction` if observed yield is not on a dry-matter basis (potato: observed
is fresh tuber weight, so `0.21`).
