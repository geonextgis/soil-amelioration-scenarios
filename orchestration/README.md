# Experiment orchestration layer

Human-readable YAML control surface for the 85-experiment matrix
(17 climate scenarios × 5 soil scenarios) per crop.

- `experiments.yaml` — the single config: crops, soils, climate sources, periods, SLURM defaults.
- `generate.py` — builds one isolated, reproducible run directory per `(crop, climate, soil)`.

## Usage

```bash
python orchestration/generate.py --list-climates                          # 17 climate ids
python orchestration/generate.py --crop maize --climate DWD        --soil S1   # baseline
python orchestration/generate.py --crop maize --climate GFDL-ESM4_ssp370 --soil S2 [--dry-run]
```

Each call writes `simplace/<crop>/runs/<climate>__<soil>/` containing a project CSV,
a templated `project.proj.xml`, the soil scenario as `data/soil/soil.csv`, symlinks to
the crop's shared inputs/solution, and a `config.yaml`. Submit with the existing runner:

```bash
python simplace/<crop>/simplace_runner_cluster.py simplace/<crop>/runs/<exp_id>/config.yaml
```

Run dirs are independent, so all 85 can be generated and submitted in parallel.

## Submitting a set of experiments

`generate.py` writes two drivers into `simplace/runs_submit/` for whatever it just
generated. They differ only in **who owns the nodes**:

| | `campaign_<label>_<hash>.sbatch` | `submit_<label>_<hash>.sh` |
|---|---|---|
| Allocation | one, `slurm.campaign_nodes`, held start to finish | one set per experiment, released after each |
| Experiments | sequential inside that allocation, each using all of it | `cluster_nodes // num_nodes` at a time |
| Queue waits | once, before the first experiment | once **per experiment** |
| Run with | `sbatch campaign_....sbatch` | `bash submit_....sh` |

```bash
sbatch simplace/runs_submit/campaign_<label>_<hash>.sbatch                       # preferred
sbatch --export=ALL,SIMPLACE_RESUME=1 simplace/runs_submit/campaign_<label>_<hash>.sbatch  # resume
```

The campaign job is the one to use when the partition is busy: re-queueing between
experiments is what dominates wall-clock there. Inside the allocation each
experiment runs as `nodes × num_tasks_per_node` concurrent `srun` job steps
(`simplace_runner_cluster.py --mode alloc`) over location-aligned chunks of its
project CSV — nothing is submitted, so nothing waits.

Its two costs are worth knowing before you submit:

- **It starts only when `campaign_nodes` are free at once.** A smaller
  `campaign_nodes` starts sooner; a larger one finishes sooner once started.
- **`campaign_walltime` has to cover every experiment end to end.** Each
  experiment writes `.completed_<exp_id>` in its run dir on success, and the job
  refuses to start an experiment it cannot finish (exit 3 → the loop stops
  cleanly rather than losing a half-written experiment to the walltime). Resubmit
  with `SIMPLACE_RESUME=1` to pick up the rest.

Progress: `simplace/runs_submit/logs/campaign_<label>_<hash>-<jobid>.out` for the
campaign, `<run_dir>/log/step_<exp>_<first>_<last>.out` for an individual chunk.

## What the generator handles (contracts that differ by climate source)

| | Baseline (DWD) | HYRAS (OBS + 5 GCMs) |
|---|---|---|
| Grid | existing `project_<crop>.csv` (DWD grid) | `point_to_nearest_grid.csv` (≠ DWD grid) |
| Weather folder key | `${vRow}` | `${vColumn}` |
| Weather filename | `daily_mean_RES1_C{col}R{row}.csv.gz` | `<MODEL>_<SCEN>_<dates>_C{col}R{row}.csv` |
| Delimiter | tab (`<divider />`) | comma (`<divider>,</divider>`) |
| `vIDPL` | dynamic per observed year | median per NUTS_ID (from baseline CSV) |
| Period | per existing CSV | historical 1951–2014 / future 2015–2100 |

## Status / next steps

Validated on **maize** (baseline `DWD__S1` + future `GFDL-ESM4_ssp370__S2`); the templated
HYRAS weather path resolves to a real on-disk file. To scale: add a batch driver that loops
`crops × climates × soils` (optionally submitting), once one full experiment is confirmed in SIMPLACE.

> First real HYRAS run should confirm the weather **column mapping** inside the solution's
> weatherfile interface — HYRAS columns (`time,Precipitation,...`) differ from DWD
> (`Date,...,Gridcell`); only the filename/divider are templated here.
