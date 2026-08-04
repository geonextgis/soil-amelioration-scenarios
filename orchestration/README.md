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
