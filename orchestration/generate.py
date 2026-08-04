#!/usr/bin/env python
"""Generate isolated, reproducible run directories for soil-amelioration experiments.

One experiment = (crop, climate, soil). For each, this builds a self-contained
run dir under ``simplace/<crop>/runs/<exp_id>/`` so that all 85 experiments per
crop can be generated and submitted without clobbering each other:

    runs/<exp_id>/
      solution/solution.sol.xml   -> symlink to the crop's shared solution
      data/{crop,management,slim,soilcnp} -> symlinks to the crop's shared inputs
      data/soil/soil.csv          (real file: the chosen soil scenario)
      data/co2/co2.csv            (real file: the CO2 forcing for this climate)
      project/project.proj.xml    (templated: weather path + divider per climate)
      project/project.csv         (generated: period + grid + vIDPL management)
      config.yaml                 (the `cluster:` block for simplace_runner_cluster.py)
      out/                        (created by the runner)

Climate contracts discovered from the data (NOT all documented in CLAUDE.md):
  * DWD baseline:  ${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz,
                   TAB-delimited, gzip. Uses the DWD grid already baked into the
                   existing project_<crop>.csv.
  * HYRAS (OBS + 5 GCMs): foldered by COLUMN, plain comma-delimited .csv with the
                   model/scenario/date-range in the filename. Uses the grid in
                   point_to_nearest_grid.csv (which differs from the DWD grid).

vIDPL management:
  * baseline  -> dynamic per observed year (already in the baseline project CSV).
  * hist/fut  -> median IDPL per district (NUTS_ID), computed from the baseline CSV.

Usage:
  python orchestration/generate.py --list-climates
  python orchestration/generate.py --crop maize --climate DWD --soil S1
  python orchestration/generate.py --crop maize --climate GFDL-ESM4_ssp370 --soil S2 --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# --- DWD weather contract (baseline) ---------------------------------------
DWD_WEATHER = "${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz"
DWD_DIVIDER = None  # whitespace/tab -> SIMPLACE self-closing <divider />

# Shared crop input subdirs that are symlinked (everything except soil and co2,
# which are staged per experiment).
SHARED_DATA_SUBDIRS = ["crop", "management", "slim", "soilcnp"]

# CO2 forcing. Every solution reads data/co2/co2.csv; the climate decides which of
# the crop's data/co2/*.csv is copied to that name — same pattern as the soil
# scenario -> data/soil/soil.csv. Without this the SSP runs would silently be
# driven by observed CO2, which also has no rows past 2026.
CO2_OBSERVED = "co2_mm_observed.csv"
CO2_BY_SSP = {"ssp126": "co2_mm_ssp126_future.csv",
              "ssp370": "co2_mm_ssp370_future.csv"}


@dataclass
class Climate:
    id: str
    kind: str            # "baseline" | "historical" | "future"
    mount_data: str      # bound to /data; weather paths are relative to it
    weather_path: str    # SIMPLACE filename template (uses ${_DATADIR_} ${vColumn} ${vRow})
    divider: str | None  # None -> whitespace; "," -> comma
    start: int
    end: int
    idpl_rule: str       # "dynamic" | "nuts_median"
    grid: str            # "baseline" (reuse existing CSV) | "hyras"
    co2_file: str        # which data/co2/*.csv is staged as data/co2/co2.csv


def build_climate_registry(cfg: dict) -> dict[str, Climate]:
    """Derive the 17 climate sources from the matrix in experiments.yaml."""
    root = cfg["climate_root_hyras"]
    hist = cfg["periods"]["historical"]
    fut = cfg["periods"]["future"]
    reg: dict[str, Climate] = {}

    # 1) DWD baseline.
    reg["DWD"] = Climate(
        id="DWD", kind="baseline", mount_data=cfg["climate_dwd"],
        weather_path=DWD_WEATHER, divider=DWD_DIVIDER,
        start=0, end=0, idpl_rule="dynamic", grid="baseline",
        co2_file=CO2_OBSERVED,
    )

    # 2) HYRAS OBS (historical observations).
    reg["HYRAS_OBS"] = Climate(
        id="HYRAS_OBS", kind="historical", mount_data=f"{root}/OBS",
        weather_path=("${_DATADIR_}/${vColumn}/"
                      f"obs_{hist['datestr']}_C${{vColumn}}R${{vRow}}.csv"),
        divider=",", start=hist["start"], end=hist["end"],
        idpl_rule="nuts_median", grid="hyras", co2_file=CO2_OBSERVED,
    )

    # 3) 5 GCMs x historical.
    for m in cfg["gcms"]:
        reg[f"{m}_historical"] = Climate(
            id=f"{m}_historical", kind="historical",
            mount_data=f"{root}/{m}/historical",
            weather_path=("${_DATADIR_}/${vColumn}/"
                          f"{m}_historical_{hist['datestr']}_C${{vColumn}}R${{vRow}}.csv"),
            divider=",", start=hist["start"], end=hist["end"],
            idpl_rule="nuts_median", grid="hyras", co2_file=CO2_OBSERVED,
        )

    # 4) 5 GCMs x {ssp126, ssp370} future.
    for m in cfg["gcms"]:
        for ssp in ("ssp126", "ssp370"):
            reg[f"{m}_{ssp}"] = Climate(
                id=f"{m}_{ssp}", kind="future",
                mount_data=f"{root}/{m}/{ssp}",
                weather_path=("${_DATADIR_}/${vColumn}/"
                              f"{m}_{ssp}_{fut['datestr']}_C${{vColumn}}R${{vRow}}.csv"),
                divider=",", start=fut["start"], end=fut["end"],
                idpl_rule="nuts_median", grid="hyras", co2_file=CO2_BY_SSP[ssp],
            )
    return reg


# --- project.proj.xml templating -------------------------------------------
def render_proj_xml(template_text: str, climate: Climate) -> str:
    """Rewrite the project-data CSV path and the weather interface for a climate."""
    # Point the project-data interface at this run's generated CSV.
    text = re.sub(
        r"(<interface id=\"projectdata\".*?<filename>)[^<]*(</filename>)",
        r"\1${_WORKDIR_}/project/project.csv\2",
        template_text, flags=re.DOTALL,
    )

    # Rebuild the weather interface body (divider + filename) for this source.
    divider_tag = "<divider />" if climate.divider is None else f"<divider>{climate.divider}</divider>"

    def _weather(match: re.Match) -> str:
        head = match.group(1)   # up to and including <poolsize>...</poolsize>
        return (f"{head}\n\t\t\t{divider_tag}"
                f"\n\t\t\t<filename>{climate.weather_path}</filename>\n\t\t")

    text, n = re.subn(
        r"(<interface id=\"weatherfile\"[^>]*>.*?</poolsize>).*?(?=</interface>)",
        _weather, text, flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError(f"weatherfile interface not rewritten cleanly (matched {n}x)")
    return text


# --- project.csv generation -------------------------------------------------
def nuts_median_idpl(baseline_csv: Path) -> tuple[dict[str, int], int]:
    """Median planting day-of-year per NUTS_ID from the baseline project CSV."""
    df = pd.read_csv(baseline_csv, sep=";", usecols=["vNUTS_ID", "vIDPL"])
    per_nuts = df.groupby("vNUTS_ID")["vIDPL"].median().round().astype(int).to_dict()
    global_median = int(round(df["vIDPL"].median()))
    return per_nuts, global_median


def parse_grid(grid_id: str) -> tuple[int, int]:
    """'C325R23' -> (325, 23)."""
    m = re.fullmatch(r"C(\d+)R(\d+)", str(grid_id).strip())
    if not m:
        raise ValueError(f"bad grid id: {grid_id!r}")
    return int(m.group(1)), int(m.group(2))


def build_hyras_project(point_grid_csv: Path, baseline_csv: Path,
                        climate: Climate) -> pd.DataFrame:
    """One row per (point, year) over the climate period; vIDPL = NUTS median."""
    pg = pd.read_csv(point_grid_csv)
    per_nuts, global_median = nuts_median_idpl(baseline_csv)

    cols = pg["nearest_grid_id"].map(parse_grid)
    pg = pg.assign(vColumn=[c for c, _ in cols], vRow=[r for _, r in cols])
    pg["vIDPL"] = pg["NUTS_ID"].map(per_nuts).fillna(global_median).astype(int)

    years = range(climate.start, climate.end + 1)
    base = pg[["PointID", "vColumn", "vRow", "nearest_grid_id",
               "NUTS_ID", "NUTS_NAME", "STATE_NAME", "vIDPL"]].copy()
    rep = base.loc[base.index.repeat(len(years))].reset_index(drop=True)
    rep["year"] = list(years) * len(base)

    out = pd.DataFrame({
        "projectid": range(1, len(rep) + 1),
        "simulationid": rep["nearest_grid_id"],
        "vColumn": rep["vColumn"],
        "vRow": rep["vRow"],
        "vLocationID": rep["PointID"],
        "vNUTS_ID": rep["NUTS_ID"],
        "vNUTS_NAME": rep["NUTS_NAME"],
        "vSTATE_NAME": rep["STATE_NAME"],
        "start_date": rep["year"].astype(str) + "-01-01",
        "end_date": rep["year"].astype(str) + "-12-31",
        "vIDPL": rep["vIDPL"],
    })
    return out


# --- run-dir assembly -------------------------------------------------------
def link_or_replace(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def generate(crop: str, climate: Climate, soil: str, cfg: dict,
             dry_run: bool = False) -> dict:
    repo = Path(cfg["repo_root"])
    crop_dir = repo / "simplace" / crop
    exp_id = f"{climate.id}__{soil}"
    run_dir = crop_dir / cfg["paths"]["runs_subdir"] / exp_id

    baseline_csv = crop_dir / "project" / f"project_{crop}.csv"
    soil_src = repo / cfg["paths"]["soil_dir"] / f"{soil}_BZE.csv"
    proj_template = (crop_dir / "project" / "project.proj.xml").read_text()

    plan = {
        "exp_id": exp_id, "crop": crop, "climate": climate.id, "soil": soil,
        "run_dir": str(run_dir), "mount_data": climate.mount_data,
        "grid": climate.grid, "idpl_rule": climate.idpl_rule,
        "co2": climate.co2_file,
        "period": "baseline (per existing CSV)" if climate.grid == "baseline"
                  else f"{climate.start}-{climate.end}",
    }

    for need in (baseline_csv, soil_src):
        if not need.exists():
            raise FileNotFoundError(need)

    if dry_run:
        plan["status"] = "dry-run (nothing written)"
        return plan

    # Directory skeleton + symlinks to shared inputs.
    (run_dir / "project").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "soil").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "co2").mkdir(parents=True, exist_ok=True)
    (run_dir / "out").mkdir(parents=True, exist_ok=True)
    (run_dir / "solution").mkdir(parents=True, exist_ok=True)
    link_or_replace(crop_dir / "solution" / "solution.sol.xml",
                    run_dir / "solution" / "solution.sol.xml")
    for sub in SHARED_DATA_SUBDIRS:
        link_or_replace(crop_dir / "data" / sub, run_dir / "data" / sub)

    # Soil scenario (real file). All crop solutions read data/soil/soil.csv.
    shutil.copyfile(soil_src, run_dir / "data" / "soil" / "soil.csv")

    # CO2 forcing (real file). All crop solutions read data/co2/co2.csv; which
    # source file lands there is what makes an SSP run actually see SSP CO2.
    co2_src = crop_dir / "data" / "co2" / climate.co2_file
    if not co2_src.exists():
        raise FileNotFoundError(co2_src)
    shutil.copyfile(co2_src, run_dir / "data" / "co2" / "co2.csv")

    # Project CSV.
    out_csv = run_dir / "project" / "project.csv"
    if climate.grid == "baseline":
        # Sort location-contiguous (vLocationID, start_date) so the cluster runner can
        # split work on location boundaries — SIMPLACE writes one output file per
        # location, so a location split across invocations would clobber itself. The
        # baseline CSV interleaves each location's recent years at the file's tail.
        df = pd.read_csv(baseline_csv, sep=";")
        df = df.sort_values(["vLocationID", "start_date"]).reset_index(drop=True)
        df["projectid"] = range(1, len(df) + 1)
        df.to_csv(out_csv, sep=";", index=False)   # dynamic per-year IDPL already set
        plan["rows"] = len(df)
    else:
        df = build_hyras_project(repo / cfg["paths"]["point_grid"], baseline_csv, climate)
        df.to_csv(out_csv, sep=";", index=False)
        plan["rows"] = len(df)

    # Templated proj.xml (weather path + divider for this climate).
    (run_dir / "project" / "project.proj.xml").write_text(
        render_proj_xml(proj_template, climate))

    # Cluster config consumed by simplace_runner_cluster.py.
    s = cfg["slurm"]
    cluster_cfg = {"cluster": {
        "exp_name": exp_id,
        "work_dir": str(run_dir),
        "output_dir": "out/",
        "solution": "solution/solution.sol.xml",
        "project": "project/project.proj.xml",
        "input_csv": str(out_csv),
        "mount_data": climate.mount_data,
        "singularity_image": s["singularity_image"],
        "debug": False,
        "testrun": False,
        "num_tasks_per_node": s["num_tasks_per_node"],
        "num_nodes": s["num_nodes"],
        "partition": s["partition"],
        "walltime": s["walltime"],
        "start_line": 1,
    }}
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cluster_cfg, fh, sort_keys=False)

    plan["status"] = "generated"
    plan["submit"] = (f"python simplace/{crop}/simplace_runner_cluster.py "
                      f"{run_dir / 'config.yaml'}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("experiments.yaml")))
    ap.add_argument("--crop")
    ap.add_argument("--climate")
    ap.add_argument("--soil")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-climates", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    registry = build_climate_registry(cfg)

    if args.list_climates:
        print(f"{len(registry)} climate sources:")
        for c in registry.values():
            span = "baseline" if c.grid == "baseline" else f"{c.start}-{c.end}"
            print(f"  {c.id:24s} {c.kind:11s} {span:10s} idpl={c.idpl_rule}")
        return 0

    missing = [n for n in ("crop", "climate", "soil") if not getattr(args, n)]
    if missing:
        ap.error(f"--{', --'.join(missing)} required (or use --list-climates)")

    # Each selector accepts an explicit value, a comma list, or "all".
    def resolve(value: str, universe: list[str], what: str) -> list[str]:
        if value == "all":
            return list(universe)
        picked = [v.strip() for v in value.split(",")]
        bad = [v for v in picked if v not in universe]
        if bad:
            ap.error(f"unknown {what}: {bad}; choose from {universe}")
        return picked

    crops = resolve(args.crop, cfg["crops"], "crop")
    climates = resolve(args.climate, list(registry), "climate")
    soils = resolve(args.soil, cfg["soils"], "soil")

    plans = []
    for crop in crops:
        for clim in climates:
            for soil in soils:
                plan = generate(crop, registry[clim], soil, cfg, args.dry_run)
                plans.append(plan)
                print(f"  [{plan['status']:>9s}] {crop:16s} {clim:22s} {soil:4s}"
                      + (f"  rows={plan['rows']:,}" if "rows" in plan else ""))

    print(f"\n{len(plans)} experiment(s) processed.")

    if not args.dry_run:
        # Emit a single reviewable submit script (one runner invocation per experiment).
        repo = Path(cfg["repo_root"])
        runs_root = repo / "simplace" / crops[0] / cfg["paths"]["runs_subdir"]
        tag = "_".join(sorted(set(climates)))[:40] if len(climates) <= 3 else "batch"
        script = runs_root.parent.parent / "runs_submit" / f"submit_{tag}.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#!/bin/bash",
                 "# Generated by orchestration/generate.py — one SIMPLACE runner per experiment.",
                 "# Each runner submits its own SLURM jobs (num_nodes per config.yaml) and BLOCKS",
                 "# until they finish. Run lines in the background (append ' &') to queue all at once.",
                 "set -euo pipefail", ""]
        for p in plans:
            lines.append(f"python simplace/{p['crop']}/simplace_runner_cluster.py "
                         f"{p['run_dir']}/config.yaml")
        script.write_text("\n".join(lines) + "\n")
        print(f"Submit script written: {script}")
        print(f"Run it with:  bash {script}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
