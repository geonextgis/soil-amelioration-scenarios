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

Three properties of a hist/future project CSV are taken from the crop's own
baseline CSV rather than configured, so every experiment stays comparable with
the baseline it is evaluated against:
  * window length  -> winter_wheat/winter_rapeseed span 2 calendar years
                      (Y-01-01 .. Y+1-12-31), the spring crops 1. See
                      baseline_span_years().
  * point set      -> the 3086 PointIDs with location.csv/fertilizer rows, not
                      all 3099 in point_to_nearest_grid.csv. See baseline_points().
  * last start year-> pulled back by the span so the final window ends inside
                      the climate period.

Usage:
  python orchestration/generate.py --list-climates
  python orchestration/generate.py --crop maize --climate DWD --soil S1
  python orchestration/generate.py --crop maize --climate GFDL-ESM4_ssp370 --soil S2 --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# Repo root derived from this file's location (orchestration/generate.py), so the
# checkout can be moved or cloned anywhere without editing config. Overridable via
# the SOIL_SCENARIOS_ROOT env var or an explicit `repo_root:` in experiments.yaml.
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_root(cfg: dict) -> Path:
    """Repo root: explicit config > SOIL_SCENARIOS_ROOT env var > this file's location.

    A stale absolute `repo_root` is the one failure that does not announce itself —
    if an older copy of the checkout still exists, every run silently reads and
    writes *there*. So an explicitly configured root must actually look like this
    repo, or we refuse it.
    """
    raw = cfg.get("repo_root") or os.environ.get("SOIL_SCENARIOS_ROOT") or "auto"
    if str(raw).strip().lower() in ("", "auto", "none"):
        return REPO_ROOT
    root = Path(raw).expanduser().resolve()
    if not (root / "simplace").is_dir():
        raise SystemExit(
            f"configured repo_root does not look like this repo: {root}\n"
            f"  (no simplace/ inside it).  Set `repo_root: auto` in experiments.yaml "
            f"to derive it from the checkout location ({REPO_ROOT})."
        )
    return root


# --- DWD weather contract (baseline) ---------------------------------------
DWD_WEATHER = "${_DATADIR_}/${vRow}/daily_mean_RES1_C${vColumn}R${vRow}.csv.gz"
DWD_DIVIDER = None  # whitespace/tab -> SIMPLACE self-closing <divider />

# Shared crop input subdirs that are symlinked (everything except soil and co2,
# which are staged per experiment).
SHARED_DATA_SUBDIRS = ["crop", "management", "slim", "soilcnp"]

# Locations included in the generated config_smoke.yaml. Enough to exercise every
# input path (weather, soil, co2, management) on one node in a few minutes.
SMOKE_LOCATIONS = 3

# CO2 forcing. Every solution reads data/co2/co2.csv; the climate decides which of
# the crop's data/co2/*.csv is copied to that name — same pattern as the soil
# scenario -> data/soil/soil.csv. Without this the SSP runs would silently be
# driven by observed CO2, which also has no rows past 2026.
#
# The observed/historical file is co2_mm_historical.csv, NOT co2_mm_observed.csv:
# the raw Mauna Loa record starts 1958-03, but the historical period starts 1951
# and the solution does an exact (CURRENT.YEAR, CURRENT.MONTH) lookup on every
# simulated day — a missing month is a NullPointerException, not a gap. See
# orchestration/build_co2_historical.py, which prepends the Law Dome
# reconstruction for 1951-01 .. 1958-02 and leaves the observed months verbatim.
CO2_OBSERVED = "co2_mm_historical.csv"
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


# --- weather coverage -------------------------------------------------------
# A simulation window that runs past the end of its weather file is not caught by
# anything: SIMPLACE reads until the rows run out and dies with a
# NullPointerException on the first missing day. The nominal period in
# experiments.yaml (and in the HYRAS *filenames*) is therefore not trustworthy on
# its own — CanESM5 and GFDL-ESM4 ship files named `19510101_20141231` that
# actually stop on 2014-11-21. So probe the data and keep only whole windows.
_COVERAGE_CACHE: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}


def weather_file_for(climate: Climate, col: int, row: int) -> Path:
    """Resolve the SIMPLACE weather template to a real path on disk."""
    rel = (climate.weather_path
           .replace("${_DATADIR_}", str(climate.mount_data))
           .replace("${vColumn}", str(col))
           .replace("${vRow}", str(row)))
    return Path(rel)


def probe_coverage(climate: Climate, col: int, row: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date actually present in this climate's weather files.

    All cells of a source share one time axis, so one representative file is
    enough; the result is cached per climate id.
    """
    if climate.id in _COVERAGE_CACHE:
        return _COVERAGE_CACHE[climate.id]

    path = weather_file_for(climate, col, row)
    if not path.exists():
        raise FileNotFoundError(
            f"weather file missing for climate {climate.id}: {path}\n"
            f"  (probed with vColumn={col}, vRow={row})")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        lines = fh.read().splitlines()
    sep = climate.divider or None          # None -> any whitespace
    body = [l for l in lines[1:] if l.strip()]
    first = pd.Timestamp(body[0].split(sep)[0].strip())
    last = pd.Timestamp(body[-1].split(sep)[0].strip())
    _COVERAGE_CACHE[climate.id] = (first, last)
    return first, last


def clamp_to_coverage(df: pd.DataFrame, cov: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[pd.DataFrame, int]:
    """Drop simulation windows not fully inside the weather record.

    Dropped, not truncated: a window cut short never reaches harvest, and the
    yearly output only fires on HarvestManagement.DoHarvest, so a truncated window
    would contribute nothing but would still look like a successful simulation.
    """
    first, last = cov
    keep = (pd.to_datetime(df["start_date"]) >= first) & (pd.to_datetime(df["end_date"]) <= last)
    out = df.loc[keep].copy()
    out["projectid"] = range(1, len(out) + 1)
    return out, int((~keep).sum())


def calendar_drift(climate: Climate, cov: tuple[pd.Timestamp, pd.Timestamp]) -> int:
    """Days between the climate's nominal start and the data's actual start.

    Large drift means a no-leap (365-day) model calendar was re-stamped onto
    consecutive real dates: the offset grows through the record, so a sowing date
    fixed by day-of-year lands progressively earlier in the real season. That is a
    scientific problem in the source data, not something generation can repair —
    it is surfaced so it cannot be missed.
    """
    if climate.grid == "baseline":
        return 0
    return int(abs((cov[0] - pd.Timestamp(f"{climate.start}-01-01")).days))


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


def baseline_span_years(baseline_csv: Path) -> int:
    """Calendar years a simulation window spans, read off the crop's baseline CSV.

    Winter crops are sown in autumn of year Y and harvested in summer of Y+1, so
    their baseline windows run Y-01-01 .. (Y+1)-12-31 (span 1); spring crops fit
    in one calendar year (span 0). Derived rather than hard-coded so the crop list
    can grow without a second place to update — and because getting it wrong is
    silent: a winter crop cut off at Dec 31 of the sowing year never reaches
    harvest, and the yearly output only fires on HarvestManagement.DoHarvest, so
    the run "succeeds" with header-only output files.
    """
    df = pd.read_csv(baseline_csv, sep=";", usecols=["start_date", "end_date"])
    spans = (pd.to_datetime(df["end_date"]).dt.year
             - pd.to_datetime(df["start_date"]).dt.year).unique()
    if len(spans) != 1:
        raise RuntimeError(f"{baseline_csv} mixes window lengths: {sorted(spans)}")
    return int(spans[0])


def baseline_points(baseline_csv: Path) -> set:
    """PointIDs the crop actually has inputs for.

    point_to_nearest_grid.csv carries 3099 points, but location.csv (Latitude,
    Altitude) and fertilizer_<crop>.csv only cover the 3086 in the baseline
    project file. Simulating the other 13 would fail for want of a location row,
    and would also make the baseline and hist/future point sets non-comparable.
    """
    return set(pd.read_csv(baseline_csv, sep=";", usecols=["vLocationID"])["vLocationID"])


def build_hyras_project(point_grid_csv: Path, baseline_csv: Path,
                        climate: Climate) -> pd.DataFrame:
    """One row per (point, start year) over the climate period; vIDPL = NUTS median.

    Window length and point set both come from the crop's baseline project CSV, so
    a hist/future experiment simulates the same crop cycle over the same sites as
    the baseline it will be compared against.
    """
    pg = pd.read_csv(point_grid_csv)
    per_nuts, global_median = nuts_median_idpl(baseline_csv)
    span = baseline_span_years(baseline_csv)

    keep = baseline_points(baseline_csv)
    pg = pg[pg["PointID"].isin(keep)].reset_index(drop=True)
    missing = keep - set(pg["PointID"])
    if missing:
        raise RuntimeError(f"{len(missing)} baseline points absent from "
                           f"{point_grid_csv.name}: {sorted(missing)[:10]}")

    cols = pg["nearest_grid_id"].map(parse_grid)
    pg = pg.assign(vColumn=[c for c, _ in cols], vRow=[r for _, r in cols])
    pg["vIDPL"] = pg["NUTS_ID"].map(per_nuts).fillna(global_median).astype(int)

    # Last window must end inside the climate period, so a span-1 crop stops one
    # start-year early rather than running off the end of the weather file.
    years = range(climate.start, climate.end - span + 1)
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
        "end_date": (rep["year"] + span).astype(str) + "-12-31",
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
    repo = resolve_repo_root(cfg)
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

    # Probe the weather before doing any work: it is cached per climate, it fails
    # loudly if the source is missing, and --dry-run should surface a bad calendar.
    if climate.grid == "baseline":
        head = pd.read_csv(baseline_csv, sep=";", usecols=["vColumn", "vRow"], nrows=1)
        probe_col, probe_row = int(head["vColumn"][0]), int(head["vRow"][0])
    else:
        head = pd.read_csv(repo / cfg["paths"]["point_grid"],
                           usecols=["nearest_grid_id"], nrows=1)
        probe_col, probe_row = parse_grid(head["nearest_grid_id"][0])
    cov = probe_coverage(climate, probe_col, probe_row)
    plan["coverage"] = f"{cov[0].date()}..{cov[1].date()}"
    plan["drift_days"] = calendar_drift(climate, cov)

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
    else:
        df = build_hyras_project(repo / cfg["paths"]["point_grid"], baseline_csv, climate)

    # Drop windows the weather record cannot cover (see clamp_to_coverage).
    df, dropped = clamp_to_coverage(df, cov)
    if df.empty:
        raise RuntimeError(f"{exp_id}: no simulation window fits inside "
                           f"{cov[0].date()}..{cov[1].date()}")
    df.to_csv(out_csv, sep=";", index=False)

    span = baseline_span_years(baseline_csv)
    plan["rows"] = len(df)
    plan["points"] = df["vLocationID"].nunique()
    plan["window_years"] = span + 1
    plan["dropped"] = dropped
    plan["period"] = (f"{df['start_date'].min()[:4]}-{df['end_date'].max()[:4]} "
                      f"({plan['window_years']}-year windows)")

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

    # Smoke config: same inputs, first few locations only, one node. Generated
    # rather than hand-written so it cannot drift from the experiment it tests —
    # the previous hand-maintained config_smoke.yaml still pointed at a checkout
    # that had moved, which fails silently because the old tree still exists.
    # exp_name is prefixed so a smoke run cannot overwrite real output.
    smoke_locs = df["vLocationID"].unique()[:SMOKE_LOCATIONS]
    smoke_cfg = {"cluster": dict(cluster_cfg["cluster"],
                                 exp_name=f"SMOKE_{exp_id}",
                                 num_nodes=1,
                                 num_tasks_per_node=2,
                                 walltime="00:20:00",
                                 end_line=int(df["vLocationID"].isin(smoke_locs).sum()))}
    with open(run_dir / "config_smoke.yaml", "w") as fh:
        yaml.safe_dump(smoke_cfg, fh, sort_keys=False)

    plan["status"] = "generated"
    plan["submit"] = (f"python simplace/{crop}/simplace_runner_cluster.py "
                      f"{run_dir / 'config.yaml'}")
    plan["smoke"] = (f"python simplace/{crop}/simplace_runner_cluster.py "
                     f"{run_dir / 'config_smoke.yaml'}")
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
                extra = ""
                if "rows" in plan:
                    extra = (f"  rows={plan['rows']:,}  points={plan['points']:,}"
                             f"  {plan['period']}")
                    if plan["dropped"]:
                        extra += f"  dropped={plan['dropped']:,}"
                print(f"  [{plan['status']:>9s}] {crop:16s} {clim:22s} {soil:4s}{extra}")

    print(f"\n{len(plans)} experiment(s) processed.")

    # Calendar drift is a property of the source data, not of this run, so report it
    # once at the end rather than burying it in the per-experiment lines.
    drifted = sorted({(p["climate"], p["coverage"], p["drift_days"])
                      for p in plans if p.get("drift_days", 0) > 15})
    if drifted:
        print("\nWARNING — weather does not start where its filename claims:")
        for cid, coverage, days in drifted:
            print(f"  {cid:24s} covers {coverage}  ({days} days off nominal start)")
        print("  A no-leap model calendar re-stamped onto real dates drifts through\n"
              "  the record, so a day-of-year sowing rule lands progressively earlier\n"
              "  in the real season. Windows past the data end were dropped, but the\n"
              "  seasonal misalignment cannot be fixed here — confirm the intended\n"
              "  handling with whoever produced the dataset before using these runs.")

    if not args.dry_run:
        repo = resolve_repo_root(cfg)
        submit_dir = repo / "simplace" / "runs_submit"
        submit_dir.mkdir(parents=True, exist_ok=True)

        # Name the script after its content, not its selection: the old scheme
        # collapsed every selection wider than 3 climates to "submit_batch.sh", so
        # successive full-matrix calls silently overwrote each other's script.
        configs = [f"{p['run_dir']}/config.yaml" for p in plans]
        digest = hashlib.sha1("\n".join(sorted(configs)).encode()).hexdigest()[:8]
        label = "-".join(x for x in (
            crops[0] if len(crops) == 1 else f"{len(crops)}crops",
            climates[0] if len(climates) == 1 else f"{len(climates)}clim",
            soils[0] if len(soils) == 1 else f"{len(soils)}soil") if x)
        script = submit_dir / f"submit_{label}_{digest}.sh"

        # Bounded concurrency. Each runner takes num_nodes nodes and BLOCKS until
        # they finish, so running all of them at once would oversubscribe the
        # partition and leave one squeue-poller per experiment hammering slurmctld.
        total = int(cfg["slurm"].get("cluster_nodes", 100))
        per_exp = int(cfg["slurm"]["num_nodes"])
        par = max(1, total // per_exp)

        # "<crop> <config>" per line: the crop cannot be recovered from an absolute
        # config path by position, so carry it explicitly.
        listing = submit_dir / f"{script.stem}.list"
        listing.write_text("".join(f"{p['crop']} {p['run_dir']}/config.yaml\n"
                                   for p in plans))

        script.write_text(f"""#!/bin/bash
# Generated by orchestration/generate.py — {len(plans)} experiment(s).
#
# Each line of {listing.name} is one experiment. The runner submits that
# experiment's SLURM jobs ({per_exp} nodes each) and blocks until they finish, so
# they are driven {par} at a time: {par} x {per_exp} = {par * per_exp} nodes, against a
# partition of {total}. Do NOT background every line instead — that oversubscribes
# the partition and starts one squeue poller per experiment.
set -euo pipefail
cd "{repo}"

# Preflight: a config that no longer exists means the run dir was regenerated
# under a different name or removed. Fail before submitting anything.
while read -r crop cfgfile; do
  [ -f "$cfgfile" ] || {{ echo "missing config: $cfgfile" >&2; exit 1; }}
done < "{listing}"

xargs -P {par} -L1 -a "{listing}" \\
  sh -c 'python simplace/"$0"/simplace_runner_cluster.py "$1"'
""")
        script.chmod(0o755)
        print(f"\nSubmit script: {script}")
        print(f"  experiment list: {listing}")
        print(f"  concurrency:     {par} experiment(s) x {per_exp} nodes = "
              f"{par * per_exp}/{total} nodes")
        print(f"  run it with:     bash {script}")
        if plans:
            print(f"  smoke test one:  {plans[0]['smoke']}")

        # Submit scripts are cheap to regenerate but expensive to trust when stale.
        # New scripts carry a .list manifest; older ones name their configs inline.
        stale = []
        for s in sorted(submit_dir.glob("submit_*.sh")):
            if s == script:
                continue
            manifest = s.with_suffix(".list")
            refs = ([l.split()[-1] for l in manifest.read_text().splitlines() if l.strip()]
                    if manifest.exists()
                    else re.findall(r"/\S+/config\.yaml", s.read_text()))
            gone = [r for r in refs if not Path(r).exists()]
            if gone:
                stale.append((s, len(gone), len(refs)))
        if stale:
            print("\nStale submit scripts (reference run dirs that no longer exist):")
            for s, gone, total_refs in stale:
                print(f"  {s}  ({gone}/{total_refs} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
