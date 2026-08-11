#!/usr/bin/env python
"""Model-side plumbing shared by every calibration run.

This layer knows how to *run the model and read it back*, and nothing about how
parameters are chosen:

    stage an isolated run dir -> edit crop.xml -> invoke SIMPLACE -> read out/

The decision layer sits on top (``calib_common.py`` for the mechanics,
``calibrate.py`` for one iteration, ``agents/`` for the local-LLM loop). Keeping
the split means the objective is computed by identical code no matter who
proposed the parameters.

Isolation contract
------------------
Calibration never touches the production crop inputs. Everything happens under
``simplace/<crop>/runs_optim/<run_subdir>/`` with the mutated
``data/crop/crop.xml`` as a *copy*; shared read-only inputs are symlinked. So an
interrupted or failed iteration can never corrupt
``simplace/<crop>/data/crop/crop.xml``.
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lxml import etree

OPTIM_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = OPTIM_DIR / "config.yaml"
# Repo root derived from this file's location (optimization/common.py) so the
# checkout works wherever it lives. See resolve_repo_root() below.
REPO_ROOT = OPTIM_DIR.parent

# Shared crop inputs that are read-only during calibration -> symlink, never copy.
SYMLINK_DATA_SUBDIRS = ["slim", "soilcnp"]
# CO2 is staged, not symlinked: every solution reads data/co2/co2.csv and the source
# file varies by climate. Calibration always runs on DWD observations.
CO2_SOURCE = "co2_mm_observed.csv"
# Read-only files inside data/management (the per-location tables are staged).
SYMLINK_MANAGEMENT_FILES = ["management.xml", "fertilizer_composition.xml"]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    """Everything one model run needs, with every path already absolute.

    A calibration stage may need more than one run to be scored: the joint
    LAI+yield stage evaluates the canopy on the GLASS-LAI point set and the yield
    on the district point set, from the same ``crop.xml``. Each of those is one
    ``RunSpec``, distinguished by ``view``.
    """

    crop: str
    crop_name: str
    target: str
    view: str
    repo_root: Path
    crop_dir: Path
    run_dir: Path
    exp_name: str
    project_src: Path
    inputs: dict[str, Path]
    subset: dict
    mount_data: str
    slurm: dict
    device: str = "cluster"
    # Dry-matter fraction applied to *observed* yield. Observed potato yield is
    # fresh tuber weight while the model reports dry matter, so comparing them
    # raw would make the yield loss chase a units mismatch. 1.0 = already DM.
    dm_fraction: float = 1.0
    # Lets a consumer score outputs from some *other* run (e.g. the evaluation
    # notebook pointing at a finished scenario run) without restaging anything.
    out_override: Path | None = None

    @property
    def crop_xml(self) -> Path:
        return self.run_dir / "data" / "crop" / "crop.xml"

    @property
    def out_dir(self) -> Path:
        return self.out_override or (self.run_dir / "out" / self.exp_name)

    @property
    def project_csv(self) -> Path:
        return self.run_dir / "project" / "project.csv"

    @property
    def run_config(self) -> Path:
        return self.run_dir / "config.yaml"


def resolve_repo_root(cfg: dict) -> Path:
    """Repo root: explicit config > SOIL_SCENARIOS_ROOT env var > this file's location.

    A stale absolute `repo_root` is the one failure that does not announce itself —
    if an older copy of the checkout still exists, calibration silently stages run
    dirs and reads crop.xml *there*. So an explicit root must look like this repo.
    """
    raw = cfg.get("repo_root") or os.environ.get("SOIL_SCENARIOS_ROOT") or "auto"
    if str(raw).strip().lower() in ("", "auto", "none"):
        return REPO_ROOT
    root = Path(raw).expanduser().resolve()
    if not (root / "simplace").is_dir():
        raise SystemExit(
            f"configured repo_root does not look like this repo: {root}\n"
            f"  (no simplace/ inside it).  Set `repo_root: auto` in the optimization "
            f"config to derive it from the checkout location ({REPO_ROOT})."
        )
    return root


def _merge(base: dict, extra: dict) -> dict:
    """Recursive dict merge; `extra` wins. Used for per-crop target overrides."""
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def target_views(cfg: dict, crop: str, target: str) -> dict[str, dict]:
    """``{view name: view config}`` for one target, with per-crop overrides merged.

    A target declares one or more views under ``targets.<target>.views``. Each is
    a complete run definition (project table, input profile, output namespace,
    subset); the target-level ``run_subdir`` is folded into each of them so a
    view config is self-contained from here on.
    """
    if target not in (cfg.get("targets") or {}):
        raise SystemExit(f"unknown target {target!r}; config has {sorted(cfg.get('targets') or {})}")
    tcfg = cfg["targets"][target]
    crop_cfg = (cfg.get("crops") or {}).get(crop) or {}
    override = ((crop_cfg.get("targets") or {}).get(target) or {}).get("views") or {}

    views = tcfg.get("views") or {}
    if not views:
        raise SystemExit(f"target {target!r} declares no views")
    out = {}
    for name, vcfg in views.items():
        merged = _merge(vcfg, override.get(name, {}))
        merged.setdefault("run_subdir", tcfg.get("run_subdir", f"calib_{target}"))
        out[name] = merged
    return out


def make_run(cfg: dict, crop: str, target: str, view: str, vcfg: dict,
             device: str = "cluster") -> RunSpec:
    """One run of one view, with every path resolved against the config."""
    if crop not in (cfg.get("crops") or {}):
        raise SystemExit(f"unknown crop {crop!r}; config has {sorted(cfg.get('crops') or {})}")
    crop_cfg = cfg["crops"][crop] or {}

    repo_root = resolve_repo_root(cfg)
    crop_dir = repo_root / "simplace" / crop
    if not crop_dir.is_dir():
        raise SystemExit(f"crop dir not found: {crop_dir}")

    def fmt(p: str) -> str:
        return p.format(crop=crop)

    # Staged per-location tables, with a graceful fallback when a crop has no
    # *_LAI variants (only the LAI-calibrated crops carry those).
    profile_name = vcfg.get("inputs", "production")
    profile = cfg["input_profiles"][profile_name]
    inputs: dict[str, Path] = {}
    for role, rel in profile.items():
        path = crop_dir / fmt(rel)
        if not path.exists() and profile_name != "production":
            fallback = crop_dir / fmt(cfg["input_profiles"]["production"][role])
            print(f"  note: {path.name} missing, falling back to {fallback.name}")
            path = fallback
        if not path.exists():
            raise SystemExit(f"required input missing: {path}")
        inputs[role] = path

    project_src = crop_dir / fmt(vcfg["project_csv"])
    if not project_src.exists():
        raise SystemExit(f"project table missing: {project_src}")

    return RunSpec(
        crop=crop,
        crop_name=crop_cfg.get("crop_name", crop),
        target=target,
        view=view,
        repo_root=repo_root,
        crop_dir=crop_dir,
        run_dir=crop_dir / "runs_optim" / vcfg["run_subdir"] / view,
        exp_name=vcfg["exp_name"],
        project_src=project_src,
        inputs=inputs,
        subset=vcfg.get("subset") or {},
        mount_data=cfg["climate"]["mount_data"],
        slurm=cfg["slurm"],
        device=device,
        dm_fraction=float(crop_cfg.get("dm_fraction", 1.0)),
    )


def load_spec(config_path: Path, crop: str, target: str, view: str | None = None,
              device: str = "cluster") -> RunSpec:
    """One run, straight from a config file. ``view`` defaults to the first one."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    views = target_views(cfg, crop, target)
    if view is None:
        view = next(iter(views))
    if view not in views:
        raise SystemExit(f"target {target!r} has no view {view!r}; have {sorted(views)}")
    return make_run(cfg, crop, target, view, views[view], device=device)


# ---------------------------------------------------------------------------
# Calibration run dir
# ---------------------------------------------------------------------------
def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def build_project_csv(spec: RunSpec) -> int:
    """Write the calibration project table: optional subset, location-contiguous.

    SIMPLACE writes one output file per location into a shared dir, so the cluster
    runner may only split work on location boundaries — which requires each
    location to be one contiguous block. The baseline ``project_<crop>.csv``
    appends each location's recent years at the file tail, so sorting here is
    load-bearing, not cosmetic.
    """
    df = pd.read_csv(spec.project_src, sep=";")

    y0, y1 = spec.subset.get("year_start"), spec.subset.get("year_end")
    if y0 or y1:
        years = pd.to_datetime(df["start_date"]).dt.year
        df = df[years.between(y0 or years.min(), y1 or years.max())]

    n_loc = spec.subset.get("n_locations")
    if n_loc:
        locs = np.sort(df["vLocationID"].unique())
        if n_loc < len(locs):
            # Evenly spaced pick -> keeps the geographic spread of the full set.
            keep = locs[np.linspace(0, len(locs) - 1, n_loc, dtype=int)]
            df = df[df["vLocationID"].isin(keep)]

    if df.empty:
        raise SystemExit(f"subset for {spec.target} selected 0 rows from {spec.project_src}")

    df = df.sort_values(["vLocationID", "start_date"]).reset_index(drop=True)
    df["projectid"] = range(1, len(df) + 1)
    df.to_csv(spec.project_csv, sep=";", index=False)
    return len(df)


def build_run_dir(spec: RunSpec, rebuild: bool = False) -> dict:
    """Stage the isolated calibration run dir. Idempotent."""
    sys.path.insert(0, str(spec.repo_root / "orchestration"))
    import generate as orch  # reuse the proj.xml templating + DWD weather contract

    if rebuild and spec.run_dir.exists():
        shutil.rmtree(spec.run_dir)

    # data/co2 used to be symlinked to the crop's shared dir; it is staged now.
    # Without this the staging copy below would write *through* the old symlink
    # and contaminate the shared source directory.
    stale_co2 = spec.run_dir / "data" / "co2"
    if stale_co2.is_symlink():
        stale_co2.unlink()

    (spec.run_dir / "project").mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "data" / "soil").mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "data" / "co2").mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "data" / "management").mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "solution").mkdir(parents=True, exist_ok=True)
    (spec.out_dir).mkdir(parents=True, exist_ok=True)

    _link(spec.crop_dir / "solution" / "solution.sol.xml",
          spec.run_dir / "solution" / "solution.sol.xml")
    for sub in SYMLINK_DATA_SUBDIRS:
        src = spec.crop_dir / "data" / sub
        if src.is_dir():
            _link(src, spec.run_dir / "data" / sub)
    for name in SYMLINK_MANAGEMENT_FILES:
        src = spec.crop_dir / "data" / "management" / name
        if src.exists():
            _link(src, spec.run_dir / "data" / "management" / name)

    # crop/ is a real copy: crop.xml is rewritten every trial and must not alias
    # the production file.
    crop_data = spec.run_dir / "data" / "crop"
    if not crop_data.is_dir() or crop_data.is_symlink():
        if crop_data.is_symlink():
            crop_data.unlink()
        shutil.copytree(spec.crop_dir / "data" / "crop", crop_data, dirs_exist_ok=True)

    # Per-location tables staged under the canonical names the solution reads.
    shutil.copyfile(spec.inputs["soil"], spec.run_dir / "data" / "soil" / "soil.csv")
    co2_src = spec.crop_dir / "data" / "co2" / CO2_SOURCE
    if not co2_src.exists():
        raise SystemExit(f"CO2 forcing missing: {co2_src}")
    shutil.copyfile(co2_src, spec.run_dir / "data" / "co2" / "co2.csv")
    shutil.copyfile(spec.inputs["location"], spec.run_dir / "data" / "management" / "location.csv")
    shutil.copyfile(spec.inputs["fertilizer"],
                    spec.run_dir / "data" / "management" / f"fertilizer_{spec.crop}.csv")

    rows = build_project_csv(spec)

    dwd = orch.Climate(
        id="DWD", kind="baseline", mount_data=spec.mount_data,
        weather_path=orch.DWD_WEATHER, divider=orch.DWD_DIVIDER,
        start=0, end=0, idpl_rule="dynamic", grid="baseline",
        co2_file=CO2_SOURCE,
    )
    template = (spec.crop_dir / "project" / "project.proj.xml").read_text()
    (spec.run_dir / "project" / "project.proj.xml").write_text(
        orch.render_proj_xml(template, dwd))

    s = spec.slurm
    spec.run_config.write_text(yaml.safe_dump({"cluster": {
        "exp_name": spec.exp_name,
        "work_dir": str(spec.run_dir),
        "output_dir": "out/",
        "solution": "solution/solution.sol.xml",
        "project": "project/project.proj.xml",
        "input_csv": str(spec.project_csv),
        "mount_data": spec.mount_data,
        "singularity_image": s["singularity_image"],
        "debug": s.get("debug", False),
        "testrun": False,
        "num_tasks_per_node": s["num_tasks_per_node"],
        "num_nodes": s["num_nodes"],
        "partition": s["partition"],
        "walltime": s["walltime"],
        "start_line": 1,
    }}, sort_keys=False))

    locs = pd.read_csv(spec.project_csv, sep=";", usecols=["vLocationID"])["vLocationID"].nunique()
    return {"run_dir": str(spec.run_dir), "rows": rows, "locations": int(locs)}


# ---------------------------------------------------------------------------
# crop.xml parameter editing
# ---------------------------------------------------------------------------
def _crop_params(root, crop_name: str, param_id: str):
    return root.xpath(
        f"//crop[parameter[@id='CropName' and text()='{crop_name}']]"
        f"/parameter[@id='{param_id}']"
    )


def bounds_of(space: dict) -> dict[str, tuple[float, float]]:
    """Flatten a parameter space to ``{flat name: (low, high)}``.

    Table elements are addressed as ``<param>_<index>``, matching the flattening
    used throughout the decision layer (``calib_common.flatten``).
    """
    out: dict[str, tuple[float, float]] = {}
    for pid, spec in (space.get("single_value_params") or {}).items():
        out[pid] = (spec["low"], spec["high"])
    for pid, spec in (space.get("multi_value_params") or {}).items():
        for i, b in enumerate(spec["values"]):
            out[f"{pid}_{i}"] = (b["low"], b["high"])
    return out


def check_within_bounds(values: dict, space: dict) -> list[str]:
    """Names of values that fall outside their configured range."""
    limits = bounds_of(space)
    return [k for k, v in values.items()
            if k in limits and not (limits[k][0] <= v <= limits[k][1])]


def apply_parameters(xml_path: Path, crop_name: str, drawn: dict) -> None:
    """Write a parameter set into the calibration crop.xml."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    for pid, value in drawn.items():
        params = _crop_params(root, crop_name, pid)
        if not params:
            raise SystemExit(f"parameter {pid!r} not found for crop {crop_name!r} in {xml_path}")
        for p in params:
            if isinstance(value, list):
                existing = p.findall("value")
                for i, v in enumerate(value):
                    if i < len(existing):
                        existing[i].text = str(v)
                    else:
                        etree.SubElement(p, "value").text = str(v)
                for extra in existing[len(value):]:
                    p.remove(extra)
            else:
                p.text = str(value)

    tree.write(str(xml_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simplace(spec: RunSpec, iteration: int, log_path: Path | None = None) -> None:
    """Run SIMPLACE once for the current crop.xml. Blocks until jobs finish.

    ``log_path`` says where the driver's stdout/stderr goes; ``calibrate.py``
    keeps it beside the iteration it belongs to.
    """
    # Stale per-location outputs from the previous iteration would silently blend
    # into this one's loss (a location dropped by SLURM keeps its old file), so
    # start each run from an empty output dir.
    for sub in ("daily", "yearly"):
        shutil.rmtree(spec.out_dir / sub, ignore_errors=True)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    runner = ("simplace_runner_cluster.py" if spec.device == "cluster"
              else "simplace_runner.py")
    log = Path(log_path) if log_path else spec.run_dir / "logs" / f"simplace_{iteration}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [sys.executable, str(spec.crop_dir / runner), str(spec.run_config)],
        capture_output=True, text=True, cwd=str(spec.crop_dir),
    )
    log.write_text(f"$ {runner} {spec.run_config}\n\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    if proc.returncode != 0:
        raise RuntimeError(f"SIMPLACE run failed (rc={proc.returncode}); see {log}")

    produced = await_outputs(spec.out_dir, "yearly")
    if produced == 0:
        # The cluster runner reports "All jobs completed" even when SIMPLACE died
        # immediately (empty squeue), so an empty out/ is the real failure signal.
        raise RuntimeError(f"SIMPLACE produced no output; see {log}")


def await_outputs(out_dir: Path, kind: str, timeout: float = 120.0,
                  interval: float = 3.0) -> int:
    """Number of output files, waiting out the shared filesystem's metadata cache.

    ``run_simplace`` deletes ``out/<kind>/`` locally and the compute nodes recreate
    it remotely. On BeeGFS the login node can keep serving a stale listing of that
    directory for a few seconds after ``squeue`` goes empty, so a single glob taken
    the instant the jobs finish reports zero files for a run that in fact
    succeeded — which is indistinguishable from a real failure. A short run (a
    small subset, an idle partition) lands inside that window; a long one never
    does, which is why this only bites the fast cases.

    Polls until files appear or ``timeout`` elapses, re-reading the parent
    directory each time to force a fresh lookup.
    """
    target = Path(out_dir) / kind
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.listdir(out_dir)          # drop the cached listing of the parent
        except OSError:
            pass
        found = len(glob(str(target / "*.csv")))
        if found or time.monotonic() >= deadline:
            return found
        time.sleep(interval)


def read_outputs(out_dir: Path, kind: str, max_workers: int = 32,
                 limit: int | None = None) -> pd.DataFrame:
    """Concatenate every per-location output CSV of one kind ('daily'|'yearly').

    ``limit`` reads an evenly spaced sample of that many locations instead of all
    of them — daily output for a full 3,000-location run is tens of millions of
    rows, which is fine for a loss but not for an interactive notebook.
    """
    paths = sorted(glob(str(out_dir / kind / "*.csv")))
    if not paths:
        raise RuntimeError(f"no {kind} outputs in {out_dir / kind}")
    if limit and limit < len(paths):
        idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[i] for i in idx]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        frames = list(ex.map(lambda p: pd.read_csv(p, delimiter=";"), paths))
    return pd.concat(frames, ignore_index=True)


def discover_runs(crop_dir: Path) -> dict[str, Path]:
    """Every output directory that holds results for a crop.

    Returns ``{label: out_dir}`` — scenario runs keep their experiment id
    (``DWD__S1``), calibration runs are prefixed (``optim:calib_growth/lai``).
    A calibration stage keeps one run dir per view, so ``runs_optim`` is searched
    one level deeper than ``runs``.
    """
    found: dict[str, Path] = {}

    def collect(run: Path, label: str) -> bool:
        out_root = run / "out"
        if not out_root.is_dir():
            return False
        hit = False
        for exp in sorted(out_root.iterdir()):
            if exp.is_dir() and (any(exp.glob("yearly/*.csv")) or any(exp.glob("daily/*.csv"))):
                found[label] = exp
                hit = True
        return hit

    for run in sorted((crop_dir / "runs").iterdir()) if (crop_dir / "runs").is_dir() else []:
        if run.is_dir():
            collect(run, run.name)
            # A scenario run is labelled by its experiment id, not its dir name.
            for exp in sorted((run / "out").iterdir()) if (run / "out").is_dir() else []:
                if exp.is_dir() and (any(exp.glob("yearly/*.csv")) or any(exp.glob("daily/*.csv"))):
                    found.pop(run.name, None)
                    found[exp.name] = exp

    base = crop_dir / "runs_optim"
    for run in sorted(base.iterdir()) if base.is_dir() else []:
        if not run.is_dir():
            continue
        if not collect(run, f"optim:{run.name}"):
            for view in sorted(run.iterdir()):
                if view.is_dir():
                    collect(view, f"optim:{run.name}/{view.name}")
    return found


