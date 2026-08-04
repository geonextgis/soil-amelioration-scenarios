#!/usr/bin/env python
"""Shared plumbing for the single-trial SIMPLACE calibration scripts.

The three ``optimize_*.py`` scripts each own one thing: the *loss*. Everything
else — resolving config, staging an isolated calibration run dir, drawing a
parameter set, editing ``crop.xml``, invoking SIMPLACE, persisting the study,
reporting — lives here.

One script invocation = one trial:

    draw params -> write crop.xml -> run SIMPLACE -> read out/ -> loss -> tell study

State lives in ``optimization/studies/<crop>__<target>.db`` (Optuna + SQLite), so
each invocation resumes the previous history and proposes a *better* parameter
set. That is what makes the scripts loop-friendly: re-running the same command is
the whole iteration protocol, no orchestration state to thread through.

Isolation contract
------------------
Calibration never touches the production crop inputs. Everything happens under
``simplace/<crop>/runs_optim/<target>/`` with the mutated ``data/crop/crop.xml``
as a *copy*; shared read-only inputs are symlinked. So an interrupted or bad
trial can never corrupt ``simplace/<crop>/data/crop/crop.xml``.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from lxml import etree

OPTIM_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = OPTIM_DIR / "config.yaml"

# Shared crop inputs that are read-only during calibration -> symlink, never copy.
SYMLINK_DATA_SUBDIRS = ["slim", "soilcnp", "co2"]
# Read-only files inside data/management (the per-location tables are staged).
SYMLINK_MANAGEMENT_FILES = ["management.xml", "fertilizer_composition.xml"]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    """Everything one calibration trial needs, with every path already absolute."""

    crop: str
    crop_name: str
    target: str
    repo_root: Path
    crop_dir: Path
    run_dir: Path
    exp_name: str
    project_src: Path
    inputs: dict[str, Path]
    parameters: dict
    subset: dict
    mount_data: str
    slurm: dict
    optuna_cfg: dict
    device: str = "cluster"
    # Dry-matter fraction applied to *observed* yield. Observed potato yield is
    # fresh tuber weight while the model reports dry matter, so comparing them
    # raw would make the yield loss chase a units mismatch. 1.0 = already DM.
    dm_fraction: float = 1.0

    @property
    def crop_xml(self) -> Path:
        return self.run_dir / "data" / "crop" / "crop.xml"

    @property
    def out_dir(self) -> Path:
        return self.run_dir / "out" / self.exp_name

    @property
    def project_csv(self) -> Path:
        return self.run_dir / "project" / "project.csv"

    @property
    def run_config(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def study_name(self) -> str:
        return f"{self.crop}__{self.target}"

    @property
    def storage(self) -> str:
        db = OPTIM_DIR / "studies" / f"{self.study_name}.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db}"

    @property
    def results_dir(self) -> Path:
        d = OPTIM_DIR / "results" / self.study_name
        d.mkdir(parents=True, exist_ok=True)
        return d


def _merge(base: dict, extra: dict) -> dict:
    """Recursive dict merge; `extra` wins. Used for per-crop target overrides."""
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_spec(config_path: Path, crop: str, target: str, device: str = "cluster") -> RunSpec:
    cfg = yaml.safe_load(Path(config_path).read_text())

    if crop not in cfg["crops"]:
        raise SystemExit(f"unknown crop {crop!r}; config has {sorted(cfg['crops'])}")
    if target not in cfg["targets"]:
        raise SystemExit(f"unknown target {target!r}; config has {sorted(cfg['targets'])}")

    crop_cfg = cfg["crops"][crop] or {}
    tcfg = _merge(cfg["targets"][target], (crop_cfg.get("targets") or {}).get(target, {}))

    repo_root = Path(cfg["repo_root"]).resolve()
    crop_dir = repo_root / "simplace" / crop
    if not crop_dir.is_dir():
        raise SystemExit(f"crop dir not found: {crop_dir}")

    def fmt(p: str) -> str:
        return p.format(crop=crop)

    # Staged per-location tables, with a graceful fallback when a crop has no
    # *_LAI variants (only the LAI-calibrated crops carry those).
    profile_name = tcfg.get("inputs", "production")
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

    project_src = crop_dir / fmt(tcfg["project_csv"])
    if not project_src.exists():
        raise SystemExit(f"project table missing: {project_src}")

    return RunSpec(
        crop=crop,
        crop_name=crop_cfg.get("crop_name", crop),
        target=target,
        repo_root=repo_root,
        crop_dir=crop_dir,
        run_dir=crop_dir / "runs_optim" / target,
        exp_name=tcfg["exp_name"],
        project_src=project_src,
        inputs=inputs,
        parameters=tcfg.get("parameters") or {},
        subset=tcfg.get("subset") or {},
        mount_data=cfg["climate"]["mount_data"],
        slurm=cfg["slurm"],
        optuna_cfg=cfg.get("optuna", {}),
        device=device,
        dm_fraction=float(crop_cfg.get("dm_fraction", 1.0)),
    )


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

    (spec.run_dir / "project").mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "data" / "soil").mkdir(parents=True, exist_ok=True)
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
    shutil.copyfile(spec.inputs["location"], spec.run_dir / "data" / "management" / "location.csv")
    shutil.copyfile(spec.inputs["fertilizer"],
                    spec.run_dir / "data" / "management" / f"fertilizer_{spec.crop}.csv")

    rows = build_project_csv(spec)

    dwd = orch.Climate(
        id="DWD", kind="baseline", mount_data=spec.mount_data,
        weather_path=orch.DWD_WEATHER, divider=orch.DWD_DIVIDER,
        start=0, end=0, idpl_rule="dynamic", grid="baseline",
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


def read_current_values(xml_path: Path, crop_name: str, space: dict) -> dict:
    """Current crop.xml values, flattened to Optuna's parameter naming."""
    root = etree.parse(str(xml_path)).getroot()
    out: dict[str, float | int] = {}

    for pid, spec in (space.get("single_value_params") or {}).items():
        for p in _crop_params(root, crop_name, pid):
            if len(p) == 0 and p.text:
                out[pid] = int(float(p.text)) if spec["type"] == "int" else float(p.text)
    for pid, spec in (space.get("multi_value_params") or {}).items():
        for p in _crop_params(root, crop_name, pid):
            for i, v in enumerate(p.findall("value")):
                if i < len(spec["values"]):
                    out[f"{pid}_{i}"] = int(float(v.text)) if spec["type"] == "int" else float(v.text)
    return out


def bounds_of(space: dict) -> dict[str, tuple[float, float]]:
    """Flatten a parameter space to {optuna param name: (low, high)}."""
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


def suggest_parameters(trial: optuna.Trial, space: dict) -> dict:
    """Draw one parameter set. Returns {param_id: scalar | [values]}."""
    drawn: dict[str, object] = {}

    for pid, spec in (space.get("single_value_params") or {}).items():
        if spec["type"] == "int":
            drawn[pid] = trial.suggest_int(pid, spec["low"], spec["high"])
        else:
            drawn[pid] = round(trial.suggest_float(pid, spec["low"], spec["high"]),
                               spec.get("precision", 4))

    for pid, spec in (space.get("multi_value_params") or {}).items():
        vals = []
        for i, b in enumerate(spec["values"]):
            if spec["type"] == "int":
                vals.append(trial.suggest_int(f"{pid}_{i}", b["low"], b["high"]))
            else:
                vals.append(round(trial.suggest_float(f"{pid}_{i}", b["low"], b["high"]),
                                  spec.get("precision", 4)))
        drawn[pid] = vals

    return drawn


def apply_parameters(xml_path: Path, crop_name: str, drawn: dict) -> None:
    """Write a drawn parameter set into the calibration crop.xml."""
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
def run_simplace(spec: RunSpec, trial_number: int) -> None:
    """Run SIMPLACE once for the current crop.xml. Blocks until jobs finish."""
    # Stale per-location outputs from the previous trial would silently blend into
    # this trial's loss (a location dropped by SLURM keeps its old file), so start
    # each trial from an empty output dir.
    for sub in ("daily", "yearly"):
        shutil.rmtree(spec.out_dir / sub, ignore_errors=True)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    runner = ("simplace_runner_cluster.py" if spec.device == "cluster"
              else "simplace_runner.py")
    log = spec.results_dir / f"simplace_trial_{trial_number}.log"

    proc = subprocess.run(
        [sys.executable, str(spec.crop_dir / runner), str(spec.run_config)],
        capture_output=True, text=True, cwd=str(spec.crop_dir),
    )
    log.write_text(f"$ {runner} {spec.run_config}\n\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    if proc.returncode != 0:
        raise RuntimeError(f"SIMPLACE run failed (rc={proc.returncode}); see {log}")

    produced = len(glob(str(spec.out_dir / "yearly" / "*.csv")))
    if produced == 0:
        # The cluster runner reports "All jobs completed" even when SIMPLACE died
        # immediately (empty squeue), so an empty out/ is the real failure signal.
        raise RuntimeError(f"SIMPLACE produced no output; see {log}")


def read_outputs(out_dir: Path, kind: str, max_workers: int = 32) -> pd.DataFrame:
    """Concatenate every per-location output CSV of one kind ('daily'|'yearly')."""
    paths = sorted(glob(str(out_dir / kind / "*.csv")))
    if not paths:
        raise RuntimeError(f"no {kind} outputs in {out_dir / kind}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        frames = list(ex.map(lambda p: pd.read_csv(p, delimiter=";"), paths))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Study + single trial
# ---------------------------------------------------------------------------
def open_study(spec: RunSpec) -> tuple[optuna.Study, bool]:
    """Load (or create) the persistent study. Returns (study, was_created)."""
    o = spec.optuna_cfg
    if o.get("sampler", "tpe") == "random":
        sampler = optuna.samplers.RandomSampler(seed=o.get("seed"))
    else:
        sampler = optuna.samplers.TPESampler(
            seed=o.get("seed"), n_startup_trials=o.get("n_startup_trials", 10))

    existing = {s.study_name for s in optuna.get_all_study_summaries(storage=spec.storage)}
    study = optuna.create_study(
        study_name=spec.study_name, storage=spec.storage, sampler=sampler,
        direction="minimize", load_if_exists=True,
    )
    return study, spec.study_name not in existing


def run_trial(spec: RunSpec, evaluate, args) -> dict:
    """Run exactly one trial and report it.

    ``evaluate(spec) -> (loss, extras)`` is supplied by the calling script and is
    the only target-specific piece.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[{spec.study_name}] staging calibration run dir ...")
    info = build_run_dir(spec, rebuild=args.rebuild)
    print(f"  run_dir={info['run_dir']}")
    print(f"  project rows={info['rows']:,} locations={info['locations']:,}")

    study, created = open_study(spec)
    if created and spec.optuna_cfg.get("enqueue_baseline", True):
        baseline = read_current_values(spec.crop_xml, spec.crop_name, spec.parameters)
        outside = check_within_bounds(baseline, spec.parameters)
        if outside:
            # Optuna would still record it, but the sampler's model and the search
            # space would disagree — better to say so than to fake a reference loss.
            print(f"  baseline NOT enqueued: {', '.join(outside)} outside the "
                  f"configured bounds (widen them in config.yaml to include it)")
        elif baseline:
            study.enqueue_trial(baseline, skip_if_exists=True)
            print("  new study: trial 0 enqueued with current crop.xml values")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  history: {len(done)} completed trial(s)")

    trial = study.ask()
    drawn = suggest_parameters(trial, spec.parameters)
    print(f"\n[trial {trial.number}] parameters:")
    for k, v in drawn.items():
        print(f"    {k} = {v}")

    apply_parameters(spec.crop_xml, spec.crop_name, drawn)
    shutil.copyfile(spec.crop_xml, spec.results_dir / f"trial_{trial.number}_crop.xml")

    try:
        if args.skip_run:
            print("\n--skip-run: scoring the existing outputs, no simulation")
        else:
            print(f"\n[trial {trial.number}] running SIMPLACE ({spec.device}) ...")
            run_simplace(spec, trial.number)

        loss, extras = evaluate(spec)
        if loss is None or not np.isfinite(loss):
            raise RuntimeError(f"loss is not finite ({loss}); extras={extras}")
    except Exception as exc:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(f"\n[trial {trial.number}] FAILED: {exc}", file=sys.stderr)
        _write_result(spec, {
            "study": spec.study_name, "trial": trial.number, "status": "failed",
            "error": str(exc), "parameters": drawn,
        })
        return {"status": "failed", "error": str(exc)}

    study.tell(trial, loss)
    best = study.best_trial

    result = {
        "study": spec.study_name,
        "crop": spec.crop,
        "target": spec.target,
        "trial": trial.number,
        "status": "completed",
        "loss": float(loss),
        "parameters": drawn,
        "metrics": extras,
        "is_new_best": best.number == trial.number,
        "best_trial": best.number,
        "best_loss": float(best.value),
        "best_parameters": best.params,
        "completed_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if result["is_new_best"]:
        shutil.copyfile(spec.crop_xml, spec.results_dir / "best_crop.xml")

    _write_result(spec, result)
    _print_result(result)
    return result


def _write_result(spec: RunSpec, result: dict) -> None:
    (spec.results_dir / "last.json").write_text(json.dumps(result, indent=2))
    with open(spec.results_dir / "history.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


def _print_result(r: dict) -> None:
    """Machine-readable block: this is what a loop iteration reads back."""
    print("\n" + "=" * 70)
    print(f"RESULT  {r['study']}  trial {r['trial']}")
    print("=" * 70)
    for k, v in (r.get("metrics") or {}).items():
        print(f"  {k:<28s} {v}")
    print(f"  {'LOSS':<28s} {r['loss']:.4f}")
    marker = "  <-- NEW BEST" if r["is_new_best"] else ""
    print(f"  {'best so far':<28s} {r['best_loss']:.4f} (trial {r['best_trial']}){marker}")
    print(f"  {'completed trials':<28s} {r['completed_trials']}")
    print("=" * 70)
    print("JSON " + json.dumps({
        "trial": r["trial"], "loss": r["loss"], "parameters": r["parameters"],
        "best_loss": r["best_loss"], "best_parameters": r["best_parameters"],
        "is_new_best": r["is_new_best"], "completed_trials": r["completed_trials"],
    }))


def show_best(spec: RunSpec) -> int:
    # Deliberately does not create the study: creating an empty one here would
    # make the next real run think it is resuming and skip the baseline trial.
    try:
        study = optuna.load_study(study_name=spec.study_name, storage=spec.storage)
    except (KeyError, optuna.exceptions.StorageInternalError):
        print(f"[{spec.study_name}] no study yet")
        return 1

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print(f"[{spec.study_name}] no completed trials yet")
        return 1
    b = study.best_trial
    print(f"[{spec.study_name}] {len(done)} completed trial(s)")
    print(f"  best trial {b.number}  loss={b.value:.4f}")
    for k, v in b.params.items():
        print(f"    {k} = {v}")
    print(f"  best crop.xml: {spec.results_dir / 'best_crop.xml'}")
    return 0


# ---------------------------------------------------------------------------
# CLI shared by the three scripts
# ---------------------------------------------------------------------------
def build_parser(target: str, description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--crop", default="winter_wheat", help="crop to calibrate")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="optimization config")
    ap.add_argument("--device", default="cluster", choices=["cluster", "local"],
                    help="SIMPLACE driver to invoke")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the calibration run dir from scratch")
    ap.add_argument("--skip-run", action="store_true",
                    help="score the outputs already in out/ instead of simulating")
    ap.add_argument("--show-best", action="store_true",
                    help="print the best trial so far and exit (runs nothing)")
    ap.set_defaults(target=target)
    return ap


def main(target: str, description: str, evaluate) -> int:
    """Entry point shared by the three scripts: one invocation = one trial."""
    args = build_parser(target, description).parse_args()
    spec = load_spec(Path(args.config), args.crop, target, device=args.device)

    if args.show_best:
        return show_best(spec)

    result = run_trial(spec, evaluate, args)
    return 0 if result["status"] == "completed" else 1
