#!/usr/bin/env python
"""Consolidate per-location SIMPLACE outputs into one CSV per experiment.

Each finished experiment leaves ~3086 tiny files behind:

    simplace/<crop>/runs/<exp_id>/out/<exp_id>/yearly/<PointID>_yearly.csv

which is a poor shape for analysis and a worse one for a shared filesystem
(1.3 M inodes for the full 425-experiment matrix). This collapses each
experiment's `yearly/` directory into a single table:

    data/processed/consolidated/<crop>/<exp_id>_yearly.csv[.gz]

The per-location files are the source of truth and are left untouched — this
only ever *reads* them.

Alongside them SIMPLACE writes `out/<exp_id>/CheckResult.csv`, the LAZY
check-level diagnostic log (negative trace-nutrient values and similar). It is
not model output, nothing downstream reads it, and across the full matrix it is
~165 GiB — more than twice the size of the results themselves. `--delete-checkresults`
removes them; without that flag they are only reported.

Usage:
  python orchestration/consolidate_outputs.py --dry-run
  python orchestration/consolidate_outputs.py                       # consolidate all
  python orchestration/consolidate_outputs.py --gzip --jobs 16
  python orchestration/consolidate_outputs.py --crop maize --climate DWD --soil S1
  python orchestration/consolidate_outputs.py --summarize-checkresults \
                                              --delete-checkresults
  python orchestration/consolidate_outputs.py --checkresults-only --delete-checkresults
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import gzip
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("experiments.yaml")

# `<PointID>_yearly.csv`; the numeric stem is what we sort output rows by.
YEARLY_RE = re.compile(r"^(\d+)_yearly\.csv$")
CHECKRESULT = "CheckResult.csv"


AUTO = ("", "auto", "none")


def resolve_repo_root(cfg: dict) -> Path:
    """Repo root: explicit config > SOIL_SCENARIOS_ROOT > this file's location.

    Same validation as generate.py — a configured root that does not contain
    simplace/ is refused, because a stale absolute path pointing at an older
    checkout would otherwise read the wrong tree in silence. Unlike generate.py,
    `repo_root: auto` here falls through to the env var before defaulting, which
    is what makes SOIL_SCENARIOS_ROOT usable without editing experiments.yaml.
    """
    raw = cfg.get("repo_root")
    if raw is None or str(raw).strip().lower() in AUTO:
        raw = os.environ.get("SOIL_SCENARIOS_ROOT") or "auto"
    if str(raw).strip().lower() in AUTO:
        return REPO_ROOT
    root = Path(raw).expanduser().resolve()
    if not (root / "simplace").is_dir():
        raise SystemExit(
            f"configured repo_root does not look like this repo: {root}\n"
            f"  (no simplace/ inside it).  Set `repo_root: auto` in experiments.yaml."
        )
    return root


# --- discovery ---------------------------------------------------------------

@dataclass
class Experiment:
    crop: str
    exp_id: str          # e.g. "GFDL-ESM4_ssp370__DLB"
    run_dir: Path
    yearly_dir: Path
    checkresults: list[Path] = field(default_factory=list)

    @property
    def climate(self) -> str:
        return self.exp_id.rsplit("__", 1)[0]

    @property
    def soil(self) -> str:
        return self.exp_id.rsplit("__", 1)[-1]


def discover(root: Path, cfg: dict, crops, climates, soils) -> list[Experiment]:
    """Walk simplace/<crop>/runs/ and pick up every experiment with a yearly dir.

    The output namespace under `out/` is the run dir's `exp_name`, which
    generate.py sets to the exp_id — but smoke tests write `SMOKE_<exp_id>`
    into the same tree, so we glob and keep only the namespace that matches the
    run dir's own name.
    """
    runs_subdir = cfg.get("paths", {}).get("runs_subdir", "runs")
    found: list[Experiment] = []
    for crop in cfg["crops"]:
        if crops and crop not in crops:
            continue
        runs = root / "simplace" / crop / runs_subdir
        if not runs.is_dir():
            continue
        for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            exp_id = run_dir.name
            exp = Experiment(crop, exp_id, run_dir, run_dir / "out" / exp_id / "yearly")
            if climates and exp.climate not in climates:
                continue
            if soils and exp.soil not in soils:
                continue
            # CheckResult.csv is per output namespace; take every one under the
            # run dir, smoke-test namespaces included.
            exp.checkresults = sorted(run_dir.glob(f"out/*/{CHECKRESULT}"))
            if exp.yearly_dir.is_dir() or exp.checkresults:
                found.append(exp)
    return found


def yearly_files(yearly_dir: Path) -> list[tuple[int, Path]]:
    """(PointID, path) for every location file, ordered by PointID.

    A missing location is normal, not an error: SIMPLACE writes a file only once
    a location reaches DoHarvest, so sites that never mature (routine for maize
    under the cooler historical GCM climate) simply have none.
    """
    out = []
    with os.scandir(yearly_dir) as it:
        for e in it:
            m = YEARLY_RE.match(e.name)
            if m and e.is_file():
                out.append((int(m.group(1)), Path(e.path)))
    out.sort()
    return out


# --- consolidation -----------------------------------------------------------

@dataclass
class Result:
    crop: str
    exp_id: str
    status: str          # written | skipped | empty | error
    n_files: int = 0
    n_rows: int = 0
    out_bytes: int = 0
    src_bytes: int = 0
    seconds: float = 0.0
    message: str = ""


def _open_out(path: Path, use_gzip: bool):
    if use_gzip:
        return gzip.open(path, "wb", compresslevel=6)
    return open(path, "wb")


def consolidate(exp: Experiment, out_path: Path, use_gzip: bool, force: bool) -> Result:
    """Concatenate one experiment's location files into out_path.

    Header handling is the whole risk here: every location file repeats the
    header, and a file whose header differs would silently misalign every column
    downstream. So the first file's header is written once and every subsequent
    header is compared byte-for-byte; a mismatch aborts this experiment rather
    than producing a corrupt table.
    """
    t0 = time.time()
    if not exp.yearly_dir.is_dir():
        return Result(exp.crop, exp.exp_id, "empty", message="no yearly/ directory")

    files = yearly_files(exp.yearly_dir)
    if not files:
        return Result(exp.crop, exp.exp_id, "empty", message="yearly/ is empty")

    src_bytes = sum(p.stat().st_size for _, p in files)
    newest = max(p.stat().st_mtime for _, p in files)
    if out_path.exists() and not force and out_path.stat().st_mtime >= newest:
        return Result(exp.crop, exp.exp_id, "skipped", len(files), 0,
                      out_path.stat().st_size, src_bytes, time.time() - t0,
                      "up to date (--force to rewrite)")

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: bytes | None = None
    n_rows = 0
    try:
        with _open_out(tmp, use_gzip) as dst:
            for _, src in files:
                with open(src, "rb") as fh:
                    line = fh.readline()
                    if not line:
                        continue                      # header-only/zero-byte file
                    if header is None:
                        header = line
                        if not header.endswith(b"\n"):
                            header += b"\n"
                        dst.write(header)
                    elif line != header:
                        raise ValueError(
                            f"header mismatch in {src.name}: this experiment's "
                            f"location files do not share one schema"
                        )
                    # Count rows while streaming: cheaper than a second pass, and
                    # it is what lets the manifest be checked against the source.
                    last = b"\n"
                    while True:
                        chunk = fh.read(1 << 20)
                        if not chunk:
                            break
                        n_rows += chunk.count(b"\n")
                        dst.write(chunk)
                        last = chunk[-1:]
                    if last != b"\n":
                        # No trailing newline would glue this file's last row to
                        # the next file's first one.
                        dst.write(b"\n")
                        n_rows += 1
        os.replace(tmp, out_path)
    except Exception as exc:                           # noqa: BLE001 - reported per experiment
        tmp.unlink(missing_ok=True)
        return Result(exp.crop, exp.exp_id, "error", len(files), 0, 0, src_bytes,
                      time.time() - t0, f"{type(exc).__name__}: {exc}")

    return Result(exp.crop, exp.exp_id, "written", len(files), n_rows,
                  out_path.stat().st_size, src_bytes, time.time() - t0)


# --- CheckResult handling ----------------------------------------------------

def summarize_checkresult(path: Path) -> collections.Counter:
    """Count check messages by kind so the log can be dropped without losing the gist.

    Format is a one-line session banner, then a tab-separated table. What follows
    VariableName depends on CheckMethod: `checkResult` puts a free-text message
    there, `checkLimits` puts value/min/max instead. Both are folded to a kind —
    numbers in the free text are masked, and a limits breach is rendered as the
    bound it violated — so thousands of near-identical daily messages collapse to
    a handful of rows.
    """
    kinds: collections.Counter = collections.Counter()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        fh.readline()                                  # session banner
        head = fh.readline().rstrip("\r\n").split("\t")
        try:
            i_how = head.index("CheckMethod")
            i_var = head.index("VariableName")
            i_msg = head.index("AdditionalInformation")
        except ValueError:
            i_how, i_var, i_msg = 0, 4, 5
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) <= i_msg:
                continue
            how = f[i_how]
            if how == "checkLimits" and len(f) > i_msg + 2:
                lo, hi = f[i_msg + 1], f[i_msg + 2]
                kind = f"value outside limits [{lo}, {hi}]"
            else:
                kind = re.sub(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", "#", f[i_msg]).strip()
            kinds[(how, f[i_var], kind)] += 1
    return kinds


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# --- driver ------------------------------------------------------------------

def _job(args):
    exp, out_path, use_gzip, force = args
    return consolidate(exp, out_path, use_gzip, force)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", default=None,
                    help="default: <repo>/data/processed/consolidated")
    ap.add_argument("--crop", action="append", default=[],
                    help="repeatable; default all crops in experiments.yaml")
    ap.add_argument("--climate", action="append", default=[],
                    help="repeatable, e.g. DWD, HYRAS_OBS, MIROC6_ssp370")
    ap.add_argument("--soil", action="append", default=[],
                    help="repeatable: S1 S2 S3 DL DLB")
    ap.add_argument("--gzip", action="store_true",
                    help="write .csv.gz (~5x smaller; pandas reads it directly)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="experiments consolidated in parallel (default 8)")
    ap.add_argument("--force", action="store_true",
                    help="rewrite outputs even if newer than their sources")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written/deleted, touch nothing")
    ap.add_argument("--checkresults-only", action="store_true",
                    help="skip consolidation; only report/delete CheckResult.csv")
    ap.add_argument("--summarize-checkresults", action="store_true",
                    help="tally check messages into checkresult_summary.csv first "
                         "(reads every log in full — slow, but keeps the gist)")
    ap.add_argument("--delete-checkresults", action="store_true",
                    help="actually delete the CheckResult.csv logs. Without this "
                         "they are only reported.")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    root = resolve_repo_root(cfg)
    out_dir = Path(a.out_dir).expanduser().resolve() if a.out_dir \
        else root / "data" / "processed" / "consolidated"

    exps = discover(root, cfg, set(a.crop), set(a.climate), set(a.soil))
    if not exps:
        print("no experiments matched the given filters", file=sys.stderr)
        return 1
    print(f"repo root : {root}")
    print(f"output dir: {out_dir}")
    print(f"matched   : {len(exps)} experiments across "
          f"{len({e.crop for e in exps})} crops\n")

    results: list[Result] = []
    if not a.checkresults_only:
        suffix = "_yearly.csv.gz" if a.gzip else "_yearly.csv"
        jobs = [(e, out_dir / e.crop / f"{e.exp_id}{suffix}", a.gzip, a.force)
                for e in exps]
        if a.dry_run:
            for e, p, _, _ in jobs:
                n = len(yearly_files(e.yearly_dir)) if e.yearly_dir.is_dir() else 0
                print(f"  would write {p.relative_to(out_dir)}  <- {n} files")
        else:
            done = 0
            with cf.ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
                for r in pool.map(_job, jobs):
                    results.append(r)
                    done += 1
                    tag = {"written": "ok", "skipped": "--", "empty": "!!",
                           "error": "ERR"}[r.status]
                    print(f"[{done:>4}/{len(jobs)}] {tag} {r.crop}/{r.exp_id}  "
                          f"{r.n_files} files -> {r.n_rows:,} rows "
                          f"({human(r.out_bytes)}, {r.seconds:.0f}s) {r.message}",
                          flush=True)
            write_manifest(out_dir, results)

    # --- CheckResult.csv -----------------------------------------------------
    logs = [(e, p) for e in exps for p in e.checkresults]
    total = sum(p.stat().st_size for _, p in logs)
    print(f"\nCheckResult.csv: {len(logs)} files, {human(total)}")

    if logs and a.summarize_checkresults and not a.dry_run:
        summarize_all(out_dir, logs, a.jobs)

    if not logs:
        pass
    elif not a.delete_checkresults:
        print("  not deleted (pass --delete-checkresults to reclaim this space)")
    elif a.dry_run:
        print("  --dry-run: nothing deleted")
    else:
        # An experiment's log is only expendable once its results are safely
        # consolidated, so refuse to delete beside a failed run.
        bad = {(r.crop, r.exp_id) for r in results if r.status == "error"}
        if a.checkresults_only and not a.force:
            # Nothing was consolidated in this invocation, so the manifest is the
            # only evidence the results were ever saved. No entry -> keep the log.
            done_ok = read_manifest_keys(out_dir)
            bad |= {(e.crop, e.exp_id) for e, _ in logs
                    if (e.crop, e.exp_id) not in done_ok}
        freed = removed = 0
        for e, p in logs:
            if (e.crop, e.exp_id) in bad:
                print(f"  keeping {p} (not consolidated / consolidation failed; "
                      f"--force overrides)")
                continue
            size = p.stat().st_size
            p.unlink()
            freed += size
            removed += 1
        print(f"  deleted {removed} files, freed {human(freed)}")

    n_err = sum(r.status == "error" for r in results)
    if n_err:
        print(f"\n{n_err} experiment(s) failed — see ERR lines above", file=sys.stderr)
    return 1 if n_err else 0


def read_manifest_keys(out_dir: Path) -> set[tuple[str, str]]:
    """(crop, exp_id) pairs already recorded as consolidated."""
    path = out_dir / "manifest.csv"
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(r["crop"], r["exp_id"]) for r in csv.DictReader(fh)}


def write_manifest(out_dir: Path, results: list[Result]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"
    prev = {}
    if path.exists():                                  # keep rows for untouched experiments
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                prev[(row["crop"], row["exp_id"])] = row
    for r in results:
        if r.status in ("written", "skipped"):
            prev[(r.crop, r.exp_id)] = {
                "crop": r.crop, "exp_id": r.exp_id,
                "climate": r.exp_id.rsplit("__", 1)[0],
                "soil": r.exp_id.rsplit("__", 1)[-1],
                "n_location_files": r.n_files, "n_rows": r.n_rows,
                "src_bytes": r.src_bytes, "out_bytes": r.out_bytes,
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
    cols = ["crop", "exp_id", "climate", "soil", "n_location_files", "n_rows",
            "src_bytes", "out_bytes", "written_utc"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in sorted(prev):
            w.writerow({c: prev[k].get(c, "") for c in cols})
    print(f"\nmanifest: {path}")


def summarize_all(out_dir: Path, logs, jobs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "checkresult_summary.csv"
    print(f"  summarizing {len(logs)} logs (this reads them in full) ...", flush=True)
    rows = []
    with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for (e, p), kinds in zip(logs, pool.map(summarize_checkresult,
                                                [p for _, p in logs])):
            for (how, var, msg), n in kinds.most_common():
                rows.append([e.crop, e.exp_id, how, var, msg, n])
            print(f"    {e.crop}/{e.exp_id}: {sum(kinds.values()):,} messages",
                  flush=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "exp_id", "check_method", "variable",
                    "message_kind", "count"])
        w.writerows(rows)
    print(f"  summary: {path}")


if __name__ == "__main__":
    sys.exit(main())
