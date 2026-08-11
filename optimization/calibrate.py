#!/usr/bin/env python
"""Agentic SIMPLACE calibration — the tool Claude drives, one iteration at a time.

There is no sampler here. Every parameter change comes from a proposal Claude
writes, with a stated reason; this script validates it, runs the model, scores it
with the *existing* loss function, produces the diagnostics needed to decide the
next move, and appends an immutable record to the ledger.

    calibrate.py status    --crop winter_wheat --target lai
    calibrate.py run       --crop winter_wheat --target lai \
                           --params '{"SLATableSLA": {"3": 0.0118}}' \
                           --reason "peak LAI 0.6 high in the anthesis bin"
    calibrate.py diagnose  --crop winter_wheat --target lai
    calibrate.py history   --crop winter_wheat --target lai
    calibrate.py handoff   --crop winter_wheat
    calibrate.py verify-lai --crop winter_wheat
    calibrate.py promote   --crop winter_wheat --target lai --yes

Iteration 0 is the baseline: run it with no ``--params`` to record the objective
of the phenology-optimized starting point, so every later iteration has a
reference to beat.

Guarantees
----------
* The optimized phenology parameters are snapshotted on first use and re-verified
  by re-reading the written XML after every change. A change that touches them
  aborts the iteration and rolls the file back.
* A proposal that breaks a bound, a table shape rule or the partitioning closure
  is rejected before the model runs, and logged to ``rejected.jsonl``.
* A failed iteration rolls ``crop.xml`` back to the last good state, so the next
  iteration always starts from a known-good parameter set.
* ``ledger.jsonl`` is append-only. Nothing in it is ever rewritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as cc  # noqa: E402
import calib_diagnostics as cd  # noqa: E402
import common  # noqa: E402
import objectives  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_json_arg(value: str | None) -> dict:
    """``--params`` accepts inline JSON or ``@path/to/file.json``."""
    if not value:
        return {}
    text = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"could not parse parameters as JSON: {exc}")


def _fmt(value, nd: int = 5) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v, nd) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.{nd}g}"
    return str(value)


def _prepare(args, target: str | None = None) -> cc.CalibSpec:
    """Load the spec and stage the isolated run dir (idempotent)."""
    spec = cc.load_spec(Path(args.config), args.crop, target or args.target,
                        device=getattr(args, "device", "cluster"),
                        n_locations=getattr(args, "locations", None))
    info = cc.stage(spec, rebuild=getattr(args, "rebuild", False))
    spec.stage_info = info
    return spec


def _baseline_files(spec: cc.CalibSpec) -> None:
    """Record the starting crop.xml and the freeze snapshot, once per study."""
    baseline = spec.ledger_dir / "baseline_crop.xml"
    if not baseline.exists():
        shutil.copyfile(spec.crop_xml, baseline)
    cc.ensure_frozen_snapshot(spec)
    if spec.target_cfg.get("protected"):
        cc.ensure_protected_baseline(spec)


def _protection_gate(spec: cc.CalibSpec, changed: dict, allowed: bool) -> int | None:
    """Refuse to write a protected target's parameters unless explicitly allowed.

    The phenology set is already optimized for every crop here. Validating it
    (running it unchanged and scoring it) is always fine; overwriting it is a
    deliberate act that has to be asked for. Returns an exit code to stop with,
    or None to continue.
    """
    if not spec.target_cfg.get("protected") or not changed or allowed:
        return None
    note = " ".join((spec.target_cfg.get("protected_note") or "").split())
    print(f"\n  REFUSED — '{spec.target}' is a protected target and this proposal "
          f"changes {len(changed)} parameter(s):")
    for pid in sorted(changed):
        print(f"    {pid}")
    if note:
        print(f"\n  {note}")
    print(f"\n  To validate the optimized values instead, run with no --params.")
    print(f"  To recalibrate anyway, add --allow-recalibration (recorded in the ledger).\n")
    return 4


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    spec = _prepare(args)
    _baseline_files(spec)

    snapshot = cc.ensure_frozen_snapshot(spec)
    drift = cc.verify_frozen(spec.crop_xml, snapshot)
    ledger = cc.read_ledger(spec)
    best = cc.best_record(spec)
    stop = cc.stop_check(spec)
    space = cc.describe_space(spec)

    payload = {
        "crop": spec.crop, "crop_name": spec.crop_name, "target": spec.target,
        "run_dir": str(spec.run.run_dir), "out_dir": str(spec.run.out_dir),
        "ledger_dir": str(spec.ledger_dir),
        "objective": spec.target_cfg.get("objective"),
        "project_rows": spec.stage_info.get("rows"),
        "project_locations": spec.stage_info.get("locations"),
        "subset": spec.run.subset,
        "frozen": {
            "parameters": spec.frozen,
            "digest": snapshot["digest"],
            "intact": not drift,
            "drift": drift,
        },
        "next_iteration": cc.next_iteration(spec),
        "n_completed": stop["n_completed"],
        "best": None if not best else {
            "iteration": best["iteration"], "objective": best["objective"],
            "parameters_changed": sorted(best.get("parameters_changed", {})),
        },
        "stopping": stop,
        "parameters": space,
        "constraints": spec.constraints,
        "history": [
            {"iteration": r["iteration"], "objective": r.get("objective"),
             "improved": r.get("improved"), "status": r.get("status"),
             "changed": sorted(r.get("parameters_changed", {})),
             "reason": r.get("reason")}
            for r in ledger
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"\n{'=' * 78}\n  {spec.crop} · {spec.target} calibration\n{'=' * 78}")
    print(f"  run dir            {spec.run.run_dir}")
    print(f"  ledger             {spec.ledger_dir}")
    print(f"  objective          {spec.target_cfg.get('objective')}")
    print(f"  project            {spec.stage_info['rows']:,} rows / "
          f"{spec.stage_info['locations']:,} locations")
    print(f"  frozen phenology   {'INTACT' if not drift else 'DRIFTED!'} "
          f"({len(spec.frozen)} parameters, digest {snapshot['digest'][:12]})")
    for line in drift:
        print(f"     ! {line}")

    print(f"\n  iterations completed  {stop['n_completed']}")
    if best:
        print(f"  best                  iteration {best['iteration']}  "
              f"objective {best['objective']:.4f}")
    print(f"  next iteration        {cc.next_iteration(spec)}")
    print(f"  stopping              {'STOP — ' if stop['stop'] else ''}{stop['reason']}")

    print(f"\n{'-' * 78}\n  calibratable parameters\n{'-' * 78}")
    disabled = []
    for row in space:
        if not row.get("present"):
            print(f"  {row['parameter']:<38s} (absent for this crop)")
            continue
        # A disabled parameter carries no bounds — it never entered the space —
        # so it cannot be rendered like a calibratable one. Collect and list it
        # separately with the reason it was switched off.
        if row.get("enabled") is False:
            disabled.append(row)
            continue
        flags = []
        if row.get("affects_lai"):
            flags.append("affects LAI")
        if row.get("coupled_with"):
            flags.append("coupled: " + ", ".join(row["coupled_with"]))
        if not row.get("movable", True):
            flags.append("IMMOVABLE")
        elif row.get("immovable_indices"):
            flags.append(f"fixed elements {row['immovable_indices']}")
        print(f"  {row['parameter']:<38s} {_fmt(row['value'])}")
        print(f"  {'':38s} bounds {_fmt(row['bounds'])}")
        print(f"  {'':38s} controls: {', '.join(row.get('controls') or []) or '-'}"
              + (f"   [{'; '.join(flags)}]" if flags else ""))

    if disabled:
        print(f"\n{'-' * 78}\n  disabled (enabled: false in calibration.yaml)\n{'-' * 78}")
        for row in disabled:
            print(f"  {row['parameter']:<38s} {_fmt(row['value'])}")
            print(f"  {'':38s} {' '.join((row.get('note') or '').split())}")

    if ledger:
        print(f"\n{'-' * 78}\n  history\n{'-' * 78}")
        for r in ledger:
            mark = {True: "+", False: " ", None: "?"}.get(r.get("improved"), " ")
            obj = f"{r['objective']:.4f}" if r.get("objective") is not None else "FAILED"
            print(f"  {mark} iter {r['iteration']:>3d}  {obj:>10s}  "
                  f"{', '.join(sorted(r.get('parameters_changed', {}))) or 'baseline'}")
            if r.get("reason"):
                print(f"        {r['reason'][:110]}")
    print()
    return 0


# ---------------------------------------------------------------------------
# run — one iteration
# ---------------------------------------------------------------------------
def _evaluate(spec: cc.CalibSpec, iter_dir: Path) -> tuple[float, dict, dict, list[str]]:
    """Objective + metrics + diagnostics + figures for the outputs now in out/."""
    process_result, loss_fn = cc.objective_for(spec.target)

    if spec.target == "phenology":
        pairs = process_result(spec.run)
        objective, metrics = loss_fn(pairs)
        points = cd.load_points(spec)
        diagnostics = cd.phenology_diagnostics(pairs, points)
        pairs.to_csv(iter_dir / "pairs.csv.gz", index=False, compression="gzip")
        figures = cd.phenology_plots(pairs, iter_dir / "diagnostics",
                                     f"{spec.crop} · phenology", points=points)
    elif spec.target == "lai":
        pairs = process_result(spec.run)
        objective, metrics = loss_fn(pairs)
        diagnostics, seasons = cd.lai_diagnostics(pairs, objectives.DVS_BINS)
        pairs.to_csv(iter_dir / "pairs.csv.gz", index=False, compression="gzip")
        if not seasons.empty:
            seasons.to_csv(iter_dir / "season_shape.csv", index=False)
        daily = cd.load_lai_daily(spec)
        figures = cd.lai_plots(pairs, seasons, objectives.DVS_BINS, iter_dir / "diagnostics",
                               f"{spec.crop} · LAI", sim_daily=daily)
    elif spec.target == "yield":
        pairs = process_result(spec.run)
        objective, metrics = loss_fn(pairs)
        frame = cd.yield_attribution_frame(spec)
        reference = (spec.cfg.get("crop_reference") or {}).get(spec.crop)
        diagnostics = cd.yield_diagnostics(frame, reference)
        frame.to_csv(iter_dir / "pairs.csv.gz", index=False, compression="gzip")
        figures = cd.yield_plots(frame, iter_dir / "diagnostics", f"{spec.crop} · yield")
    else:
        raise SystemExit(f"target {spec.target!r} has no agentic evaluation path")

    return float(objective), metrics, diagnostics, figures


def cmd_run(args) -> int:
    spec = _prepare(args)
    _baseline_files(spec)
    snapshot = cc.ensure_frozen_snapshot(spec)

    current = cc.current_values(spec.crop_xml, spec.crop_name, spec.space)
    proposed_raw = _load_json_arg(args.params)
    proposal = cc.normalise_proposal(proposed_raw, current, spec.space)
    changed = cc.changed_entries(proposal, current)

    iteration = cc.next_iteration(spec)
    if iteration == 0 and changed:
        print("  note: iteration 0 usually records the unchanged baseline. Proceeding with "
              "the proposed change; the ledger will have no reference objective.")

    # --- protected target: writing has to be asked for ------------------------
    # A dry run is inspection, not a write, so the gate reports rather than blocks;
    # it is enforced below, before anything reaches the file.
    allow_recal = bool(getattr(args, "allow_recalibration", False))
    blocked_by_protection = bool(spec.protected and changed and not allow_recal)
    if not args.dry_run:
        gate = _protection_gate(spec, changed, allow_recal)
        if gate is not None:
            return gate

    # --- validate before anything is written ---------------------------------
    violations = cc.validate(proposal, current, spec.space, spec.meta, spec.constraints,
                             spec.frozen, spec.crop_xml, spec.crop_name)

    # A dry run answers "would this be accepted?" and writes nothing at all — not
    # even to rejected.jsonl, because a pre-flight check is not a proposal. This is
    # what an agent calls before spending a SLURM run on a change that cannot pass.
    if args.dry_run:
        print(json.dumps({
            "iteration": iteration, "valid": not violations,
            "would_change": changed,
            "violations": [v.as_dict() for v in violations],
            "blocked_by_protection": blocked_by_protection,
        }, indent=2, default=str))
        return 0 if not (violations or blocked_by_protection) else 2

    if violations:
        record = {"timestamp": cc.utcnow(), "crop": spec.crop, "target": spec.target,
                  "intended_iteration": iteration, "proposal": proposed_raw,
                  "reason": args.reason, "violations": [v.as_dict() for v in violations],
                  "forced": bool(args.force)}
        with open(spec.ledger_dir / "rejected.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"\n  PROPOSAL REJECTED — {len(violations)} constraint violation(s):")
        for v in violations:
            print(f"    [{v.constraint}] {v.message}")
        if not args.force:
            print("\n  Nothing was written. Revise the proposal, or pass --force to override "
                  "(the override is recorded in the ledger).\n")
            return 2
        print("\n  --force: proceeding anyway; the violations are recorded in the ledger.\n")

    iter_dir = spec.iteration_dir(iteration)
    previous_xml = spec.ledger_dir / "current_crop.xml"
    if not previous_xml.exists():
        shutil.copyfile(spec.crop_xml, previous_xml)

    # --- apply, then verify the frozen set on the written file ---------------
    if proposal:
        common.apply_parameters(spec.crop_xml, spec.crop_name, proposal)
    drift = cc.verify_frozen(spec.crop_xml, snapshot)
    if drift:
        shutil.copyfile(previous_xml, spec.crop_xml)
        print("\n  ABORTED — the write changed frozen phenology parameters:")
        for line in drift:
            print(f"    {line}")
        print("  crop.xml was rolled back. Nothing was run.\n")
        return 3

    shutil.copyfile(spec.crop_xml, iter_dir / "crop.xml")
    (iter_dir / "proposal.json").write_text(json.dumps({
        "iteration": iteration, "proposal": proposed_raw,
        "normalised": proposal, "changed": changed,
        "reason": args.reason, "hypothesis": args.hypothesis,
        "reasoning": args.reasoning, "expected_effect": args.expected_effect,
    }, indent=2, default=str) + "\n")

    print(f"\n[{spec.crop} · {spec.target}] iteration {iteration}")
    print(f"  run dir   {spec.run.run_dir}")
    print(f"  project   {spec.stage_info['rows']:,} rows / "
          f"{spec.stage_info['locations']:,} locations")
    if changed:
        print("  changing:")
        for pid, change in changed.items():
            if change["indices"] is not None:
                for i in change["indices"]:
                    print(f"    {pid}[{i}]  {_fmt(change['from'][i])} -> {_fmt(change['to'][i])}")
            else:
                print(f"    {pid}  {_fmt(change['from'])} -> {_fmt(change['to'])}")
    else:
        print("  changing: nothing (baseline iteration)")
    if args.reason:
        print(f"  reason:   {args.reason}")

    # --- run + score ---------------------------------------------------------
    started = time.time()
    status, error = "completed", None
    objective = metrics = diagnostics = figures = None
    try:
        if args.skip_run:
            print("\n  --skip-run: scoring the outputs already in out/, no simulation")
        else:
            print(f"\n  running SIMPLACE ({spec.run.device}) ...")
            common.run_simplace(spec.run, iteration, log_path=iter_dir / "simplace.log")
        objective, metrics, diagnostics, figures = _evaluate(spec, iter_dir)
    except Exception as exc:  # noqa: BLE001 — any failure must roll back and be recorded
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        shutil.copyfile(previous_xml, spec.crop_xml)
        print(f"\n  ITERATION FAILED: {error}")
        print("  crop.xml rolled back to the last good state.")

    elapsed = round(time.time() - started, 1)
    previous_best = cc.best_record(spec)
    # Two different questions, deliberately kept apart:
    #   improved      — did it get *materially* better? drives patience/stopping.
    #   best_record   — is it the lowest objective so far? drives best_crop.xml.
    # A run that beats the record by less than min_improvement is the new best but
    # is not progress, and conflating the two either freezes best_crop.xml or makes
    # the patience counter never fire.
    min_improvement = float(spec.stopping.get("min_improvement", 0.0))
    improved = is_best = None
    if status == "completed":
        improved = (previous_best is None
                    or objective < previous_best["objective"] - min_improvement)
        is_best = previous_best is None or objective < previous_best["objective"]

    record = {
        "iteration": iteration,
        "timestamp": cc.utcnow(),
        "crop": spec.crop,
        "crop_name": spec.crop_name,
        "target": spec.target,
        "status": status,
        "error": error,
        "scope": {
            "project_csv": str(spec.run.project_csv),
            "rows": spec.stage_info.get("rows"),
            "locations": spec.stage_info.get("locations"),
            "subset": spec.run.subset,
            "climate": spec.run.mount_data,
            "years": (diagnostics or {}).get("years"),
        },
        "parameters": {k: v for k, v in {**current, **proposal}.items()},
        "parameters_changed": {k: v["to"] for k, v in changed.items()},
        "previous_values": {k: v["from"] for k, v in changed.items()},
        "changed_detail": changed,
        "objective": objective,
        "objective_name": spec.target_cfg.get("objective"),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "reason": args.reason,
        "hypothesis": args.hypothesis,
        "agent_reasoning": args.reasoning,
        "expected_effect": args.expected_effect,
        "improved": improved,
        "is_best": is_best,
        "previous_best": None if not previous_best else {
            "iteration": previous_best["iteration"], "objective": previous_best["objective"]},
        "delta_vs_best": (None if (status != "completed" or previous_best is None)
                          else round(objective - previous_best["objective"], 6)),
        "frozen_digest": snapshot["digest"],
        "frozen_intact": not cc.verify_frozen(spec.crop_xml, snapshot),
        # For a protected target: how far the current values have moved from the
        # already-optimized set that was preserved on first use. Empty means the
        # iteration was a pure validation run.
        "protected": spec.protected,
        "protected_drift": cc.protected_drift(spec) if spec.protected else [],
        "recalibration_allowed": allow_recal,
        "constraint_violations": [v.as_dict() for v in violations],
        "forced": bool(args.force),
        "figures": figures or [],
        "iteration_dir": str(iter_dir),
        "elapsed_seconds": elapsed,
    }
    cc.append_ledger(spec, record)
    (iter_dir / "metrics.json").write_text(json.dumps({
        "objective": objective, "metrics": metrics, "diagnostics": diagnostics,
    }, indent=2, default=str) + "\n")

    if status == "completed":
        shutil.copyfile(spec.crop_xml, previous_xml)
        if is_best:
            shutil.copyfile(spec.crop_xml, spec.ledger_dir / "best_crop.xml")

    state = cc.read_state(spec)
    best_now = cc.best_record(spec)
    state.update({
        "crop": spec.crop, "target": spec.target, "updated": cc.utcnow(),
        "n_iterations": cc.next_iteration(spec),
        "last_iteration": iteration, "last_status": status,
        "best_iteration": None if not best_now else best_now["iteration"],
        "best_objective": None if not best_now else best_now["objective"],
        "frozen_digest": snapshot["digest"],
        "run_dir": str(spec.run.run_dir),
    })
    state.setdefault("created", cc.utcnow())
    cc.write_state(spec, state)

    cd.history_plot(cc.read_ledger(spec), spec.ledger_dir / "history.png",
                    f"{spec.crop} · {spec.target} · objective per iteration")

    _report(spec, record)
    return 0 if status == "completed" else 1


def _report(spec: cc.CalibSpec, record: dict) -> None:
    """Human summary plus the one-line JSON the calling agent reads back."""
    print("\n" + "=" * 78)
    print(f"ITERATION {record['iteration']}  {spec.crop} · {spec.target}  [{record['status']}]")
    print("=" * 78)
    for key, value in (record.get("metrics") or {}).items():
        print(f"  {key:<34s} {value}")
    if record["objective"] is not None:
        print(f"  {'OBJECTIVE':<34s} {record['objective']:.4f}")
        if record["previous_best"]:
            if record["improved"]:
                arrow = "IMPROVED"
            elif record["is_best"]:
                arrow = "new best, but below the min_improvement threshold"
            else:
                arrow = "no improvement"
            print(f"  {'previous best':<34s} {record['previous_best']['objective']:.4f} "
                  f"(iter {record['previous_best']['iteration']})  -> {arrow} "
                  f"({record['delta_vs_best']:+.4f})")
    print(f"  {'frozen phenology intact':<34s} {record['frozen_intact']}")
    print(f"  {'elapsed':<34s} {record['elapsed_seconds']}s")
    print(f"  {'iteration dir':<34s} {record['iteration_dir']}")

    diagnostics = record.get("diagnostics") or {}
    if spec.target == "lai" and "by_dvs_bin" in diagnostics:
        print(f"\n  {'DVS bin':<26s} {'n':>6s} {'obs':>8s} {'sim':>8s} {'bias':>8s} {'rmse':>8s}")
        for row in diagnostics["by_dvs_bin"]:
            print(f"  {row['dvs_bin']:<26s} {row['n']:>6} {row['obs_mean']:>8.3f} "
                  f"{row['sim_mean']:>8.3f} {row['bias']:>+8.3f} {row['rmse']:>8.3f}")
        shape = diagnostics.get("curve_shape", {})
        for name in ("peak_lai", "peak_timing", "plateau_duration", "rise_rate", "decline_rate",
                     "early_canopy"):
            item = shape.get(name)
            if isinstance(item, dict) and item.get("n"):
                print(f"  {name:<26s} obs {item['obs_median']:>8.4g}   sim {item['sim_median']:>8.4g}"
                      f"   diff {item['median_diff']:>+8.4g}  [{item.get('unit', '')}]")
    if spec.target == "phenology":
        for name in ("flowering", "maturity", "duration"):
            m = diagnostics.get(name) or {}
            if m.get("n"):
                print(f"\n  {name:<14s} RMSE {m['RMSE']:>6.2f} d   bias {m['bias']:>+6.2f} d   "
                      f"R2 {m.get('R2') if m.get('R2') is None else round(m['R2'], 3)}   "
                      f"n={m['n']:,}")
        structure = diagnostics.get("residual_structure") or {}
        if structure:
            print(f"\n  warmth slope    {structure.get('vs_season_warmth_slope')} d/d"
                  f"   latitude slope {structure.get('vs_latitude_slope')} d/deg")
        if diagnostics.get("attribution"):
            print(f"  attribution     {diagnostics['attribution']['verdict']}")
        if record.get("protected_drift"):
            print(f"\n  ! this run has moved {len(record['protected_drift'])} parameter(s) "
                  f"away from the preserved optimized set:")
            for line in record["protected_drift"]:
                print(f"      {line}")
        elif record.get("protected"):
            print(f"  optimized set   unchanged (validation run)")
    if spec.target == "yield" and "attribution" in diagnostics:
        a = diagnostics["attribution"]
        print(f"\n  harvest index   sim {a['hi_simulated_median']:.3f}   "
              f"required {a['hi_required_to_match_obs']:.3f}   ref {a['reference_harvest_index']}")
        print(f"  AG biomass      sim {a['ag_biomass_simulated_median']:.2f}   "
              f"required {a['ag_biomass_required_to_match_obs']:.2f}   "
              f"ref {a['reference_ag_biomass_t_ha']}")
        print(f"  attribution     {a['verdict']}")
    if spec.target == "yield" and diagnostics.get("translocation"):
        t = diagnostics["translocation"]
        print(f"\n  translocation   Yield_t_ha              RMSE {t['yield_plain']['RMSE']:.3f} "
              f"t/ha  mean {t['mean_plain_t_ha']:.2f}")
        print(f"                  Yield_translocated_t_ha RMSE "
              f"{t['yield_translocated']['RMSE']:.3f} t/ha  mean {t['mean_translocated_t_ha']:.2f}")
        print(f"                  FRTDM contribution      "
              f"{t['median_translocation_contribution_t_ha']:+.3f} t/ha (median)")
        print(f"  FRTDM verdict   {t['verdict']}")

    stop = cc.stop_check(spec)
    print(f"\n  stopping: {'STOP — ' if stop['stop'] else 'continue — '}{stop['reason']}")
    print("=" * 78)
    print("JSON " + json.dumps({
        "iteration": record["iteration"], "status": record["status"],
        "objective": record["objective"], "improved": record["improved"],
        "delta_vs_best": record["delta_vs_best"],
        "parameters_changed": record["parameters_changed"],
        "frozen_intact": record["frozen_intact"],
        "best_objective": stop["best_objective"], "best_iteration": stop.get("best_iteration"),
        "n_completed": stop["n_completed"], "since_improvement": stop["since_improvement"],
        "stop": stop["stop"], "stop_reason": stop["reason"],
    }, default=str))


# ---------------------------------------------------------------------------
# diagnose / history / show
# ---------------------------------------------------------------------------
def cmd_diagnose(args) -> int:
    """Re-derive diagnostics without simulating — from out/, or from a recorded iteration."""
    spec = _prepare(args)
    if args.iteration is not None:
        record = next((r for r in cc.read_ledger(spec) if r["iteration"] == args.iteration), None)
        if record is None:
            raise SystemExit(f"no iteration {args.iteration} in the ledger")
        print(json.dumps({"iteration": record["iteration"], "objective": record["objective"],
                          "metrics": record["metrics"], "diagnostics": record["diagnostics"],
                          "figures": record["figures"]}, indent=2, default=str))
        return 0

    iter_dir = spec.ledger_dir / "adhoc"
    iter_dir.mkdir(parents=True, exist_ok=True)
    objective, metrics, diagnostics, figures = _evaluate(spec, iter_dir)
    print(json.dumps({"objective": objective, "metrics": metrics,
                      "diagnostics": diagnostics, "figures": figures},
                     indent=2, default=str))
    return 0


def cmd_history(args) -> int:
    spec = cc.load_spec(Path(args.config), args.crop, args.target)
    records = cc.read_ledger(spec)
    if args.json:
        print(json.dumps(records, indent=2, default=str))
        return 0
    if not records:
        print(f"no iterations yet for {spec.crop} · {spec.target}")
        return 0
    print(f"\n  {spec.crop} · {spec.target} — {len(records)} iteration(s)\n")
    print(f"  {'it':>3s} {'objective':>10s} {'Δbest':>9s} {'changed':<44s} reason")
    print("  " + "-" * 108)
    for r in records:
        obj = f"{r['objective']:.4f}" if r.get("objective") is not None else "FAILED"
        delta = f"{r['delta_vs_best']:+.4f}" if r.get("delta_vs_best") is not None else ""
        mark = "+" if r.get("improved") else " "
        changed = ", ".join(sorted(r.get("parameters_changed", {}))) or "baseline"
        print(f" {mark}{r['iteration']:>3d} {obj:>10s} {delta:>9s} {changed[:44]:<44s} "
              f"{(r.get('reason') or '')[:52]}")
    best = cc.best_record(spec)
    if best:
        print(f"\n  best: iteration {best['iteration']}  objective {best['objective']:.4f}")
        print(f"  crop.xml: {spec.ledger_dir / 'best_crop.xml'}")
    print()
    return 0


def cmd_show(args) -> int:
    spec = cc.load_spec(Path(args.config), args.crop, args.target)
    record = next((r for r in cc.read_ledger(spec) if r["iteration"] == args.iteration), None)
    if record is None:
        raise SystemExit(f"no iteration {args.iteration} in the ledger")
    print(json.dumps(record, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# handoff / verify-lai / promote
# ---------------------------------------------------------------------------
def cmd_handoff(args) -> int:
    """Seed yield calibration from the LAI-calibrated crop.xml.

    This is the LAI -> yield edge of the workflow. Without it, yield calibration
    would start from the pre-LAI parameters and the LAI work would be lost.
    """
    lai = cc.load_spec(Path(args.config), args.crop, "lai")
    best_lai = lai.ledger_dir / "best_crop.xml"
    if not best_lai.exists():
        raise SystemExit(
            f"no calibrated LAI crop.xml at {best_lai}\n"
            f"  run the LAI calibration first: calibrate.py status --crop {args.crop} --target lai")

    best_record = cc.best_record(lai)
    yld = _prepare(args, target="yield")

    if (yld.ledger_dir / "ledger.jsonl").exists() and not args.force:
        raise SystemExit(
            f"yield calibration for {args.crop} already has a ledger at {yld.ledger_path}.\n"
            f"  Handing off now would change the starting point mid-study. Pass --force if "
            f"that is what you intend (the previous ledger is kept, not deleted).")

    shutil.copyfile(best_lai, yld.crop_xml)
    cc.refresh_space(yld)
    # Re-anchor the freeze snapshot on the LAI-calibrated file: for the yield
    # target the frozen set includes the LAI parameters, whose values are the ones
    # that were just handed over.
    snapshot = cc.snapshot_frozen(yld.crop_xml, yld.crop_name, yld.frozen)
    yld.frozen_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    shutil.copyfile(yld.crop_xml, yld.ledger_dir / "baseline_crop.xml")
    shutil.copyfile(yld.crop_xml, yld.ledger_dir / "current_crop.xml")

    provenance = {
        "handoff_at": cc.utcnow(),
        "from": {"target": "lai", "crop_xml": str(best_lai),
                 "iteration": None if not best_record else best_record["iteration"],
                 "objective": None if not best_record else best_record["objective"]},
        "frozen_digest": snapshot["digest"],
        "frozen_parameters": yld.frozen,
    }
    (yld.ledger_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    state = cc.read_state(yld)
    state.update({"crop": args.crop, "target": "yield", "created": cc.utcnow(),
                  "handoff": provenance})
    cc.write_state(yld, state)

    print(f"\n  yield calibration seeded from the LAI-calibrated crop.xml")
    print(f"    source     {best_lai}")
    if best_record:
        print(f"    LAI best   iteration {best_record['iteration']}  "
              f"objective {best_record['objective']:.4f}")
    print(f"    target     {yld.crop_xml}")
    print(f"    frozen     {len(yld.frozen)} parameters (phenology + the LAI set)")
    print(f"\n  Next: calibrate.py run --crop {args.crop} --target yield   "
          f"(iteration 0 = baseline)\n")
    return 0


def cmd_verify_lai(args) -> int:
    """Check that the yield-calibrated parameters have not undone the LAI calibration.

    Several yield parameters (RUE, KDIF, partitioning) legitimately move LAI. This
    stages the *LAI* target with the current yield crop.xml, runs it, and compares
    the LAI objective against the LAI calibration's best.
    """
    yld = cc.load_spec(Path(args.config), args.crop, "yield")
    if not yld.crop_xml.exists():
        raise SystemExit(f"no yield calibration crop.xml at {yld.crop_xml}")

    lai_cfg = cc.load_calib_config(Path(args.config)).get("lai_regression", {})
    args.locations = args.locations or lai_cfg.get("n_locations")
    lai = _prepare(args, target="lai")

    reference = cc.best_record(lai)
    if reference is None:
        print("  note: the LAI calibration has no completed iterations; reporting the "
              "current LAI objective without a comparison.")

    saved = lai.ledger_dir / "pre_verify_crop.xml"
    shutil.copyfile(lai.crop_xml, saved)
    shutil.copyfile(yld.crop_xml, lai.crop_xml)

    out_dir = lai.ledger_dir / "lai_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.skip_run:
            print(f"\n  running the LAI target with the yield-calibrated crop.xml ...")
            common.run_simplace(lai.run, 9000, log_path=out_dir / "simplace.log")
        objective, metrics, diagnostics, figures = _evaluate(lai, out_dir)
    finally:
        shutil.copyfile(saved, lai.crop_xml)
        saved.unlink(missing_ok=True)

    max_degradation = float(lai_cfg.get("max_relative_degradation", 0.10))
    verdict = {"lai_objective_now": objective,
               "lai_objective_calibrated": None if not reference else reference["objective"],
               "max_relative_degradation": max_degradation}
    if reference:
        rel = (objective - reference["objective"]) / max(reference["objective"], 1e-9)
        verdict["relative_degradation"] = round(rel, 4)
        verdict["pass"] = bool(rel <= max_degradation)
    record = {"timestamp": cc.utcnow(), "crop": args.crop, "source": "yield",
              "yield_iteration": cc.next_iteration(yld) - 1,
              "verdict": verdict, "metrics": metrics, "figures": figures}
    with open(yld.ledger_dir / "lai_regression.jsonl", "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")

    print("\n" + "=" * 78)
    print(f"  LAI REGRESSION CHECK  {args.crop}")
    print("=" * 78)
    print(f"  LAI objective with the yield parameters   {objective:.4f}")
    if reference:
        print(f"  LAI objective after LAI calibration       {reference['objective']:.4f} "
              f"(iteration {reference['iteration']})")
        print(f"  relative change                           {verdict['relative_degradation']:+.1%} "
              f"(allowed {max_degradation:+.0%})")
        print(f"  verdict                                   "
              f"{'PASS' if verdict['pass'] else 'FAIL — the yield changes degraded LAI'}")
    print("=" * 78)
    print("JSON " + json.dumps(verdict, default=str))
    return 0 if verdict.get("pass", True) else 1


def cmd_promote(args) -> int:
    """Copy a calibration's best crop.xml into the production crop inputs."""
    spec = cc.load_spec(Path(args.config), args.crop, args.target)
    best_xml = spec.ledger_dir / "best_crop.xml"
    if not best_xml.exists():
        raise SystemExit(f"nothing to promote: {best_xml} does not exist")

    production = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    snapshot = cc.snapshot_frozen(production, spec.crop_name, spec.frozen)
    drift = cc.verify_frozen(best_xml, snapshot)
    if drift:
        print("  REFUSING to promote — the candidate differs from production in frozen "
              "parameters:")
        for line in drift:
            print(f"    {line}")
        return 3

    best = cc.best_record(spec)
    changed = cc.changed_entries(
        cc.current_values(best_xml, spec.crop_name, spec.space),
        cc.current_values(production, spec.crop_name, spec.space))

    print(f"\n  promote {spec.crop} · {spec.target}")
    print(f"    from   {best_xml}")
    print(f"    to     {production}")
    if best:
        print(f"    best   iteration {best['iteration']}  objective {best['objective']:.4f}")
    print(f"    frozen phenology unchanged: yes ({len(spec.frozen)} parameters)")
    print("    parameter changes vs production:")
    for pid, change in changed.items():
        print(f"      {pid}  {_fmt(change['from'])} -> {_fmt(change['to'])}")
    if not changed:
        print("      (none)")

    if not args.yes:
        print("\n  Nothing written. Re-run with --yes to apply.\n")
        return 0

    backup = production.with_suffix(f".xml.pre_{spec.target}_calibration")
    if not backup.exists():
        shutil.copyfile(production, backup)
        print(f"\n  backup written: {backup}")
    shutil.copyfile(best_xml, production)
    print(f"  promoted.\n")
    return 0


def cmd_restore_optimized(args) -> int:
    """Put a protected target's preserved optimized values back into the run dir.

    The escape hatch for a recalibration that went wrong: ``optimized_baseline.json``
    holds the values as they stood before the first agentic iteration, so the
    original phenology can always be recovered from the ledger alone.
    """
    spec = _prepare(args)
    if not spec.protected:
        raise SystemExit(f"target {spec.target!r} is not protected; nothing is preserved for it")
    if not spec.protected_path.exists():
        raise SystemExit(
            f"no preserved values at {spec.protected_path}\n"
            f"  They are written on the first run of this target; if it has never run, "
            f"the production crop.xml is still the optimized set.")

    preserved = json.loads(spec.protected_path.read_text())
    drift = cc.protected_drift(spec)

    print(f"\n  restore {spec.crop} · {spec.target}")
    print(f"    from   {spec.protected_path}  (taken {preserved.get('taken_at')})")
    print(f"    to     {spec.crop_xml}")
    if not drift:
        print("\n  Already identical to the preserved optimized values. Nothing to do.\n")
        return 0
    print(f"    {len(drift)} parameter(s) currently differ:")
    for line in drift:
        print(f"      {line}")

    if not args.yes:
        print("\n  Nothing written. Re-run with --yes to restore.\n")
        return 0

    common.apply_parameters(spec.crop_xml, spec.crop_name, preserved["parameters"])
    remaining = cc.protected_drift(spec)
    if remaining:
        print("\n  RESTORE INCOMPLETE — these still differ after the write:")
        for line in remaining:
            print(f"      {line}")
        return 1
    shutil.copyfile(spec.crop_xml, spec.ledger_dir / "current_crop.xml")
    print("\n  restored. The ledger is untouched — the iterations that moved these "
          "values are still recorded.\n")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def shared(parser, target: bool = True):
        parser.add_argument("--crop", default="winter_wheat")
        if target:
            parser.add_argument("--target", default="lai",
                                choices=["phenology", "lai", "yield"])
        parser.add_argument("--config", default=str(cc.DEFAULT_CALIB_CONFIG))
        return parser

    p = shared(sub.add_parser("status", help="current parameters, bounds, history, stopping"))
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int, help="override the subset size")
    p.add_argument("--rebuild", action="store_true", help="rebuild the run dir from scratch")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = shared(sub.add_parser("run", help="apply a proposal and run one iteration"))
    p.add_argument("--params", help='JSON, or @file.json. e.g. \'{"RGRLAI": 0.021}\' or '
                                    '\'{"SLATableSLA": {"3": 0.0118}}\'. Omit for a baseline run.')
    p.add_argument("--reason", default="", help="why this parameter was chosen (recorded)")
    p.add_argument("--hypothesis", default="", help="what the diagnostics suggested (recorded)")
    p.add_argument("--reasoning", default="", help="the agent's full reasoning (recorded)")
    p.add_argument("--expected-effect", default="", dest="expected_effect",
                   help="what the change should do to the metrics (recorded)")
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--skip-run", action="store_true",
                   help="score the outputs already in out/ instead of simulating")
    p.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    p.add_argument("--force", action="store_true",
                   help="apply despite constraint violations (recorded in the ledger)")
    p.add_argument("--allow-recalibration", action="store_true", dest="allow_recalibration",
                   help="permit writes to a protected target (phenology). Without it, a "
                        "protected target can only be validated, never changed.")
    p.set_defaults(func=cmd_run)

    p = shared(sub.add_parser("diagnose", help="diagnostics without simulating"))
    p.add_argument("--iteration", type=int, help="report a recorded iteration instead")
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.set_defaults(func=cmd_diagnose, rebuild=False)

    p = shared(sub.add_parser("history", help="the calibration ledger"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_history)

    p = shared(sub.add_parser("show", help="one ledger record in full"))
    p.add_argument("--iteration", type=int, required=True)
    p.set_defaults(func=cmd_show)

    p = shared(sub.add_parser("handoff", help="seed yield calibration from the LAI result"),
               target=False)
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--force", action="store_true", help="re-seed an existing yield study")
    p.set_defaults(func=cmd_handoff, target="yield")

    p = shared(sub.add_parser("verify-lai", help="has yield calibration degraded LAI?"),
               target=False)
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--skip-run", action="store_true")
    p.set_defaults(func=cmd_verify_lai, target="lai")

    p = shared(sub.add_parser("promote", help="copy the best crop.xml into production"))
    p.add_argument("--yes", action="store_true", help="actually write (default: show the diff)")
    p.set_defaults(func=cmd_promote)

    p = shared(sub.add_parser(
        "restore-optimized",
        help="put a protected target's preserved optimized values back"))
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--yes", action="store_true", help="actually write (default: show the diff)")
    p.set_defaults(func=cmd_restore_optimized, rebuild=False, target="phenology")

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
