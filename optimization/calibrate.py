#!/usr/bin/env python
"""SIMPLACE calibration — the tool an agent drives, one iteration at a time.

There is no sampler here. Every parameter change comes from a proposal someone
writes, with a stated reason; this script validates it, runs the model, scores it
with the losses in ``objectives.py``, produces the diagnostics needed to decide
the next move, and appends an immutable record to the ledger.

Two stages, in this order:

    stage 1  --target phenology   thermal time, calibrated from scratch
    stage 2  --target growth      LAI and yield, calibrated JOINTLY with the
                                  stage-1 phenology frozen

    calibrate.py status   --crop winter_wheat --target phenology
    calibrate.py run      --crop winter_wheat --target phenology \
                          --params '{"TSUM1": 1180}' --reason "flowering 4 d early"
    calibrate.py promote  --crop winter_wheat --target phenology --yes
    calibrate.py handoff  --crop winter_wheat
    calibrate.py run      --crop winter_wheat --target growth
    calibrate.py promote  --crop winter_wheat --target growth --yes

Iteration 0 is the baseline: run it with no ``--params`` to record the objective
of the starting point, so every later iteration has a reference to beat.

A ``growth`` iteration runs the model **twice** — once on the GLASS-LAI point set
and once on the district yield point set — from one ``crop.xml``, and scores a
single combined objective. That is what makes the interdependent LAI and yield
parameters calibratable together instead of one undoing the other.

Guarantees
----------
* The frozen parameters of a stage are snapshotted on first use and re-verified
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as cc  # noqa: E402
import calib_diagnostics as cd  # noqa: E402
import common  # noqa: E402
import evaluation  # noqa: E402

TARGETS = ["phenology", "growth"]


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
    """Load the spec and stage the isolated run dir of every view (idempotent)."""
    spec = cc.load_spec(Path(args.config), args.crop, target or args.target,
                        device=getattr(args, "device", "cluster"),
                        n_locations=getattr(args, "locations", None))
    spec.stage_info = cc.stage(spec, rebuild=getattr(args, "rebuild", False))
    return spec


def _baseline_files(spec: cc.CalibSpec) -> None:
    """Record the starting crop.xml and the freeze snapshot, once per study."""
    baseline = spec.ledger_dir / "baseline_crop.xml"
    if not baseline.exists():
        shutil.copyfile(spec.crop_xml, baseline)
    cc.ensure_frozen_snapshot(spec)


def _scope(spec: cc.CalibSpec) -> dict:
    """What was simulated, per view — recorded with every iteration."""
    return {view: {"project_csv": str(spec.runs[view].project_csv),
                   "rows": (spec.stage_info.get(view) or {}).get("rows"),
                   "locations": (spec.stage_info.get(view) or {}).get("locations"),
                   "subset": spec.runs[view].subset,
                   "climate": spec.runs[view].mount_data}
            for view in spec.views}


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
        "views": spec.views,
        "objective": spec.objective_name,
        "objective_components": spec.components,
        "scope": _scope(spec),
        "run_dirs": {view: str(spec.runs[view].run_dir) for view in spec.views},
        "ledger_dir": str(spec.ledger_dir),
        "frozen": {
            "parameters": spec.frozen,
            "digest": snapshot["digest"],
            "intact": not drift,
            "drift": drift,
        },
        "next_iteration": cc.next_iteration(spec),
        "n_completed": stop["n_completed"],
        # Which parameter set the values above are, and what happens to a change
        # that does not improve. The agent has to know this: under `best`, a
        # rejected change is already undone by the time it reads the history.
        "search": {
            "acceptance": spec.acceptance,
            "starts_from": "the best parameter set so far" if spec.acceptance == "best"
                           else "the most recent parameter set",
            "current_matches_best": (spec.best_xml.exists()
                                     and spec.best_xml.read_bytes() == spec.crop_xml.read_bytes()),
            "base_iteration": None if not best else best["iteration"],
        },
        "best": None if not best else {
            "iteration": best["iteration"], "objective": best["objective"],
            "components": best.get("objective_components"),
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
    print(f"  ledger             {spec.ledger_dir}")
    print(f"  objective          {spec.objective_name}")
    for view in spec.views:
        info = spec.stage_info.get(view) or {}
        weight = spec.components.get(view, {})
        print(f"  view {view:<13s} {info.get('rows', 0):,} rows / "
              f"{info.get('locations', 0):,} locations   "
              f"weight {weight.get('weight')}, scale {weight.get('scale')}")
        print(f"  {'':18s} {spec.runs[view].run_dir}")
    print(f"  frozen             {'INTACT' if not drift else 'DRIFTED!'} "
          f"({len(spec.frozen)} parameters, digest {snapshot['digest'][:12]})")
    for line in drift:
        print(f"     ! {line}")

    print(f"\n  iterations completed  {stop['n_completed']}")
    if best:
        print(f"  best                  iteration {best['iteration']}  "
              f"objective {best['objective']:.4f}")
    print(f"  acceptance            {spec.acceptance} — the next iteration changes "
          f"{payload['search']['starts_from']}")
    if best and not payload["search"]["current_matches_best"]:
        print(f"  ! the staged crop.xml differs from best_crop.xml (iteration "
              f"{best['iteration']}); it was edited outside calibrate.py")
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
def cmd_run(args) -> int:
    spec = _prepare(args)
    _baseline_files(spec)
    snapshot = cc.ensure_frozen_snapshot(spec)

    current = cc.current_values(spec.crop_xml, spec.crop_name, spec.space)
    proposed_raw = _load_json_arg(args.params)
    try:
        proposal = cc.normalise_proposal(proposed_raw, current, spec.space)
    except SystemExit as exc:
        # A proposal that cannot even be parsed is still a verdict a caller needs
        # in machine-readable form. Dying with a stderr message here is what turns
        # an agent's recoverable mistake — a mis-shaped table edit — into a crash
        # of the whole calibration loop, because the pre-flight gets nothing it
        # can hand back to the model.
        if not args.dry_run:
            raise
        print(json.dumps({
            "iteration": cc.next_iteration(spec), "valid": False, "would_change": {},
            "violations": [{"constraint": "proposal_shape", "parameter": None,
                            "message": " ".join(str(exc).split())}],
        }, indent=2, default=str))
        return 2
    changed = cc.changed_entries(proposal, current)

    iteration = cc.next_iteration(spec)
    if iteration == 0 and changed:
        print("  note: iteration 0 usually records the unchanged baseline. Proceeding with "
              "the proposed change; the ledger will have no reference objective.")

    # --- validate before anything is written ---------------------------------
    violations = cc.validate(proposal, current, spec.space, spec.meta, spec.constraints,
                             spec.frozen, spec.crop_xml, spec.crop_name)

    # A dry run answers "would this be accepted?" and writes nothing at all — not
    # even to rejected.jsonl, because a pre-flight check is not a proposal. This is
    # what an agent calls before spending cluster time on a change that cannot pass.
    if args.dry_run:
        print(json.dumps({
            "iteration": iteration, "valid": not violations,
            "would_change": changed,
            "violations": [v.as_dict() for v in violations],
        }, indent=2, default=str))
        return 0 if not violations else 2

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
    # The set this iteration is a change *to*, and the one it rolls back to. Under
    # `acceptance: best` it is always the best-so-far, so a rejected change never
    # ends up in the baseline of the next hypothesis.
    base_xml = spec.base_xml
    if not base_xml.exists():
        shutil.copyfile(spec.crop_xml, base_xml)

    # --- apply, then verify the frozen set on the written file ---------------
    if proposal:
        common.apply_parameters(spec.crop_xml, spec.crop_name, proposal)
    drift = cc.verify_frozen(spec.crop_xml, snapshot)
    if drift:
        shutil.copyfile(base_xml, spec.crop_xml)
        cc.sync_crop_xml(spec)
        print("\n  ABORTED — the write changed frozen parameters:")
        for line in drift:
            print(f"    {line}")
        print("  crop.xml was rolled back. Nothing was run.\n")
        return 3

    # Every view must simulate the same parameter set, or the components of the
    # objective would describe different crops.
    cc.sync_crop_xml(spec)
    shutil.copyfile(spec.crop_xml, iter_dir / "crop.xml")
    (iter_dir / "proposal.json").write_text(json.dumps({
        "iteration": iteration, "proposal": proposed_raw,
        "normalised": proposal, "changed": changed,
        "reason": args.reason, "hypothesis": args.hypothesis,
        "reasoning": args.reasoning, "expected_effect": args.expected_effect,
    }, indent=2, default=str) + "\n")

    print(f"\n[{spec.crop} · {spec.target}] iteration {iteration}")
    for view in spec.views:
        info = spec.stage_info.get(view) or {}
        print(f"  view {view:<10s} {info.get('rows', 0):,} rows / "
              f"{info.get('locations', 0):,} locations   {spec.runs[view].run_dir}")
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
    result = None
    try:
        if args.skip_run:
            print("\n  --skip-run: scoring the outputs already in out/, no simulation")
        else:
            for view in spec.views:
                print(f"\n  running SIMPLACE · view {view} ({spec.runs[view].device}) ...")
                common.run_simplace(spec.runs[view], iteration,
                                    log_path=iter_dir / f"simplace_{view}.log")
        result = evaluation.evaluate(spec, iter_dir)
        objective, metrics = result.objective, result.metrics
        diagnostics, figures = result.diagnostics, result.figures
    except Exception as exc:  # noqa: BLE001 — any failure must roll back and be recorded
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        shutil.copyfile(base_xml, spec.crop_xml)
        cc.sync_crop_xml(spec)
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
    improved = is_best = accepted = None
    if status == "completed":
        improved = (previous_best is None
                    or objective < previous_best["objective"] - min_improvement)
        is_best = previous_best is None or objective < previous_best["objective"]
        # Third question, and the one that decides what the *next* iteration is a
        # change to: is this set kept, or undone in favour of the best so far?
        accepted = cc.accept(spec, is_best)

    record = {
        "iteration": iteration,
        "timestamp": cc.utcnow(),
        "crop": spec.crop,
        "crop_name": spec.crop_name,
        "target": spec.target,
        "views": spec.views,
        "status": status,
        "error": error,
        "scope": _scope(spec),
        "parameters": {k: v for k, v in {**current, **proposal}.items()},
        "parameters_changed": {k: v["to"] for k, v in changed.items()},
        "previous_values": {k: v["from"] for k, v in changed.items()},
        "changed_detail": changed,
        "objective": objective,
        "objective_name": spec.objective_name,
        "objective_components": None if result is None else result.breakdown,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "reason": args.reason,
        "hypothesis": args.hypothesis,
        "agent_reasoning": args.reasoning,
        "expected_effect": args.expected_effect,
        "improved": improved,
        "is_best": is_best,
        # Whether these parameters survived the iteration. False means crop.xml was
        # put back to best_crop.xml, so the next iteration is a change to the best
        # set and not to this one.
        "accepted": accepted,
        "acceptance": spec.acceptance,
        "reverted_to_best": (None if accepted is not False or not previous_best else {
            "iteration": previous_best["iteration"], "objective": previous_best["objective"]}),
        "previous_best": None if not previous_best else {
            "iteration": previous_best["iteration"], "objective": previous_best["objective"]},
        "delta_vs_best": (None if (status != "completed" or previous_best is None)
                          else round(objective - previous_best["objective"], 6)),
        "frozen_digest": snapshot["digest"],
        "frozen_intact": not cc.verify_frozen(spec.crop_xml, snapshot),
        "constraint_violations": [v.as_dict() for v in violations],
        "forced": bool(args.force),
        "figures": figures or [],
        "iteration_dir": str(iter_dir),
        "elapsed_seconds": elapsed,
    }
    cc.append_ledger(spec, record)
    (iter_dir / "metrics.json").write_text(json.dumps({
        "objective": objective,
        "objective_components": None if result is None else result.breakdown,
        "metrics": metrics, "diagnostics": diagnostics,
    }, indent=2, default=str) + "\n")

    if status == "completed":
        if is_best:
            shutil.copyfile(spec.crop_xml, spec.best_xml)
        if not accepted:
            # Rejected: undo it. Without this the next proposal would be a change
            # to a parameter set already known to be worse, and its effect would no
            # longer be attributable to the one thing it changed.
            cc.restore_best(spec)
        shutil.copyfile(spec.crop_xml, base_xml)

    state = cc.read_state(spec)
    best_now = cc.best_record(spec)
    state.update({
        "crop": spec.crop, "target": spec.target, "updated": cc.utcnow(),
        "n_iterations": cc.next_iteration(spec),
        "last_iteration": iteration, "last_status": status,
        "best_iteration": None if not best_now else best_now["iteration"],
        "best_objective": None if not best_now else best_now["objective"],
        "frozen_digest": snapshot["digest"],
        "run_dirs": {view: str(spec.runs[view].run_dir) for view in spec.views},
    })
    state.setdefault("created", cc.utcnow())
    cc.write_state(spec, state)

    cd.history_plot(cc.read_ledger(spec), spec.ledger_dir / "history.png",
                    f"{spec.crop} · {spec.target} · objective per iteration")

    _report(spec, record)
    return 0 if status == "completed" else 1


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _report_phenology(diagnostics: dict) -> None:
    for name in ("flowering", "maturity", "duration"):
        m = diagnostics.get(name) or {}
        if m.get("n"):
            print(f"    {name:<12s} RMSE {m['RMSE']:>6.2f} d   bias {m['bias']:>+6.2f} d   "
                  f"R2 {m.get('R2') if m.get('R2') is None else round(m['R2'], 3)}   "
                  f"n={m['n']:,}")
    structure = diagnostics.get("residual_structure") or {}
    if structure:
        print(f"    warmth slope  {structure.get('vs_season_warmth_slope')} d/d   "
              f"latitude slope {structure.get('vs_latitude_slope')} d/deg")
    if diagnostics.get("attribution"):
        print(f"    attribution   {diagnostics['attribution']['verdict']}")


def _report_lai(diagnostics: dict) -> None:
    if "by_dvs_bin" in diagnostics:
        print(f"    {'DVS bin':<26s} {'n':>6s} {'obs':>8s} {'sim':>8s} {'bias':>8s} {'rmse':>8s}")
        for row in diagnostics["by_dvs_bin"]:
            print(f"    {row['dvs_bin']:<26s} {row['n']:>6} {row['obs_mean']:>8.3f} "
                  f"{row['sim_mean']:>8.3f} {row['bias']:>+8.3f} {row['rmse']:>8.3f}")
    shape = diagnostics.get("curve_shape", {})
    for name in ("peak_lai", "peak_timing", "plateau_duration", "rise_rate", "decline_rate",
                 "early_canopy"):
        item = shape.get(name)
        if isinstance(item, dict) and item.get("n"):
            print(f"    {name:<26s} obs {item['obs_median']:>8.4g}   sim {item['sim_median']:>8.4g}"
                  f"   diff {item['median_diff']:>+8.4g}  [{item.get('unit', '')}]")


def _report_yield(diagnostics: dict) -> None:
    a = diagnostics.get("attribution")
    if a:
        print(f"    harvest index sim {a['hi_simulated_median']:.3f}   "
              f"required {a['hi_required_to_match_obs']:.3f}   ref {a['reference_harvest_index']}")
        print(f"    AG biomass    sim {a['ag_biomass_simulated_median']:.2f}   "
              f"required {a['ag_biomass_required_to_match_obs']:.2f}   "
              f"ref {a['reference_ag_biomass_t_ha']}")
        print(f"    attribution   {a['verdict']}")
    t = diagnostics.get("translocation")
    if t:
        print(f"    translocation Yield_t_ha RMSE {t['yield_plain']['RMSE']:.3f} t/ha   "
              f"translocated RMSE {t['yield_translocated']['RMSE']:.3f} t/ha   "
              f"FRTDM {t['median_translocation_contribution_t_ha']:+.3f} t/ha")
        print(f"    FRTDM verdict {t['verdict']}")


VIEW_REPORTS = {"phenology": _report_phenology, "lai": _report_lai, "yield": _report_yield}


def _report(spec: cc.CalibSpec, record: dict) -> None:
    """Human summary plus the one-line JSON the calling agent reads back."""
    print("\n" + "=" * 78)
    print(f"ITERATION {record['iteration']}  {spec.crop} · {spec.target}  [{record['status']}]")
    print("=" * 78)
    for key, value in (record.get("metrics") or {}).items():
        print(f"  {key:<40s} {value}")

    components = record.get("objective_components") or {}
    if components:
        print()
        for view, c in components.items():
            print(f"  {view + ' component':<40s} {c['loss']:.4f} / scale {c['scale']} "
                  f"= {c['scaled']:.3f}  (weight {c['weight']})")
    if record["objective"] is not None:
        print(f"  {'OBJECTIVE':<40s} {record['objective']:.4f}")
        if record["previous_best"]:
            if record["improved"]:
                arrow = "IMPROVED"
            elif record["is_best"]:
                arrow = "new best, but below the min_improvement threshold"
            else:
                arrow = "no improvement"
            print(f"  {'previous best':<40s} {record['previous_best']['objective']:.4f} "
                  f"(iter {record['previous_best']['iteration']})  -> {arrow} "
                  f"({record['delta_vs_best']:+.4f})")
    if record["status"] == "completed":
        if record["accepted"]:
            print(f"  {'parameters':<40s} KEPT — the next iteration starts from these")
        else:
            back = record["reverted_to_best"] or {}
            print(f"  {'parameters':<40s} REJECTED — rolled back to the best set "
                  f"(iteration {back.get('iteration')}, objective "
                  f"{back.get('objective', float('nan')):.4f})")
            print(f"  {'':40s} the next iteration is a change to THAT set, not to this one")
    print(f"  {'frozen parameters intact':<40s} {record['frozen_intact']}")
    print(f"  {'elapsed':<40s} {record['elapsed_seconds']}s")
    print(f"  {'iteration dir':<40s} {record['iteration_dir']}")

    diagnostics = record.get("diagnostics") or {}
    for view in spec.views:
        block = diagnostics.get(view) or {}
        report = VIEW_REPORTS.get(view)
        if block and report:
            print(f"\n  --- {view} ---")
            report(block)

    stop = cc.stop_check(spec)
    print(f"\n  stopping: {'STOP — ' if stop['stop'] else 'continue — '}{stop['reason']}")
    print("=" * 78)
    print("JSON " + json.dumps({
        "iteration": record["iteration"], "status": record["status"],
        "objective": record["objective"],
        "objective_components": {v: c["loss"] for v, c in (components or {}).items()},
        "improved": record["improved"],
        "accepted": record["accepted"],
        "reverted_to_best": record["reverted_to_best"],
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
                          "objective_components": record.get("objective_components"),
                          "metrics": record["metrics"], "diagnostics": record["diagnostics"],
                          "figures": record["figures"]}, indent=2, default=str))
        return 0

    result = evaluation.evaluate(spec, spec.ledger_dir / "adhoc")
    print(json.dumps({"objective": result.objective, "objective_components": result.breakdown,
                      "metrics": result.metrics, "diagnostics": result.diagnostics,
                      "figures": result.figures}, indent=2, default=str))
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
        for view, c in (best.get("objective_components") or {}).items():
            print(f"        {view:<10s} loss {c['loss']}")
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
# handoff / promote / restore-baseline
# ---------------------------------------------------------------------------
def cmd_handoff(args) -> int:
    """Seed the growth stage from the phenology-calibrated crop.xml.

    This is the stage 1 -> stage 2 edge. Without it, the joint LAI+yield stage
    would start from the pre-calibration thermal time and every canopy and yield
    parameter would be tuned against the wrong development clock.
    """
    phen = cc.load_spec(Path(args.config), args.crop, "phenology")
    best_xml = phen.ledger_dir / "best_crop.xml"
    if not best_xml.exists():
        raise SystemExit(
            f"no calibrated phenology crop.xml at {best_xml}\n"
            f"  Run stage 1 first: calibrate.py status --crop {args.crop} --target phenology")

    best_record = cc.best_record(phen)
    growth = _prepare(args, target="growth")

    if growth.ledger_path.exists() and not args.force:
        raise SystemExit(
            f"the growth calibration for {args.crop} already has a ledger at "
            f"{growth.ledger_path}.\n"
            f"  Handing off now would change the starting point mid-study. Pass --force if "
            f"that is what you intend (the previous ledger is kept, not deleted).")

    shutil.copyfile(best_xml, growth.crop_xml)
    cc.sync_crop_xml(growth)
    cc.refresh_space(growth)
    # Re-anchor the freeze snapshot on the phenology-calibrated file: the frozen
    # set for this stage is exactly what was just handed over.
    snapshot = cc.snapshot_frozen(growth.crop_xml, growth.crop_name, growth.frozen)
    growth.frozen_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    shutil.copyfile(growth.crop_xml, growth.ledger_dir / "baseline_crop.xml")
    shutil.copyfile(growth.crop_xml, growth.ledger_dir / "current_crop.xml")

    provenance = {
        "handoff_at": cc.utcnow(),
        "from": {"target": "phenology", "crop_xml": str(best_xml),
                 "iteration": None if not best_record else best_record["iteration"],
                 "objective": None if not best_record else best_record["objective"]},
        "frozen_digest": snapshot["digest"],
        "frozen_parameters": growth.frozen,
    }
    (growth.ledger_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    state = cc.read_state(growth)
    state.update({"crop": args.crop, "target": "growth", "created": cc.utcnow(),
                  "handoff": provenance})
    cc.write_state(growth, state)

    print(f"\n  growth calibration seeded from the phenology-calibrated crop.xml")
    print(f"    source     {best_xml}")
    if best_record:
        print(f"    phenology  iteration {best_record['iteration']}  "
              f"objective {best_record['objective']:.4f} (RMSE days)")
    print(f"    target     {growth.crop_xml}")
    for view in growth.views[1:]:
        print(f"               {growth.runs[view].crop_xml}  (mirror)")
    print(f"    frozen     {len(growth.frozen)} parameters (phenology + the response grids)")

    # Promoting stage 1 is what makes `promote --target growth` possible later:
    # promote refuses when the candidate differs from production in a frozen
    # parameter, and after a phenology calibration it always does.
    production = growth.run.crop_dir / "data" / "crop" / "crop.xml"
    if cc.verify_frozen(production, snapshot):
        print(f"\n  NOTE: the production crop.xml does not yet carry the calibrated "
              f"phenology.\n  Promote stage 1 before promoting stage 2:\n"
              f"    python optimization/calibrate.py promote --crop {args.crop} "
              f"--target phenology --yes")
    print(f"\n  Next: calibrate.py run --crop {args.crop} --target growth   "
          f"(iteration 0 = baseline)\n")
    return 0


def cmd_promote(args) -> int:
    """Copy a stage's best crop.xml into the production crop inputs."""
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
        if spec.target == "growth":
            print(f"\n  This is what it looks like when stage 1 was calibrated but never "
                  f"promoted.\n  Promote it first:\n"
                  f"    python optimization/calibrate.py promote --crop {spec.crop} "
                  f"--target phenology --yes")
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
        for view, c in (best.get("objective_components") or {}).items():
            print(f"           {view:<10s} loss {c['loss']}")
    print(f"    frozen parameters unchanged: yes ({len(spec.frozen)} parameters)")
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


def cmd_restore_baseline(args) -> int:
    """Put the stage's starting crop.xml back into the run dirs.

    The escape hatch for a calibration that went somewhere useless:
    ``baseline_crop.xml`` is the file as it stood before iteration 0, so the
    starting point is always recoverable from the ledger alone. The ledger itself
    is untouched — the iterations that moved the parameters stay on record.
    """
    spec = _prepare(args)
    baseline = spec.ledger_dir / "baseline_crop.xml"
    if not baseline.exists():
        raise SystemExit(
            f"no baseline at {baseline}\n"
            f"  It is written on the first run of this stage; if it has never run, the "
            f"production crop.xml is still the starting point.")

    changed = cc.changed_entries(
        cc.current_values(baseline, spec.crop_name, spec.space),
        cc.current_values(spec.crop_xml, spec.crop_name, spec.space))

    print(f"\n  restore {spec.crop} · {spec.target}")
    print(f"    from   {baseline}")
    print(f"    to     {spec.crop_xml}")
    if not changed:
        print("\n  Already identical to the baseline. Nothing to do.\n")
        return 0
    print(f"    {len(changed)} parameter(s) currently differ:")
    for pid, change in changed.items():
        print(f"      {pid}  {_fmt(change['from'])} -> {_fmt(change['to'])}")

    if not args.yes:
        print("\n  Nothing written. Re-run with --yes to restore.\n")
        return 0

    shutil.copyfile(baseline, spec.crop_xml)
    cc.sync_crop_xml(spec)
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
            parser.add_argument("--target", default="growth", choices=TARGETS)
        parser.add_argument("--config", default=str(cc.DEFAULT_CALIB_CONFIG))
        return parser

    p = shared(sub.add_parser("status", help="current parameters, bounds, history, stopping"))
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int, help="override the subset size")
    p.add_argument("--rebuild", action="store_true", help="rebuild the run dirs from scratch")
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

    p = shared(sub.add_parser(
        "handoff", help="seed the growth stage from the phenology result"), target=False)
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--force", action="store_true", help="re-seed an existing growth study")
    p.set_defaults(func=cmd_handoff, target="growth")

    p = shared(sub.add_parser("promote", help="copy the best crop.xml into production"))
    p.add_argument("--yes", action="store_true", help="actually write (default: show the diff)")
    p.set_defaults(func=cmd_promote)

    p = shared(sub.add_parser(
        "restore-baseline", help="put the stage's starting crop.xml back"))
    p.add_argument("--device", default="cluster", choices=["cluster", "local"])
    p.add_argument("--locations", type=int)
    p.add_argument("--yes", action="store_true", help="actually write (default: show the diff)")
    p.set_defaults(func=cmd_restore_baseline, rebuild=False)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
