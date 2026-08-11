#!/usr/bin/env python
"""Self-test for the agentic calibration machinery. No cluster, no SIMPLACE.

Covers everything that can go wrong without running the model: config loading,
per-crop space resolution, every constraint rule, the frozen-phenology guard,
proposal normalisation, the ledger, and the diagnostics on synthetic data.

    python optimization/test_calibrate.py

Deliberately a plain script with asserts rather than a pytest suite — the repo
has no test framework, and this needs to be runnable on the cluster login node
with nothing installed.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

OPTIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIM_DIR))

import calib_common as cc  # noqa: E402
import calib_diagnostics as cd  # noqa: E402
import common  # noqa: E402

CONFIG = OPTIM_DIR / "calibration.yaml"
CROPS = ["winter_wheat", "winter_rapeseed", "spring_barley", "potato", "maize"]
#: the two calibration stages, in order
TARGETS = ["phenology", "growth"]

_results: list[tuple[str, str, str]] = []


class Skip(Exception):
    """This check does not apply to the current configuration."""


def require_params(spec, *pids):
    """Skip a check whose parameters the config no longer declares.

    The parameter space is a tuning surface: a user may legitimately disable or
    delete any entry. A test that hard-codes one is asserting a configuration
    choice rather than an invariant, so it steps aside instead of failing.
    """
    space = set(spec.space.get("single_value_params") or {}) | \
        set(spec.space.get("multi_value_params") or {})
    missing = [p for p in pids if p not in space]
    if missing:
        raise Skip(f"not declared for {spec.crop}/{spec.target}: {', '.join(missing)}")


def test(name):
    def wrap(fn):
        try:
            fn()
            _results.append((name, "pass", ""))
            print(f"  PASS  {name}")
        except Skip as exc:
            _results.append((name, "skip", str(exc)))
            print(f"  SKIP  {name}: {exc}")
        # SystemExit is how calib_common reports a bad proposal, and it does not
        # derive from Exception — catching only Exception lets one such raise
        # abort the whole run silently mid-suite.
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            _results.append((name, "fail", traceback.format_exc()))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


def proposal_check(spec, raw, xml=None):
    """(violations, constraint ids) for a raw proposal against the production XML."""
    xml = xml or spec.run.crop_dir / "data" / "crop" / "crop.xml"
    current = cc.current_values(xml, spec.crop_name, spec.space)
    proposal = cc.normalise_proposal(raw, current, spec.space)
    violations = cc.validate(proposal, current, spec.space, spec.meta, spec.constraints,
                             spec.frozen, xml, spec.crop_name)
    return violations, {v.constraint for v in violations}


# ---------------------------------------------------------------------------
print("\nconfig + space")


@test("calibration.yaml extends config.yaml")
def _():
    cfg = cc.load_calib_config(CONFIG)
    assert sorted(cfg["targets"]) == ["growth", "phenology"]
    assert set(cfg["crops"]) == set(CROPS), "crops must come from the base config"
    assert "singularity_image" in cfg["slurm"]
    assert "input_profiles" in cfg
    assert "llm" in cfg, "the local-agent block must be part of the calibration config"


@test("no Optuna anywhere in the calibration layer")
def _():
    # The removal has to hold at the import level, not just in the config: an
    # `import optuna` left in a module would still pull the dependency in.
    for module in ("common.py", "calib_common.py", "calibrate.py", "objectives.py",
                   "calib_diagnostics.py", "agentic.py"):
        text = (OPTIM_DIR / module).read_text()
        assert "import optuna" not in text, f"{module} still imports optuna"
    assert not (OPTIM_DIR / "optimize_lai.py").exists()
    assert not (OPTIM_DIR / "optimize_yield.py").exists()
    assert not (OPTIM_DIR / "optimize_phenology.py").exists()
    assert not (OPTIM_DIR / "studies").exists(), "Optuna study dir should be gone"


@test("one loss per view, and an evaluator for each of them")
def _():
    import evaluation
    import objectives
    assert sorted(objectives.VIEWS) == ["lai", "phenology", "yield"]
    for view in objectives.VIEWS:
        process_result, loss_fn = objectives.for_view(view)
        assert callable(process_result) and callable(loss_fn)
        assert view in evaluation.EVALUATORS, f"{view} has a loss but no evaluator"
    # The DVS bins the LAI diagnostics key off must still be the ones the loss uses.
    assert objectives.DVS_BINS == [0.0, 0.25, 0.5, 1.0, 1.25, 1.50, 1.75, 2.0]


@test("every view of every stage is both simulated and scored")
def _():
    # A view with no weight would be simulated and silently ignored; a weight with
    # no view would score something that was never run. Both are refused at load.
    for target in TARGETS:
        spec = cc.load_spec(CONFIG, "winter_wheat", target)
        assert set(spec.components) == set(spec.views), (spec.views, spec.components)
        for view, cfg in spec.components.items():
            assert float(cfg["weight"]) > 0 and float(cfg["scale"]) > 0, (view, cfg)


@test("the joint objective is a weighted mean of the scaled component losses")
def _():
    import objectives
    components = {"lai": {"weight": 0.5, "scale": 0.15},
                  "yield": {"weight": 0.5, "scale": 0.5}}

    # Both components exactly at their scale -> 1.0, whatever their units.
    objective, breakdown = objectives.combine(components, {"lai": 0.15, "yield": 0.5})
    assert abs(objective - 1.0) < 1e-12, objective
    assert breakdown["lai"]["scaled"] == 1.0 and breakdown["yield"]["scaled"] == 1.0

    # Halving one component moves the objective by its weighted share and nothing
    # else — this is what stops one view being traded away silently.
    better_lai, _ = objectives.combine(components, {"lai": 0.075, "yield": 0.5})
    assert abs(better_lai - 0.75) < 1e-12, better_lai
    better_yield, _ = objectives.combine(components, {"lai": 0.15, "yield": 0.25})
    assert abs(better_yield - 0.75) < 1e-12, better_yield

    # A single-component stage with scale 1.0 reproduces the raw loss, so the
    # phenology objective stays an RMSE in days.
    single, _ = objectives.combine({"phenology": {"weight": 1.0, "scale": 1.0}},
                                   {"phenology": 6.4})
    assert single == 6.4

    # A missing component is an error, never a silently smaller objective.
    try:
        objectives.combine(components, {"lai": 0.2})
    except SystemExit:
        return
    raise AssertionError("a component with no loss must not be scored")


@test("space resolves for every crop and target")
def _():
    for crop in CROPS:
        for target in TARGETS:
            spec = cc.load_spec(CONFIG, crop, target)
            names = list(spec.space.get("single_value_params") or {}) + \
                list(spec.space.get("multi_value_params") or {})
            assert names, f"{crop}/{target} resolved an empty space"
            missing = [p for p, m in spec.meta.items() if not m.get("present")]
            assert not missing, f"{crop}/{target} parameters absent from crop.xml: {missing}"


@test("table bounds match each crop's own table length")
def _():
    # The five crops have different SLA/RUE table lengths; a hard-coded space
    # would silently truncate or overrun them.
    lengths = {}
    for crop in CROPS:
        spec = cc.load_spec(CONFIG, crop, "growth")
        xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
        actual = cc.read_param(cc.crop_root(xml), spec.crop_name, "SLATableSLA")
        resolved = spec.space["multi_value_params"]["SLATableSLA"]["values"]
        assert len(actual) == len(resolved), f"{crop}: {len(actual)} vs {len(resolved)}"
        lengths[crop] = len(actual)
    assert len(set(lengths.values())) > 1, "test is vacuous if all crops have equal tables"


@test("current values are always inside their own bounds")
def _():
    # Resolve the space against the *same* file the values are read from. Bounds
    # are relative to whatever crop.xml they were anchored on, so anchoring on a
    # staged run dir and then checking production values compares two different
    # parameter sets — which is a staleness question (covered below), not a
    # question about whether the declared bounds admit the shipped values.
    cfg = cc.load_calib_config(CONFIG)
    for crop in CROPS:
        for target in TARGETS:
            spec = cc.load_spec(CONFIG, crop, target)
            xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
            tcfg = cfg["targets"][target]
            space, meta = cc.resolve_space(xml, spec.crop_name, tcfg.get("parameters") or {})
            cc.apply_closure_locks(space, meta, tcfg.get("constraints") or [],
                                   spec.frozen, xml, spec.crop_name)
            current = cc.current_values(xml, spec.crop_name, space)
            outside = common.check_within_bounds(cc.flatten(current), space)
            assert not outside, f"{crop}/{target}: {outside} outside their bounds"


@test("a fresh study is re-anchored on the production crop.xml")
def _():
    # The failure this guards against: a run dir left behind by an earlier study
    # keeps its mutated crop.xml, so a new study records that as its baseline and
    # anchors every relative bound on it. stage() resets it while the ledger is
    # empty; once iterations exist it must never touch the file again.
    import inspect
    source = inspect.getsource(cc.stage)
    assert "ledger_path.exists()" in source, "stage() must check for an empty ledger"
    assert "provenance.json" in source, "the stage-1 handoff must be exempt from the reset"
    assert "rebuild" in source


@test("the growth stage freezes the whole stage-1 result and the table grids")
def _():
    phen = cc.load_spec(CONFIG, "winter_wheat", "phenology")
    growth = cc.load_spec(CONFIG, "winter_wheat", "growth")

    movable = set(phen.space.get("single_value_params") or {}) | \
        set(phen.space.get("multi_value_params") or {})
    # Everything stage 1 can move must be frozen in stage 2, or the joint
    # calibration could quietly re-time development and invalidate stage 1.
    assert movable <= set(growth.frozen), f"not frozen in growth: {movable - set(growth.frozen)}"
    for pid in ("TSUM1", "TSUM2", "TEFFMX", "SLATableDVS", "RUETableDVS"):
        assert pid in growth.frozen, f"{pid} not frozen"
    # The DVS grids are frozen in both stages: moving one would re-index every
    # response table and break the bin-to-element mapping the agents reason with.
    for pid in ("SLATableDVS", "RUETableDVS"):
        assert pid in phen.frozen, f"{pid} must stay frozen while calibrating phenology"

    # `frozen_exclude` un-freezes a parameter a stage needs as a structural
    # counterweight. Excusing something that is not in the space is a silently dead
    # config line, so assert the consistency rather than a fixed name.
    cfg = cc.load_calib_config(CONFIG)
    for target in TARGETS:
        spec = cc.load_spec(CONFIG, "winter_wheat", target)
        excluded = set(cfg["targets"][target].get("frozen_exclude") or [])
        space = set(spec.space.get("single_value_params") or {}) | \
            set(spec.space.get("multi_value_params") or {})
        assert excluded <= space, f"{target}: frozen_exclude names {excluded - space}"
        for pid in excluded:
            assert pid not in spec.frozen, f"{pid} is excused but still frozen"


@test("LAI and yield parameters live in ONE stage, so neither can undo the other")
def _():
    # The point of the restructure: a parameter that moves biomass and leaf area
    # (RUE, KDIF, partitioning) must be calibratable while both are being scored.
    growth = cc.load_spec(CONFIG, "winter_wheat", "growth")
    space = set(growth.space.get("single_value_params") or {}) | \
        set(growth.space.get("multi_value_params") or {})
    canopy = {"RGRLAI", "TDWI", "SLATableSLA", "LAICR", "DVSDLT"}
    yielding = {"FRTDM", "NMAXSO", "TCNT", "DVSNT", "DVSNLT"}
    shared = {"RUETableRUE", "KDIFTableK", "StorageOrgansPartitioningTableFraction"}
    for group, label in ((canopy, "canopy"), (yielding, "yield"), (shared, "shared")):
        missing = group - space
        assert not missing, f"{label} parameters missing from the joint space: {missing}"
    assert set(growth.views) == {"lai", "yield"}, growth.views
    assert not (space & set(growth.frozen))


@test("no parameter is both frozen and calibratable")
def _():
    for crop in CROPS:
        for target in TARGETS:
            spec = cc.load_spec(CONFIG, crop, target)
            names = set(spec.space.get("single_value_params") or {}) | \
                set(spec.space.get("multi_value_params") or {})
            overlap = names & set(spec.frozen)
            assert not overlap, f"{crop}/{target}: {overlap}"


# ---------------------------------------------------------------------------
print("\nproposal normalisation")


@test("index-keyed table edit changes only that element")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    current = cc.current_values(xml, spec.crop_name, spec.space)
    out = cc.normalise_proposal({"SLATableSLA": {"3": 0.0118}}, current, spec.space)
    assert len(out["SLATableSLA"]) == len(current["SLATableSLA"])
    assert out["SLATableSLA"][3] == 0.0118
    assert out["SLATableSLA"][:3] == current["SLATableSLA"][:3]
    changed = cc.changed_entries(out, current)
    assert changed["SLATableSLA"]["indices"] == [3]
    assert changed["SLATableSLA"]["n_elements_changed"] == 1


@test("unknown, frozen and wrong-shaped parameters are refused")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    current = cc.current_values(xml, spec.crop_name, spec.space)
    for raw in ({"NOT_A_PARAM": 1},
                {"TSUM1": 1300},                       # frozen -> not in the space
                {"RGRLAI": [0.02, 0.03]},              # scalar given a list
                {"SLATableSLA": 0.01},                 # table given a scalar
                {"SLATableSLA": [0.01, 0.01]}):        # wrong length
        try:
            cc.normalise_proposal(raw, current, spec.space)
        except SystemExit:
            continue
        raise AssertionError(f"accepted an invalid proposal: {raw}")


# ---------------------------------------------------------------------------
print("\nconstraints")


@test("a well-formed single-node change passes")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    violations, _ = proposal_check(spec, {"SLATableSLA": {"3": 0.0135}})
    assert not violations, [v.message for v in violations]


@test("out-of-bounds is caught")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"RGRLAI": 0.5})
    assert "bounds" in ids


@test("step-size limit is enforced")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"RGRLAI": 0.035})       # +85%
    assert "bounded_step_size" in ids
    violations, _ = proposal_check(spec, {"RGRLAI": 0.0227})  # +20%
    assert not violations, [v.message for v in violations]


@test("blast radius is limited to 3 parameters")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"RGRLAI": 0.0208, "TDWI": 23.0,
                                   "LAICR": 3.4, "RDRSHM": 0.022})
    assert "small_steps" in ids


@test("SLA profile must stay smooth and must not rise at maturity")
def _():
    # Values chosen to sit inside the bounds and inside the step limit, so it is
    # the shape rule under test and not one of the cheaper guards.
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"SLATableSLA": {"1": 0.0178}})   # 1.91x above node 0
    assert "sla_profile_smooth" in ids, ids
    _, ids = proposal_check(spec, {"SLATableSLA": {"7": 0.0135}})   # above the profile max
    assert "sla_declines_to_maturity" in ids, ids


@test("leaf death rate must not fall as temperature rises")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"RDRLeavesTableRelativeRate": {"3": 0.0070}})
    assert "leaf_rdr_non_decreasing_with_temp" in ids


@test("above-ground partitioning must stay closed at 1")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    _, ids = proposal_check(spec, {"LeavesPartitioningTableFraction": {"0": 0.72}})
    assert "above_ground_partitioning_sums_to_one" in ids
    # ...and passes when the stem fraction takes the difference.
    violations, _ = proposal_check(spec, {"LeavesPartitioningTableFraction": {"0": 0.72},
                                          "StemsPartitioningTableFraction": {"0": 0.28}})
    assert not violations, [v.message for v in violations]


@test("partitioning closure tolerates a crop whose baseline already differs")
def _():
    # spring_barley's three tables sit on different DVS grids, so the union-grid
    # sum is not 1 everywhere even before any change. That must not be reported.
    spec = cc.load_spec(CONFIG, "spring_barley", "growth")
    violations, ids = proposal_check(spec, {})
    assert "above_ground_partitioning_sums_to_one" not in ids, [v.message for v in violations]


@test("RUE must not rise after anthesis; storage organs must not fall back")
def _():
    # RUETableDVS is [0, 1, 1.3, 2]; the rule looks at the nodes from DVS 1.0 on.
    # Node 2 (DVS 1.3) raised above node 1 (DVS 1.0) is the violation, and it sits
    # inside both the bounds and the step limit.
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    require_params(spec, "RUETableRUE")
    _, ids = proposal_check(spec, {"RUETableRUE": {"2": 4.2}})     # 3.5 -> 4.2, above 3.8
    assert "rue_declines_after_anthesis" in ids, ids
    _, ids = proposal_check(spec, {"RUETableRUE": {"2": 3.2}})     # a legitimate decrease
    assert "rue_declines_after_anthesis" not in ids, ids

    spec = cc.load_spec(CONFIG, "winter_rapeseed", "growth")
    require_params(spec, "StorageOrgansPartitioningTableFraction")
    _, ids = proposal_check(spec, {"StorageOrgansPartitioningTableFraction": {"8": 0.4}})
    assert "storage_organ_non_decreasing" in ids, ids


@test("N threshold ordering is enforced")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    require_params(spec, "DVSNT", "DVSNLT")
    _, ids = proposal_check(spec, {"DVSNT": 1.15})   # DVSNLT is 1.3 -> ok
    assert "n_thresholds_ordered" not in ids
    _, ids = proposal_check(spec, {"DVSNLT": 1.05})  # DVSNT is 0.8 -> still ok
    assert "n_thresholds_ordered" not in ids
    _, ids = proposal_check(spec, {"DVSNT": 1.1, "DVSNLT": 1.05})
    assert "n_thresholds_ordered" in ids


# ---------------------------------------------------------------------------
print("\nfrozen phenology guard")


@test("guard passes a legitimate change and catches a phenology tamper")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    source = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "crop.xml"
        shutil.copyfile(source, xml)
        snapshot = cc.snapshot_frozen(xml, spec.crop_name, spec.frozen)
        assert len(snapshot["parameters"]) >= 12
        assert cc.verify_frozen(xml, snapshot) == []

        common.apply_parameters(xml, spec.crop_name, {"RGRLAI": 0.021})
        assert cc.verify_frozen(xml, snapshot) == [], "an LAI change must not trip the guard"

        common.apply_parameters(xml, spec.crop_name, {"TSUM1": 1300})
        drift = cc.verify_frozen(xml, snapshot)
        assert any("TSUM1" in d for d in drift), drift


@test("guard catches a moved response-table DVS grid")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    source = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "crop.xml"
        shutil.copyfile(source, xml)
        snapshot = cc.snapshot_frozen(xml, spec.crop_name, spec.frozen)
        grid = cc.read_param(cc.crop_root(xml), spec.crop_name, "SLATableDVS")
        moved = list(grid)
        moved[1] += 0.05
        common.apply_parameters(xml, spec.crop_name, {"SLATableDVS": moved})
        assert any("SLATableDVS" in d for d in cc.verify_frozen(xml, snapshot))


# ---------------------------------------------------------------------------
print("\nledger")


@test("ledger appends, never overwrites, and reports best + stopping")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    # Well above the configured target, or the objectives below would satisfy the
    # stage's target_objective and stop it for the wrong reason. The rules are a
    # tuning knob; the invariant under test is the bookkeeping.
    unit = 3.0 * float(spec.stopping["target_objective"])
    with tempfile.TemporaryDirectory() as tmp:
        original = cc.LEDGER_ROOT
        cc.LEDGER_ROOT = Path(tmp)
        try:
            assert cc.next_iteration(spec) == 0
            for i, objective in enumerate([1.0, 0.8, 0.9, 0.85]):
                cc.append_ledger(spec, {"iteration": i, "status": "completed",
                                        "objective": objective * unit,
                                        "parameters_changed": {"RGRLAI": 0.02}})
            records = cc.read_ledger(spec)
            assert [r["iteration"] for r in records] == [0, 1, 2, 3]
            assert cc.next_iteration(spec) == 4
            assert cc.best_record(spec)["iteration"] == 1

            stop = cc.stop_check(spec)
            assert stop["n_completed"] == 4
            assert stop["best_objective"] == 0.8 * unit
            assert stop["since_improvement"] == 2, stop
            assert stop["stop"] is False, stop

            # Flat iterations up to the configured patience must not stop it; one
            # past it must. Read the value rather than hard-coding it — the
            # stopping rules are a tuning knob and the invariant is the behaviour,
            # not the number.
            patience = int(spec.stopping["patience"])
            n = 4
            for _ in range(patience - stop["since_improvement"]):
                cc.append_ledger(spec, {"iteration": n, "status": "completed",
                                        "objective": 0.95 * unit, "parameters_changed": {}})
                n += 1
            stop = cc.stop_check(spec)
            assert stop["since_improvement"] == patience, stop
            assert stop["stop"] is True and "patience" in stop["reason"], stop
        finally:
            cc.LEDGER_ROOT = original


@test("a failed iteration keeps its slot and is excluded from best")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    with tempfile.TemporaryDirectory() as tmp:
        original = cc.LEDGER_ROOT
        cc.LEDGER_ROOT = Path(tmp)
        try:
            cc.append_ledger(spec, {"iteration": 0, "status": "completed", "objective": 1.0,
                                    "parameters_changed": {}})
            cc.append_ledger(spec, {"iteration": 1, "status": "failed", "objective": None,
                                    "error": "SIMPLACE produced no output"})
            assert cc.next_iteration(spec) == 2
            assert cc.best_record(spec)["iteration"] == 0
            assert cc.stop_check(spec)["n_completed"] == 1
        finally:
            cc.LEDGER_ROOT = original


# ---------------------------------------------------------------------------
print("\ndiagnostics")


def _synthetic_lai(peak_shift=0.0, level=1.0, n_seasons=6):
    """Sparse 8-day observations of a plausible LAI season, plus a perturbed sim."""
    rows = []
    for season in range(n_seasons):
        location, year = 100 + season, 2018
        dates = pd.date_range(f"{year}-03-01", f"{year}-08-01", freq="8D")
        dvs = np.linspace(0.1, 2.0, len(dates))
        obs = 4.0 * np.exp(-((dvs - 1.0) ** 2) / 0.35)
        sim = level * 4.0 * np.exp(-((dvs - 1.0 - peak_shift) ** 2) / 0.35)
        for d, v, o, s in zip(dates, dvs, obs, sim):
            rows.append({"location": location, "year": year, "date": d, "dvs": v,
                         "LAI": float(o), "LAI_sim": float(s)})
    return pd.DataFrame(rows)


@test("LAI diagnostics detect a pure level bias")
def _():
    pairs = _synthetic_lai(level=1.3)
    diag, seasons = cd.lai_diagnostics(pairs, [0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0])
    assert diag["overall"]["bias"] > 0.2, diag["overall"]
    assert diag["curve_shape"]["peak_lai"]["median_diff"] > 0.5
    assert abs(diag["curve_shape"]["peak_timing"]["median_diff"]) < 1e-6, "timing must be clean"
    assert len(seasons) == 6
    assert all(row["bias"] > 0 for row in diag["by_dvs_bin"])


@test("LAI diagnostics detect a phase shift with no level bias")
def _():
    diag, _ = cd.lai_diagnostics(_synthetic_lai(peak_shift=0.3),
                                 [0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0])
    assert diag["curve_shape"]["peak_timing"]["median_diff"] > 0, "late peak must show up"
    signs = {np.sign(row["bias"]) for row in diag["by_dvs_bin"] if row["n"] > 0}
    assert len(signs) > 1, "a phase error must flip the sign of the bias across bins"


@test("LAI figures render")
def _():
    pairs = _synthetic_lai(level=1.2)
    bins = [0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
    _, seasons = cd.lai_diagnostics(pairs, bins)
    with tempfile.TemporaryDirectory() as tmp:
        figures = cd.lai_plots(pairs, seasons, bins, Path(tmp), "test")
        assert len(figures) >= 3
        for path in figures:
            assert Path(path).stat().st_size > 5_000, path


@test("yield attribution separates a biomass error from a harvest-index error")
def _():
    years = list(range(2000, 2020))
    base = pd.DataFrame({"year": years * 2,
                         "NUTS_ID": ["DE111"] * 20 + ["DE222"] * 20,
                         "STATE_NAME": ["A"] * 20 + ["B"] * 20,
                         "yield": 6.0})
    reference = {"harvest_index": [0.38, 0.55], "ag_biomass_t_ha": [11.0, 20.0]}

    # Plausible biomass, implausible HI -> partitioning.
    hi_case = base.assign(AGBiomass_t_ha=15.0, Yield_t_ha=15.0 * 0.20)
    verdict = cd.yield_diagnostics(hi_case, reference)["attribution"]["verdict"]
    assert "harvest index" in verdict and "partitioning" in verdict, verdict

    # Plausible HI, implausible biomass -> RUE / KDIF.
    agb_case = base.assign(AGBiomass_t_ha=6.0, Yield_t_ha=6.0 * 0.45)
    verdict = cd.yield_diagnostics(agb_case, reference)["attribution"]["verdict"]
    assert "RUE" in verdict or "biomass parameters" in verdict, verdict


@test("yield diagnostics separate the temporal and spatial dimensions")
def _():
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2020)
    rows = []
    for state, offset in (("A", 0.0), ("B", 1.5)):     # a spatial gradient the model misses
        for year in years:
            observed = 6.0 + offset + 0.4 * rng.standard_normal()
            rows.append({"year": year, "NUTS_ID": f"DE{state}", "STATE_NAME": state,
                         "yield": observed, "Yield_t_ha": 6.75,
                         "AGBiomass_t_ha": 15.0, "maxLAI": 4.0})
    diag = cd.yield_diagnostics(pd.DataFrame(rows),
                                {"harvest_index": [0.38, 0.55], "ag_biomass_t_ha": [11.0, 20.0]})
    assert diag["spatial"]["RMSE_state_means"] > diag["temporal"]["RMSE_year_means"], diag
    assert diag["spatial"]["state_bias_range"][1] - diag["spatial"]["state_bias_range"][0] > 1.0


@test("history plot renders and annotates what changed")
def _():
    records = [{"iteration": i, "status": "completed", "objective": 1.0 - 0.05 * i,
                "parameters_changed": {"RGRLAI": 0.02}} for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        path = cd.history_plot(records, Path(tmp) / "history.png", "test")
        assert path and Path(path).stat().st_size > 5_000


# ---------------------------------------------------------------------------
print("\nisolation")


@test("every view of every stage gets its own isolated run dir")
def _():
    seen = {}
    for crop in CROPS:
        for target in TARGETS:
            spec = cc.load_spec(CONFIG, crop, target)
            for view, run in spec.runs.items():
                assert run.run_dir.name == view, run.run_dir
                assert run.run_dir.parent.name.startswith("calib_"), run.run_dir
                assert run.run_dir not in seen, \
                    f"{crop}/{target}/{view} collides with {seen.get(run.run_dir)}"
                seen[run.run_dir] = f"{crop}/{target}/{view}"
            # Two views of one stage must not share an output namespace either —
            # they write per-location files into out/<exp_name>/ and would clobber
            # each other's results.
            names = [run.exp_name for run in spec.runs.values()]
            assert len(set(names)) == len(names), names


@test("the views of a stage are driven from one crop.xml")
def _():
    # If the yield view kept its own parameter file, the two components of the
    # objective would describe different crops and the joint calibration would be
    # meaningless.
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    assert len(spec.views) == 2
    assert spec.crop_xml == spec.runs[spec.views[0]].crop_xml
    assert spec.runs["lai"].crop_xml != spec.runs["yield"].crop_xml, \
        "each run dir has its own file on disk; sync_crop_xml is what keeps them equal"

    source = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for view in spec.views:
            xml = Path(tmp) / view / "data" / "crop" / "crop.xml"
            xml.parent.mkdir(parents=True)
            shutil.copyfile(source, xml)
            runs[view] = dataclasses.replace(spec.runs[view], run_dir=Path(tmp) / view)
        spec.runs = runs
        common.apply_parameters(spec.crop_xml, spec.crop_name, {"RGRLAI": 0.0211})
        mirrored = cc.sync_crop_xml(spec)
        assert [str(p) for p in mirrored] == [str(spec.runs["yield"].crop_xml)]
        for view in spec.views:
            value = cc.read_param(cc.crop_root(spec.runs[view].crop_xml),
                                  spec.crop_name, "RGRLAI")
            assert value == 0.0211, f"{view} did not receive the proposal: {value}"


@test("nothing in the workflow writes to the production crop.xml")
def _():
    for target in TARGETS:
        spec = cc.load_spec(CONFIG, "winter_wheat", target)
        production = spec.run.crop_dir / "data" / "crop" / "crop.xml"
        for run in spec.runs.values():
            assert run.crop_xml != production
            assert "runs_optim" in str(run.crop_xml)


# ---------------------------------------------------------------------------
print("\nharvest index, translocation, disabling")


@test("every declared parameter carries an explicit enabled switch")
def _():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    missing = [f"{t}.{name}"
               for t, tcfg in cfg["targets"].items()
               for name, pc in (tcfg.get("parameters") or {}).items()
               if "enabled" not in pc]
    assert not missing, f"no enabled switch on: {missing}"
    # A switch that is off must say why — otherwise the next reader cannot tell
    # a deliberate exclusion from an accident.
    for t, tcfg in cfg["targets"].items():
        for name, pc in (tcfg.get("parameters") or {}).items():
            if pc["enabled"] is False:
                assert pc.get("disabled_reason"), f"{t}.{name} is off with no reason given"


@test("toggling the switch adds and removes the parameter from the space")
def _():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load((OPTIM_DIR / "config.yaml").read_text())

    def space_for(target, tweak):
        c = yaml.safe_load(CONFIG.read_text())
        tweak(c["targets"][target]["parameters"])
        out = Path(tempfile.mkdtemp())
        (out / "calibration.yaml").write_text(yaml.safe_dump(c))
        (out / "config.yaml").write_text(yaml.safe_dump(base))
        spec = cc.load_spec(out / "calibration.yaml", "winter_wheat", target)
        names = set(spec.space.get("single_value_params") or {}) | \
            set(spec.space.get("multi_value_params") or {})
        shutil.rmtree(out, ignore_errors=True)
        return names

    # Off: gone from the space. On: back, with no other parameter disturbed.
    on = space_for("growth", lambda p: None)
    off = space_for("growth", lambda p: p["RGRLAI"].__setitem__("enabled", False))
    assert "RGRLAI" in on and "RGRLAI" not in off
    assert on - {"RGRLAI"} == off, "disabling one parameter changed the others"

    # And the reverse, on a parameter that ships disabled.
    on_y = space_for("growth", lambda p: p["YieldAdjustRatio"].__setitem__("enabled", True))
    assert "YieldAdjustRatio" not in on and "YieldAdjustRatio" in on_y


@test("a disabled parameter leaves the space and cannot be proposed")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    names = set(spec.space.get("single_value_params") or {}) | \
        set(spec.space.get("multi_value_params") or {})
    for pid in ("YieldAdjustRatio", "FreshratioStorageOrgan"):
        assert pid not in names, f"{pid} is disabled and must not be in the space"
        row = next(r for r in cc.describe_space(spec) if r["parameter"] == pid)
        assert row["enabled"] is False and row["movable"] is False
        assert row["note"], "a disabled parameter must carry the reason it was disabled"
    # Refused at parse time, before any constraint is even consulted.
    xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    current = cc.current_values(xml, spec.crop_name, spec.space)
    try:
        cc.normalise_proposal({"YieldAdjustRatio": 1.1}, current, spec.space)
    except SystemExit as exc:
        assert "not a calibratable parameter" in str(exc)
        return
    raise AssertionError("a disabled parameter must be refused")


@test("FRTDM is calibratable for every crop, on bounds that fit all of them")
def _():
    values = {}
    for crop in CROPS:
        spec = cc.load_spec(CONFIG, crop, "growth")
        assert "FRTDM" in (spec.space.get("single_value_params") or {}), crop
        row = next(r for r in cc.describe_space(spec) if r["parameter"] == "FRTDM")
        assert row["movable"], f"{crop}: FRTDM resolved immovable"
        low, high = row["bounds"]
        assert low <= row["value"] <= high, f"{crop}: {row['value']} outside {row['bounds']}"
        values[crop] = row["value"]
    # The spread is the reason the bounds are absolute rather than relative.
    assert min(values.values()) < 0.05 < max(values.values()), values


@test("yield metrics report harvest index and the translocation term")
def _():
    import objectives
    rng = np.random.default_rng(5)
    n = 240
    biomass = rng.normal(16.0, 1.5, n)
    df = pd.DataFrame({
        "NUTS_ID": [f"DE{i % 30:02d}" for i in range(n)],
        "STATE_NAME": ["Bayern", "Hessen"] * (n // 2),
        "year": rng.integers(2000, 2020, n),
        "ag_biomass_sim": biomass,
        "yield_sim": biomass * 0.37,            # simulated HI = 0.37
        "yield": biomass * 0.45,                # observations imply HI = 0.45
    })
    df["yield_translocated_sim"] = df["yield_sim"] + 0.8   # FRTDM adds 0.8 t/ha

    loss, metrics = objectives.yield_loss_fn(df)
    assert abs(metrics["harvest index (sim)"] - 0.37) < 0.01, metrics
    assert abs(metrics["harvest index (required)"] - 0.45) < 0.01, metrics
    assert metrics["HI bias"] < 0, "a low simulated HI must report a negative bias"
    # Metrics are rounded to 4 dp for a readable ledger, so compare at that scale.
    assert abs(metrics["mean translocated yield (t/ha)"]
               - float(df["yield_translocated_sim"].mean())) < 1e-4
    assert metrics["translocated / yield ratio"] > 1.0, "FRTDM adds to the storage organ"

    # The objective itself must not move when the extra columns are present —
    # otherwise every objective already in the ledger becomes incomparable.
    bare = df.drop(columns=["ag_biomass_sim", "yield_translocated_sim"])
    assert abs(objectives.yield_loss_fn(bare)[0] - loss) < 1e-12
    assert "harvest index (sim)" not in objectives.yield_loss_fn(bare)[1]


@test("the translocation diagnostic says whether FRTDM helps or overshoots")
def _():
    rng = np.random.default_rng(6)
    n = 200
    observed = rng.normal(7.0, 0.6, n)
    base = dict(NUTS_ID=[f"DE{i % 25:02d}" for i in range(n)],
                STATE_NAME=["Bayern"] * n, year=rng.integers(2000, 2020, n))

    # (a) plain yield is 0.8 t/ha short; translocation closes the gap -> helps
    helps = pd.DataFrame({**base, "yield": observed,
                          "yield_sim": observed - 0.8,
                          "Yield_translocated_t_ha": observed - 0.05})
    d = cd._translocation_diagnostics(helps)
    assert d["rmse_delta_translocated_minus_plain"] < 0, d
    assert "worth calibrating" in d["verdict"], d["verdict"]

    # (b) plain yield is right; translocation overshoots -> lower FRTDM
    over = pd.DataFrame({**base, "yield": observed,
                         "yield_sim": observed - 0.02,
                         "Yield_translocated_t_ha": observed + 1.2})
    d = cd._translocation_diagnostics(over)
    assert d["rmse_delta_translocated_minus_plain"] > 0, d
    assert "lower FRTDM" in d["verdict"], d["verdict"]

    # (c) no translocation column at all -> reported as absent, not as an error
    assert cd._translocation_diagnostics(
        pd.DataFrame({**base, "yield": observed, "yield_sim": observed})) is None


# ---------------------------------------------------------------------------
print("\nstage 1: phenology")


@test("phenology is a single-view stage over the thermal-time set")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "phenology")
    assert spec.views == ["phenology"], spec.views
    names = set(spec.space.get("single_value_params") or {}) | \
        set(spec.space.get("multi_value_params") or {})
    for pid in ("TSUM1", "TSUM2", "TSUMEM", "TBASEM", "TEFFMX"):
        assert pid in names, f"{pid} must be calibratable in stage 1"
    # Calibrated from scratch: the objective is the raw RMSE in days, so the
    # stopping target stays readable as a number of days.
    assert spec.components == {"phenology": {"weight": 1.0, "scale": 1.0}}, spec.components
    assert spec.objective_name == "phenology_rmse_days"


@test("the thermal-time parameters are calibrated from scratch, not anchored on crop.xml")
def _():
    # Relative bounds around the shipped values would make stage 1 a refinement of
    # whatever is already in the file. The bounds that matter here are absolute and
    # physiological, so the same declaration works for a crop that has never been
    # calibrated.
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    params = cfg["targets"]["phenology"]["parameters"]
    for pid in ("TSUM1", "TSUM2", "TSUMEM", "TBASEM", "TEFFMX"):
        assert params[pid]["mode"] == "absolute", f"{pid} is anchored on the current value"

    widths = {}
    for crop in CROPS:
        spec = cc.load_spec(CONFIG, crop, "phenology")
        row = next(r for r in cc.describe_space(spec) if r["parameter"] == "TSUM1")
        low, high = row["bounds"]
        assert low <= row["value"] <= high, f"{crop}: {row['value']} outside {row['bounds']}"
        widths[crop] = (low, high)
    # Absolute bounds mean every crop searches the same physiological range.
    assert len(set(widths.values())) == 1, widths


@test("phenology constraints catch an inverted temperature pair")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "phenology")
    xml = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    current = cc.current_values(xml, spec.crop_name, spec.space)
    # TBASEM must stay below TEFFMX. Push the base above the ceiling.
    ceiling = float(current["TEFFMX"])
    _, ids = proposal_check(spec, {"TBASEM": round(ceiling + 1.0, 2)}, xml)
    assert "emergence_base_below_ceiling" in ids, ids
    # A small, ordered move must pass.
    violations, ids = proposal_check(
        spec, {"TSUM1": round(float(current["TSUM1"]) * 1.05)}, xml)
    assert not violations, [v.message for v in violations]


@test("phenology attribution splits flowering from duration")
def _():
    rng = np.random.default_rng(11)
    n = 400
    obs_flowering = rng.normal(150, 9, n).round()
    duration = rng.normal(55, 5, n).round()
    base = pd.DataFrame({
        "PointID": rng.integers(1, 60, n),
        "harvest_year": rng.integers(2000, 2020, n),
        "flowering_doy": obs_flowering,
        "maturity_doy": obs_flowering + duration,
    })

    # (a) flowering 6 d late, duration correct -> TSUM1 only
    a = base.copy()
    a["flowering_doy_sim"] = a["flowering_doy"] + 6
    a["maturity_doy_sim"] = a["maturity_doy"] + 6
    verdict = cd.phenology_diagnostics(a)["attribution"]["verdict"]
    assert "TSUM1" in verdict, verdict
    assert "TSUM2" not in verdict, f"a pure flowering shift must not implicate TSUM2: {verdict}"

    # (b) flowering correct, duration 7 d short -> TSUM2 only
    b = base.copy()
    b["flowering_doy_sim"] = b["flowering_doy"]
    b["maturity_doy_sim"] = b["maturity_doy"] - 7
    verdict = cd.phenology_diagnostics(b)["attribution"]["verdict"]
    assert "TSUM2" in verdict, verdict
    assert "TSUM1" not in verdict, f"a pure duration error must not implicate TSUM1: {verdict}"

    # (c) both correct -> no parameter change indicated
    c = base.copy()
    c["flowering_doy_sim"] = c["flowering_doy"]
    c["maturity_doy_sim"] = c["maturity_doy"]
    verdict = cd.phenology_diagnostics(c)["attribution"]["verdict"]
    assert "no parameter change" in verdict, verdict


@test("phenology figures render")
def _():
    rng = np.random.default_rng(3)
    n = 200
    flowering = rng.normal(150, 8, n).round()
    df = pd.DataFrame({
        "PointID": rng.integers(1, 40, n),
        "harvest_year": rng.integers(2000, 2020, n),
        "flowering_doy": flowering,
        "maturity_doy": flowering + 55,
        "flowering_doy_sim": flowering + rng.normal(3, 2, n),
        "maturity_doy_sim": flowering + 55 + rng.normal(3, 2, n),
    })
    out = Path(tempfile.mkdtemp())
    try:
        figures = cd.phenology_plots(df, out, "test")
        assert len(figures) == 2
        assert all(Path(f).stat().st_size > 5000 for f in figures)
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ---------------------------------------------------------------------------
print("\nlocal LLM client")


@test("a model reply is recovered from prose, fences and raw JSON alike")
def _():
    from agents import llm
    wanted = {"parameter_changes": {"RGRLAI": 0.02}}
    for text in (
        json.dumps(wanted),
        "```json\n" + json.dumps(wanted) + "\n```",
        "Here is my decision:\n\n" + json.dumps(wanted) + "\n\nHope that helps.",
        '{"note": "a } brace inside a string", "parameter_changes": {"RGRLAI": 0.02}}',
    ):
        assert llm.extract_json(text)["parameter_changes"]["RGRLAI"] == 0.02, text[:60]
    for bad in ("", "no json here at all", "{unclosed: "):
        try:
            llm.extract_json(bad)
        except llm.LLMError:
            continue
        raise AssertionError(f"should have refused: {bad!r}")


@test("OLLAMA_HOST is normalised whether or not it carries a scheme")
def _():
    from agents import llm
    assert llm.normalise_host("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert llm.normalise_host("http://gpu01:11434/") == "http://gpu01:11434"
    assert llm.normalise_host("") == llm.DEFAULT_HOST


@test("the Ollama client speaks the real HTTP protocol")
def _():
    """Runs OllamaBackend against a stand-in server that implements /api/tags and
    /api/chat exactly as Ollama does. This is what verifies the wire format on a
    machine with no Ollama installed."""
    import http.server
    import threading
    from agents import llm

    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):  # keep the test output clean
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send({"models": [{"name": "qwen2.5:14b-instruct"},
                                       {"name": "llama3.1:8b"}]})
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.update(json.loads(self.rfile.read(length)))
            self._send({"message": {"role": "assistant",
                                    "content": '{"parameter_changes": {"RGRLAI": 0.021}}'}})

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = f"http://127.0.0.1:{server.server_address[1]}"
        backend = llm.OllamaBackend(model="qwen2.5:14b-instruct", host=host,
                                    temperature=0.15, num_ctx=4096, timeout=20)
        assert backend.models() == ["llama3.1:8b", "qwen2.5:14b-instruct"]
        ok, detail = backend.available()
        assert ok, detail

        reply = backend.json_chat([{"role": "user", "content": "decide"}])
        assert reply["parameter_changes"]["RGRLAI"] == 0.021

        # The request must be what Ollama expects, not just something it tolerates.
        assert received["model"] == "qwen2.5:14b-instruct"
        assert received["stream"] is False
        assert received["format"] == "json"
        assert received["options"]["temperature"] == 0.15
        assert received["options"]["num_ctx"] == 4096

        # A model that is not pulled must be reported, not silently substituted.
        absent = llm.OllamaBackend(model="mistral:7b", host=host, timeout=20)
        ok, detail = absent.available()
        assert not ok and "not pulled" in detail, detail
    finally:
        server.shutdown()
        server.server_close()


@test("reasoning models: thinking is disabled and its failures are named")
def _():
    """qwen3.6 and friends return a separate `thinking` field and will spend the
    whole generation budget on it, leaving `content` empty or truncated. The two
    resulting errors used to surface as 'no JSON object in the reply', which names
    neither the cause nor the fix."""
    import http.server
    import threading
    from agents import llm

    sent, scenario = {}, {"mode": "ok"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            sent.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            mode = scenario["mode"]
            if mode == "ok":
                self._send({"message": {"content": '{"parameter_changes": {"RGRLAI": 0.02}}'},
                            "done_reason": "stop"})
            elif mode == "empty_thinking":
                self._send({"message": {"content": "", "thinking": "Let me consider " * 40},
                            "done_reason": "stop"})
            elif mode == "json_in_thinking":
                self._send({"message": {"content": "",
                                        "thinking": 'so: {"parameter_changes": {"TDWI": 19.0}}'},
                            "done_reason": "stop"})
            elif mode == "truncated":
                self._send({"message": {"content": '{"analysis": "the diagnostics sh'},
                            "done_reason": "length", "eval_count": 2048})

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host = f"http://127.0.0.1:{server.server_address[1]}"
        backend = llm.OllamaBackend(model="qwen3.6:35b-a3b", host=host, timeout=20)

        # Thinking must be switched off in the request, and a budget reserved.
        assert backend.json_chat([{"role": "user", "content": "x"}])["parameter_changes"]
        assert sent["think"] is False, sent
        assert sent["options"]["num_predict"] > 0
        assert sent["options"]["num_ctx"] >= 32768

        # All thinking, no answer -> the error must name think and num_ctx.
        scenario["mode"] = "empty_thinking"
        try:
            backend.json_chat([{"role": "user", "content": "x"}])
            raise AssertionError("should have refused")
        except llm.LLMError as exc:
            assert "think" in str(exc) and "num_ctx" in str(exc), exc

        # If the JSON is in the thinking text after all, use it rather than fail.
        scenario["mode"] = "json_in_thinking"
        assert backend.json_chat([{"role": "user", "content": "x"}]) \
            ["parameter_changes"]["TDWI"] == 19.0

        # Cut off mid-JSON -> say so, instead of "no JSON object".
        scenario["mode"] = "truncated"
        try:
            backend.json_chat([{"role": "user", "content": "x"}])
            raise AssertionError("should have refused")
        except llm.LLMError as exc:
            assert "cut off" in str(exc) and "num_ctx" in str(exc), exc
    finally:
        server.shutdown()
        server.server_close()


@test("the history section stays bounded as the ledger grows")
def _():
    from agents.base import render_history
    records = [{"iteration": i, "objective": 1.5 - i * 0.01, "status": "completed",
                "improved": i % 3 == 0, "parameters_changed": {"RUETableRUE": [1, 2]},
                "reason": "a reason long enough to matter " * 4,
                "expected_effect": "an expectation " * 6}
               for i in range(80)]
    full = render_history(records, max_full=12)
    # The recent window is verbatim; everything older is one line each.
    assert "Most recent 12 iterations in full" in full
    assert "iter  79" not in full.split("Most recent")[0], "the tail must be in full"
    assert "iter   0" in full, "older iterations must still be listed, briefly"
    # Bounded: 80 iterations must not cost much more than 20 do.
    small = render_history(records[:20], max_full=12)
    assert len(full) < len(small) * 2.0, (len(small), len(full))


@test("an unreachable server is an error, never a silent fallback")
def _():
    from agents import llm
    # Port 1 is never an Ollama server.
    backend = llm.OllamaBackend(model="x", host="http://127.0.0.1:1", timeout=2)
    ok, detail = backend.available()
    assert not ok and "cannot reach" in detail, detail
    try:
        backend.json_chat([{"role": "user", "content": "hi"}])
    except llm.LLMUnavailable:
        return
    raise AssertionError("an unreachable server must raise LLMUnavailable")


# ---------------------------------------------------------------------------
print("\nagent loop")


def _stub_agent(replies, *, valid=True, violations=()):
    """A growth agent whose calibrate.py calls are replaced by canned answers.

    Exercises the agent's own logic — context building, reply parsing, the repair
    round-trip, what reaches the ledger — without a model server or a cluster.
    """
    import agents as ag

    spec = cc.load_spec(CONFIG, "winter_wheat", "growth")
    rows = cc.describe_space(spec)
    status = {
        "crop": "winter_wheat", "target": "growth", "objective": "joint_lai_yield",
        "views": spec.views, "objective_components": spec.components,
        "scope": {view: {"rows": 400, "locations": 400, "subset": {"n_locations": 400}}
                  for view in spec.views},
        "n_completed": 1, "next_iteration": 1,
        "frozen": {"parameters": spec.frozen, "intact": True, "drift": []},
        "best": {"iteration": 0, "objective": 2.71},
        "stopping": {"stop": False, "reason": "criteria not met", "n_completed": 1},
        "parameters": rows, "constraints": spec.constraints,
        "ledger_dir": "/tmp/ledger",
    }
    history = [{
        "iteration": 0, "objective": 2.71, "status": "completed", "improved": True,
        "parameters_changed": {}, "reason": "baseline",
        "diagnostics": {
            "objective": {"lai": {"loss": 0.56, "scaled": 3.76},
                          "yield": {"loss": 0.84, "scaled": 1.67}},
            "lai": {"by_dvs_bin": [{"dvs_bin": "1.00-1.25 anthesis", "bias": -0.6}]},
            "yield": {"attribution": {"verdict": "biomass is plausible"}}},
    }]

    agent = ag.GrowthAgent("winter_wheat", ag.MockBackend(replies=list(replies)),
                           config_path=CONFIG, verbose=False)
    executed = {}
    agent.status = lambda: status
    agent.history = lambda: history
    agent.preflight = lambda params: {"valid": valid, "violations": list(violations)}
    agent.execute = lambda decision: executed.update(decision=decision) or {"status": "ok"}
    return agent, executed


@test("the context carries values, bounds, meaning, history and the schema")
def _():
    agent, _ = _stub_agent([])
    context = agent.context(agent.status(), agent.history())
    assert "RGRLAI" in context and "SLATableSLA" in context
    assert "bounds" in context
    assert "Maximum relative growth rate" in context, "the biological meaning must be present"
    assert "iteration 0: objective 2.7100" in context
    assert "parameter_changes" in context, "the reply schema must be stated"
    assert "TSUM1" in context, "the frozen list must be visible so the model does not try"


@test("the context of a joint stage names both components and both diagnostics")
def _():
    agent, _ = _stub_agent([])
    context = agent.context(agent.status(), agent.history())
    for view in ("lai", "yield"):
        assert f"view {view}" in context, f"the {view} component must be described"
    assert "weight 0.5" in context, "the model must see how the components are weighted"
    assert "by_dvs_bin" in context and "attribution" in context, \
        "both diagnostic blocks must reach the model"


@test("a well-formed reply becomes a Decision and reaches execute")
def _():
    reply = json.dumps({
        "analysis": "peak LAI is 0.6 low in the anthesis bin",
        "hypothesis": "specific leaf area is too low around anthesis",
        "parameter_changes": {"SLATableSLA": {"3": 0.0135}},
        "reason": "bias is -0.6 in the anthesis bin and near zero before it",
        "reasoning": "ruled out RGRLAI because the early bins are unbiased",
        "expected_effect": "the anthesis-bin bias should move toward zero",
        "confidence": 0.7, "stop": False,
    })
    agent, executed = _stub_agent([reply])
    agent.iterate()
    decision = executed["decision"]
    assert decision.parameters == {"SLATableSLA": {"3": 0.0135}}
    assert "anthesis bin" in decision.reason
    blob = agent._reasoning_blob(decision)
    assert "ruled out RGRLAI" in blob and "growth agent" in blob


@test("a rejected proposal is handed back with the violation and repaired")
def _():
    bad = json.dumps({"parameter_changes": {"SLATableSLA": {"3": 0.9}},
                      "reason": "way too high"})
    good = json.dumps({"parameter_changes": {"SLATableSLA": {"3": 0.0135}},
                       "reason": "within bounds this time"})

    import agents as ag
    agent, executed = _stub_agent([bad, good])
    calls = {"n": 0}

    def preflight(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"valid": False, "violations": [
                {"constraint": "bounds", "message": "SLATableSLA[3]=0.9 outside [0.006, 0.025]"}]}
        return {"valid": True, "violations": []}

    agent.preflight = preflight
    agent.iterate()
    assert calls["n"] == 2, "the repaired proposal must be pre-flighted again"
    assert executed["decision"].parameters == {"SLATableSLA": {"3": 0.0135}}
    # The rejection text has to reach the model, or the repair is guesswork.
    last = agent.backend.calls[-1][-1]["content"]
    assert "outside [0.006, 0.025]" in last

    # And a proposal that never becomes valid must fail loudly rather than run.
    agent, _ = _stub_agent([bad] * 6, valid=False, violations=[
        {"constraint": "bounds", "message": "still out of bounds"}])
    try:
        agent.iterate()
    except ag.AgentError as exc:
        assert "repair attempt" in str(exc)
        return
    raise AssertionError("an unrepairable proposal must not reach the model run")


@test("a hoisted table index is re-nested under the named parameter")
def _():
    from agents.base import Decision
    # What a local model actually produced: the index map at the top level, with
    # the parameter name left behind in `parameter`.
    d = Decision.from_reply({"parameter": "RUETableRUE",
                             "parameter_changes": {"2": 3.1}, "reason": "x"})
    assert d.parameters == {"RUETableRUE": {"2": 3.1}}, d.parameters
    # The other two shorthands.
    assert Decision.from_reply(
        {"parameter": "RGRLAI", "value": 0.021}).parameters == {"RGRLAI": 0.021}
    assert Decision.from_reply(
        {"parameter": "RUETableRUE", "index": 2, "new_value": 3.1}
    ).parameters == {"RUETableRUE": {"2": 3.1}}
    # A well-formed proposal must be left exactly as it is.
    good = {"parameter_changes": {"RUETableRUE": {"2": 3.1}}, "parameter": "RUETableRUE"}
    assert Decision.from_reply(good).parameters == {"RUETableRUE": {"2": 3.1}}
    # And a multi-parameter change must not be mangled by the re-nesting rule.
    multi = {"parameter": "RGRLAI", "parameter_changes": {"RGRLAI": 0.02, "TDWI": 18.0}}
    assert Decision.from_reply(multi).parameters == {"RGRLAI": 0.02, "TDWI": 18.0}


@test("an unparseable proposal becomes a repairable verdict, not a crash")
def _():
    import agents as ag
    # A real agent, so the real preflight() runs; only the subprocess is faked.
    agent = ag.GrowthAgent("winter_wheat", ag.MockBackend(replies=[]),
                        config_path=CONFIG, verbose=False)
    agent._calibrate = lambda *a, **k: type(
        "P", (), {"returncode": 1, "stdout": "",
                  "stderr": "'2' is not a calibratable parameter for this target."})()
    verdict = agent.preflight({"2": 3.1})
    assert verdict["valid"] is False
    assert "not a calibratable parameter" in verdict["violations"][0]["message"]

    # A well-formed verdict on stdout must still be parsed normally.
    agent._calibrate = lambda *a, **k: type(
        "P", (), {"returncode": 0, "stderr": "",
                  "stdout": 'note: staging\n{"valid": true, "violations": []}'})()
    assert agent.preflight({"RGRLAI": 0.02})["valid"] is True


@test("an unusable reply ends the session without losing the ledger")
def _():
    import agents as ag
    bad = json.dumps({"parameter_changes": {"nonsense": 1.0}})
    agent, executed = _stub_agent([bad] * 20, valid=False, violations=[
        {"constraint": "bounds", "message": "still wrong"}])
    summary = agent.loop(max_iterations=3)
    assert "unusable model reply" in summary["stopped_because"], summary
    assert summary["iterations_this_session"] == 0
    assert not executed, "nothing may run once the proposal cannot be repaired"


@test("iteration 0 is a baseline that changes nothing, with no model call")
def _():
    agent, executed = _stub_agent([])
    status = agent.status()
    status["next_iteration"] = 0
    agent.iterate()
    assert executed["decision"].parameters == {}
    assert agent.backend.calls == [], "the baseline must not consult the model"


@test("drifted frozen parameters stop the loop before anything runs")
def _():
    import agents as ag
    agent, executed = _stub_agent(['{"parameter_changes": {"RGRLAI": 0.02}}'])
    agent.status()["frozen"].update(intact=False, drift=["TSUM1: 1280.0 -> 1300.0"])
    try:
        agent.iterate()
    except ag.AgentError as exc:
        assert "TSUM1" in str(exc)
        assert not executed, "nothing may run once the freeze is broken"
        return
    raise AssertionError("a frozen-parameter drift must abort the iteration")


@test("the agent can stop itself, and says why")
def _():
    agent, executed = _stub_agent([json.dumps({
        "parameter_changes": {}, "stop": True,
        "reason": "the residual is inside the observation scatter"})])
    result = agent.iterate()
    assert result["stopped_by_agent"] is True
    assert "observation scatter" in result["reason"]
    assert not executed, "stopping must not run the model"


@test("there is one agent per stage, and it targets that stage")
def _():
    import agents as ag
    assert sorted(ag.AGENTS) == sorted(TARGETS)
    for target, cls in ag.AGENTS.items():
        agent = cls("winter_wheat", ag.MockBackend(replies=[]),
                    config_path=CONFIG, verbose=False)
        assert agent.target == target
        # The scope handed to calibrate.py must name the stage, so an agent can
        # never drive the other one's ledger.
        assert "--target" in agent._scope() and target in agent._scope()


@test("every agent has a prompt, and the prompts state their hard rules")
def _():
    import agents as ag
    from agents.base import PROMPT_DIR
    required = {
        "growth.md": ["Peak timing", "sum to one", "expected_effect", "FRTDM",
                      "both components"],
        "phenology.md": ["from scratch", "duration", "TSUM1"],
        "analyst.md": ["do not propose", "thrashing"],
    }
    for name, phrases in required.items():
        raw = (PROMPT_DIR / name).read_text()
        assert len(raw) > 800, f"{name} is too short to be a real prompt"
        # Normalise whitespace: the prompts are hard-wrapped, so a phrase can
        # straddle a line break without being any less present.
        text = " ".join(raw.split()).lower()
        for phrase in phrases:
            assert phrase.lower() in text, f"{name} is missing: {phrase!r}"
    for target, cls in ag.AGENTS.items():
        assert (PROMPT_DIR / cls.prompt_file).exists(), f"{target} prompt missing"


@test("long diagnostics are trimmed but say how much was dropped")
def _():
    from agents.base import compact
    trimmed = compact({"by_year": [{"year": y} for y in range(2000, 2060)]}, max_items=10)
    rows = trimmed["by_year"]
    assert len(rows) == 11
    assert "50 more entries omitted" in rows[-1]
    # Short lists must survive untouched.
    assert compact({"a": [1, 2, 3]}, max_items=10) == {"a": [1, 2, 3]}


# ---------------------------------------------------------------------------
def main() -> int:
    failed = [(n, tb) for n, status, tb in _results if status == "fail"]
    skipped = [(n, why) for n, status, why in _results if status == "skip"]
    passed = sum(1 for _, status, _ in _results if status == "pass")
    print(f"\n{'=' * 70}")
    print(f"  {passed}/{len(_results) - len(skipped)} passed"
          + (f", {len(skipped)} skipped (not in this configuration)" if skipped else ""))
    print("=" * 70)
    for name, why in skipped:
        print(f"  skipped: {name} — {why}")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
