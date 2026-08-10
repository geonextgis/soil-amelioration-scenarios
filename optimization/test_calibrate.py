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

_results: list[tuple[str, bool, str]] = []


def test(name):
    def wrap(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            _results.append((name, False, traceback.format_exc()))
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
    assert sorted(cfg["targets"]) == ["lai", "phenology", "yield"]
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


@test("the three losses survived the removal and are importable")
def _():
    import objectives
    assert sorted(objectives.TARGETS) == ["lai", "phenology", "yield"]
    for target in objectives.TARGETS:
        process_result, loss_fn = cc.objective_for(target)
        assert callable(process_result) and callable(loss_fn)
    # The DVS bins the LAI diagnostics key off must still be the ones the loss uses.
    assert objectives.DVS_BINS == [0.0, 0.25, 0.5, 1.0, 1.25, 1.50, 1.75, 2.0]


@test("space resolves for every crop and target")
def _():
    for crop in CROPS:
        for target in ("phenology", "lai", "yield"):
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
        spec = cc.load_spec(CONFIG, crop, "lai")
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
        for target in ("phenology", "lai", "yield"):
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
    assert "provenance.json" in source, "the yield handoff must be exempt from the reset"
    assert "rebuild" in source


@test("frozen set covers phenology; yield additionally freezes the LAI set")
def _():
    lai = cc.load_spec(CONFIG, "winter_wheat", "lai")
    yld = cc.load_spec(CONFIG, "winter_wheat", "yield")
    for pid in ("TSUM1", "TSUM2", "TEFFMX", "SLATableDVS", "RUETableDVS"):
        assert pid in lai.frozen and pid in yld.frozen, f"{pid} not frozen"
    for pid in ("RGRLAI", "SLATableSLA", "LAICR", "DVSDLT"):
        assert pid in yld.frozen, f"{pid} must be frozen during yield calibration"
        assert pid not in lai.frozen, f"{pid} must be calibratable during LAI calibration"
    # ...except the structural counterweight, which yield needs.
    assert "StemsPartitioningTableFraction" not in yld.frozen


@test("no parameter is both frozen and calibratable")
def _():
    for crop in CROPS:
        for target in ("phenology", "lai", "yield"):
            spec = cc.load_spec(CONFIG, crop, target)
            names = set(spec.space.get("single_value_params") or {}) | \
                set(spec.space.get("multi_value_params") or {})
            overlap = names & set(spec.frozen)
            assert not overlap, f"{crop}/{target}: {overlap}"


# ---------------------------------------------------------------------------
print("\nproposal normalisation")


@test("index-keyed table edit changes only that element")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    violations, _ = proposal_check(spec, {"SLATableSLA": {"3": 0.0135}})
    assert not violations, [v.message for v in violations]


@test("out-of-bounds is caught")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    _, ids = proposal_check(spec, {"RGRLAI": 0.5})
    assert "bounds" in ids


@test("step-size limit is enforced")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    _, ids = proposal_check(spec, {"RGRLAI": 0.035})       # +85%
    assert "bounded_step_size" in ids
    violations, _ = proposal_check(spec, {"RGRLAI": 0.0227})  # +20%
    assert not violations, [v.message for v in violations]


@test("blast radius is limited to 3 parameters")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    _, ids = proposal_check(spec, {"RGRLAI": 0.0208, "TDWI": 23.0,
                                   "LAICR": 3.4, "RDRSHM": 0.022})
    assert "small_steps" in ids


@test("SLA profile must stay smooth and must not rise at maturity")
def _():
    # Values chosen to sit inside the bounds and inside the step limit, so it is
    # the shape rule under test and not one of the cheaper guards.
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    _, ids = proposal_check(spec, {"SLATableSLA": {"1": 0.0178}})   # 1.91x above node 0
    assert "sla_profile_smooth" in ids, ids
    _, ids = proposal_check(spec, {"SLATableSLA": {"7": 0.0135}})   # above the profile max
    assert "sla_declines_to_maturity" in ids, ids


@test("leaf death rate must not fall as temperature rises")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    _, ids = proposal_check(spec, {"RDRLeavesTableRelativeRate": {"3": 0.0070}})
    assert "leaf_rdr_non_decreasing_with_temp" in ids


@test("above-ground partitioning must stay closed at 1")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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
    spec = cc.load_spec(CONFIG, "spring_barley", "lai")
    violations, ids = proposal_check(spec, {})
    assert "above_ground_partitioning_sums_to_one" not in ids, [v.message for v in violations]


@test("RUE must not rise after anthesis; storage organs must not fall back")
def _():
    # RUETableDVS is [0, 1, 1.3, 2]; the rule looks at the nodes from DVS 1.0 on.
    # Node 2 (DVS 1.3) raised above node 1 (DVS 1.0) is the violation, and it sits
    # inside both the bounds and the step limit.
    spec = cc.load_spec(CONFIG, "winter_wheat", "yield")
    _, ids = proposal_check(spec, {"RUETableRUE": {"2": 4.2}})     # 3.5 -> 4.2, above 3.8
    assert "rue_declines_after_anthesis" in ids, ids
    _, ids = proposal_check(spec, {"RUETableRUE": {"2": 3.2}})     # a legitimate decrease
    assert "rue_declines_after_anthesis" not in ids, ids

    spec = cc.load_spec(CONFIG, "winter_rapeseed", "yield")
    _, ids = proposal_check(spec, {"StorageOrgansPartitioningTableFraction": {"8": 0.4}})
    assert "storage_organ_non_decreasing" in ids, ids


@test("N threshold ordering is enforced")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "yield")
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
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    with tempfile.TemporaryDirectory() as tmp:
        original = cc.LEDGER_ROOT
        cc.LEDGER_ROOT = Path(tmp)
        try:
            assert cc.next_iteration(spec) == 0
            for i, objective in enumerate([1.0, 0.8, 0.9, 0.85]):
                cc.append_ledger(spec, {"iteration": i, "status": "completed",
                                        "objective": objective,
                                        "parameters_changed": {"RGRLAI": 0.02}})
            records = cc.read_ledger(spec)
            assert [r["iteration"] for r in records] == [0, 1, 2, 3]
            assert cc.next_iteration(spec) == 4
            assert cc.best_record(spec)["iteration"] == 1

            stop = cc.stop_check(spec)
            assert stop["n_completed"] == 4
            assert stop["best_objective"] == 0.8
            assert stop["since_improvement"] == 2, stop
            assert stop["stop"] is False

            # Flat iterations up to the configured patience must not stop it; one
            # past it must. Read the value rather than hard-coding it — the
            # stopping rules are a tuning knob and the invariant is the behaviour,
            # not the number.
            patience = int(spec.stopping["patience"])
            n = 4
            for _ in range(patience - stop["since_improvement"]):
                cc.append_ledger(spec, {"iteration": n, "status": "completed",
                                        "objective": 0.95, "parameters_changed": {}})
                n += 1
            stop = cc.stop_check(spec)
            assert stop["since_improvement"] == patience, stop
            assert stop["stop"] is True and "patience" in stop["reason"], stop
        finally:
            cc.LEDGER_ROOT = original


@test("a failed iteration keeps its slot and is excluded from best")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
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


@test("each target gets its own isolated run dir")
def _():
    seen = {}
    for crop in CROPS:
        for target in ("phenology", "lai", "yield"):
            run_dir = cc.load_spec(CONFIG, crop, target).run.run_dir
            assert run_dir.name.startswith("calib_"), run_dir
            assert run_dir not in seen, f"{crop}/{target} collides with {seen.get(run_dir)}"
            seen[run_dir] = f"{crop}/{target}"


@test("nothing in the workflow writes to the production crop.xml")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    production = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    assert spec.crop_xml != production
    assert "runs_optim" in str(spec.crop_xml)


# ---------------------------------------------------------------------------
print("\nphenology target")


@test("phenology is protected and its space is the thermal-time set")
def _():
    spec = cc.load_spec(CONFIG, "winter_wheat", "phenology")
    assert spec.protected, "the phenology target must be protected"
    names = set(spec.space.get("single_value_params") or {}) | \
        set(spec.space.get("multi_value_params") or {})
    for pid in ("TSUM1", "TSUM2", "TSUMEM", "TBASEM", "TEFFMX"):
        assert pid in names, f"{pid} must be calibratable when recalibrating phenology"
    # Nothing is frozen here — the protection is the gate, not a freeze — but the
    # values must still be recoverable, which is what optimized_baseline.json is for.
    assert spec.frozen == []
    assert spec.protected_path.name == "optimized_baseline.json"


@test("phenology parameters frozen elsewhere are exactly the ones this target moves")
def _():
    phen = cc.load_spec(CONFIG, "winter_wheat", "phenology")
    lai = cc.load_spec(CONFIG, "winter_wheat", "lai")
    movable = set(phen.space.get("single_value_params") or {}) | \
        set(phen.space.get("multi_value_params") or {})
    # Everything the phenology agent can move must be frozen during LAI calibration,
    # or the freeze would not actually protect the optimized set.
    assert movable <= set(lai.frozen), f"not frozen during LAI: {movable - set(lai.frozen)}"


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
    """An LAI agent whose calibrate.py calls are replaced by canned answers.

    Exercises the agent's own logic — context building, reply parsing, the repair
    round-trip, what reaches the ledger — without a model server or a cluster.
    """
    import agents as ag

    spec = cc.load_spec(CONFIG, "winter_wheat", "lai")
    rows = cc.describe_space(spec)
    status = {
        "crop": "winter_wheat", "target": "lai", "objective": "lai_rmse_dvs_binned",
        "n_completed": 1, "next_iteration": 1, "project_rows": 300,
        "project_locations": 150, "subset": {"n_locations": 150},
        "frozen": {"parameters": spec.frozen, "intact": True, "drift": []},
        "best": {"iteration": 0, "objective": 0.78},
        "stopping": {"stop": False, "reason": "criteria not met", "n_completed": 1},
        "parameters": rows, "constraints": spec.constraints,
        "ledger_dir": "/tmp/ledger",
    }
    history = [{
        "iteration": 0, "objective": 0.78, "status": "completed", "improved": True,
        "parameters_changed": {}, "reason": "baseline",
        "diagnostics": {"by_dvs_bin": [{"dvs_bin": "1.00-1.25 anthesis", "bias": -0.6}]},
    }]

    agent = ag.LAIAgent("winter_wheat", ag.MockBackend(replies=list(replies)),
                        config_path=CONFIG, verbose=False)
    executed = {}
    agent.status = lambda: status
    agent.history = lambda: history
    agent.preflight = lambda params: {
        "valid": valid, "violations": list(violations), "blocked_by_protection": False}
    agent.execute = lambda decision: executed.update(decision=decision) or {"status": "ok"}
    return agent, executed


@test("the context carries values, bounds, meaning, history and the schema")
def _():
    agent, _ = _stub_agent([])
    context = agent.context(agent.status(), agent.history())
    assert "RGRLAI" in context and "SLATableSLA" in context
    assert "bounds" in context
    assert "Maximum relative growth rate" in context, "the biological meaning must be present"
    assert "iteration 0: objective 0.7800" in context
    assert "parameter_changes" in context, "the reply schema must be stated"
    assert "TSUM1" in context, "the frozen list must be visible so the model does not try"


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
    assert "ruled out RGRLAI" in blob and "lai agent" in blob


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
            return {"valid": False, "blocked_by_protection": False, "violations": [
                {"constraint": "bounds", "message": "SLATableSLA[3]=0.9 outside [0.006, 0.025]"}]}
        return {"valid": True, "violations": [], "blocked_by_protection": False}

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


@test("the phenology agent will not propose anything in validation mode")
def _():
    import agents as ag
    agent = ag.PhenologyAgent("winter_wheat", ag.MockBackend(replies=[]),
                              config_path=CONFIG, mode="validate", verbose=False)
    assert agent.allow_recalibration is False
    executed = {}
    agent.status = lambda: {"next_iteration": 0, "crop": "winter_wheat", "target": "phenology"}
    agent.execute = lambda d: executed.update(decision=d) or {"status": "ok"}
    agent.iterate()
    assert executed["decision"].parameters == {}, "validation must change nothing"
    assert agent.backend.calls == [], "validation must not consult the model"

    recal = ag.PhenologyAgent("winter_wheat", ag.MockBackend(replies=[]),
                              config_path=CONFIG, mode="recalibrate", verbose=False)
    assert recal.allow_recalibration is True


@test("every agent has a prompt, and the prompts state their hard rules")
def _():
    import agents as ag
    from agents.base import PROMPT_DIR
    required = {
        "lai.md": ["Peak timing is not yours to fix", "frozen"],
        "yield.md": ["YieldAdjustRatio", "sum to one", "also move LAI"],
        "phenology.md": ["already been optimized", "duration", "TSUM1"],
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
    failed = [(n, tb) for n, ok, tb in _results if not ok]
    print(f"\n{'=' * 70}")
    print(f"  {len(_results) - len(failed)}/{len(_results)} passed")
    print("=" * 70)
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
