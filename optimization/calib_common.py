#!/usr/bin/env python
"""Plumbing for the agentic SIMPLACE calibration.

This module is the *decision-support* layer that stands in for an optimizer's
search machinery. It owns four things and nothing else:

  1. **Space resolution** — turning the human-readable parameter declarations in
     ``calibration.yaml`` into concrete per-crop bounds, by reading what is
     actually in that crop's ``crop.xml``. Wheat's SLA table has 8 nodes,
     rapeseed 5, potato 3; relative bounds fit whatever is there.
  2. **The freeze guard** — a snapshot of the optimized phenology parameters,
     re-verified against the written XML after every single parameter change.
  3. **The constraint engine** — bounds, table shape rules, partitioning
     closure, step-size and blast-radius limits.
  4. **The ledger** — an append-only record of every iteration, with the reason
     and the reasoning behind each parameter change.

Everything below this layer comes from ``common.py``: staging the isolated run
dir, editing ``crop.xml``, invoking SIMPLACE, reading outputs. The losses come
from ``objectives.py``. Neither depends on who proposed the parameters, which is
what makes a local-agent iteration, a Claude iteration and a hand-typed one
directly comparable.

Isolation contract
------------------
Calibration runs live in ``simplace/<crop>/runs_optim/calib_<target>/``, one
directory per target, so no two targets can disturb each other. Nothing writes to
``simplace/<crop>/data/crop/crop.xml`` except the explicit ``calibrate.py
promote`` step.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OPTIM_DIR = Path(__file__).resolve().parent
DEFAULT_CALIB_CONFIG = OPTIM_DIR / "calibration.yaml"
LEDGER_ROOT = OPTIM_DIR / "calibration"

# A value this small is treated as a structural zero: relative bounds around it
# are meaningless, so the parameter stays pinned (see _bounds_for).
ZERO = 1e-12


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# crop.xml reading (generic — common.py only reads what is in a search space)
# ---------------------------------------------------------------------------
def crop_root(xml_path: Path):
    return etree.parse(str(xml_path)).getroot()


def read_param(root, crop_name: str, pid: str):
    """Raw value of one parameter: float, list[float], or str for non-numerics.

    Returns ``None`` when the crop does not define the parameter — several
    parameters in the space are crop-specific, and a missing one is skipped
    rather than being an error.
    """
    nodes = common._crop_params(root, crop_name, pid)
    if not nodes:
        return None
    node = nodes[0]
    values = node.findall("value")
    if values:
        try:
            return [float(v.text) for v in values]
        except (TypeError, ValueError):
            return [(v.text or "").strip() for v in values]
    text = (node.text or "").strip()
    try:
        return float(text)
    except ValueError:
        return text


def read_params(root, crop_name: str, pids) -> dict:
    """``{pid: value}`` for every requested parameter the crop actually defines."""
    out = {}
    for pid in pids:
        value = read_param(root, crop_name, pid)
        if value is not None:
            out[pid] = value
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_calib_config(path: Path) -> dict:
    """Load calibration.yaml and merge in whatever ``extends:`` points at.

    config.yaml owns the model/run side (crops, climate, slurm, input profiles,
    per-target run definitions); calibration.yaml owns the calibration side
    (parameter spaces, constraints, freeze groups, stopping, llm). Keys defined in
    calibration.yaml win.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base_name = cfg.pop("extends", None)
    if not base_name:
        return cfg
    base_path = (path.parent / base_name).resolve()
    if not base_path.exists():
        raise SystemExit(f"calibration config extends {base_name!r}, which does not exist: {base_path}")
    base = yaml.safe_load(base_path.read_text())
    merged = common._merge(base, cfg)
    # `targets` means different things in the two files: run definitions in the
    # base, parameter spaces here. Keep them apart rather than deep-merging two
    # schemas that only share their keys.
    merged["targets"] = cfg.get("targets", {})
    merged["run_targets"] = base.get("targets", {})
    merged["_base_config"] = str(base_path)
    return merged


# ---------------------------------------------------------------------------
# Parameter space resolution
# ---------------------------------------------------------------------------
def _bounds_for(current: float, pc: dict) -> tuple[float, float]:
    """Concrete (low, high) for one value, per the declared bound mode."""
    mode = pc.get("mode", "relative")
    if mode == "absolute":
        low, high = float(pc["low"]), float(pc["high"])
    else:
        if abs(current) <= ZERO:
            # A structural zero (e.g. leaf allocation after anthesis). Relative
            # bounds collapse to nothing, which is the right answer: keep it
            # pinned instead of inventing a range around zero.
            return 0.0, 0.0
        f_lo, f_hi = pc.get("factor", [0.5, 2.0])
        low, high = current * float(f_lo), current * float(f_hi)
        if low > high:
            low, high = high, low

    clip = pc.get("clip")
    if clip:
        low, high = max(low, float(clip[0])), min(high, float(clip[1]))
    # The current value must always be attainable, otherwise the very first
    # "keep everything" proposal would be rejected by its own bounds.
    low, high = min(low, current), max(high, current)
    return float(low), float(high)


def resolve_space(xml_path: Path, crop_name: str, params_cfg: dict) -> tuple[dict, dict]:
    """Declared parameters -> (``common``-schema space, per-parameter metadata).

    The returned space is exactly the schema ``common.bounds_of``,
    ``common.check_within_bounds`` and ``common.apply_parameters`` already
    understand, so nothing downstream needs to know this layer exists.
    """
    root = crop_root(xml_path)
    single: dict[str, dict] = {}
    multi: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    for pid, pc in (params_cfg or {}).items():
        current = read_param(root, crop_name, pid)
        if current is None:
            meta[pid] = {**pc, "present": False,
                         "note": f"not defined for crop {crop_name!r}; excluded"}
            continue

        kind = pc.get("kind", "scalar")
        ptype = pc.get("type", "float")
        precision = int(pc.get("precision", 4))

        if kind == "table":
            if not isinstance(current, list):
                raise SystemExit(f"{pid} declared as a table but crop.xml holds a scalar")
            values = [dict(zip(("low", "high"), _bounds_for(float(c), pc))) for c in current]
            multi[pid] = {"type": ptype, "values": values, "precision": precision}
        else:
            if isinstance(current, list):
                raise SystemExit(f"{pid} declared as a scalar but crop.xml holds a table")
            low, high = _bounds_for(float(current), pc)
            single[pid] = {"type": ptype, "low": low, "high": high, "precision": precision}

        meta[pid] = {**pc, "present": True, "current": current,
                     "pinned": kind != "table" and _bounds_for(float(current), pc) == (0.0, 0.0)}

    return {"single_value_params": single or None, "multi_value_params": multi or None}, meta


def flatten(values: dict) -> dict[str, float]:
    """``{pid: scalar | [..]}`` -> ``{flat name: value}`` (``<param>_<index>`` for tables)."""
    out: dict[str, float] = {}
    for pid, value in values.items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                out[f"{pid}_{i}"] = v
        else:
            out[pid] = value
    return out


def unflatten(flat: dict, space: dict) -> dict:
    """Inverse of :func:`flatten`, driven by the space's table lengths."""
    out: dict[str, object] = {}
    for pid, spec in (space.get("single_value_params") or {}).items():
        if pid in flat:
            out[pid] = flat[pid]
    for pid, spec in (space.get("multi_value_params") or {}).items():
        keys = [f"{pid}_{i}" for i in range(len(spec["values"]))]
        if any(k in flat for k in keys):
            out[pid] = [flat[k] for k in keys]
    return out


def current_values(xml_path: Path, crop_name: str, space: dict) -> dict:
    """Current value of every parameter in the space, as ``{pid: scalar | [..]}``."""
    root = crop_root(xml_path)
    out: dict[str, object] = {}
    for pid in list((space.get("single_value_params") or {})) + \
               list((space.get("multi_value_params") or {})):
        value = read_param(root, crop_name, pid)
        if value is not None:
            out[pid] = value
    return out


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------
def normalise_proposal(proposed: dict, current: dict, space: dict) -> dict:
    """Expand a proposal into complete ``{pid: scalar | full list}`` values.

    A proposal names only what it changes. Table edits may be given either as a
    full list or as ``{"3": 0.0115}`` — index-keyed, which is how a targeted
    single-node SLA change is most naturally expressed.
    """
    tables = space.get("multi_value_params") or {}
    scalars = space.get("single_value_params") or {}
    out: dict[str, object] = {}

    for pid, value in (proposed or {}).items():
        if pid not in tables and pid not in scalars:
            raise SystemExit(
                f"{pid!r} is not a calibratable parameter for this target.\n"
                f"  available: {sorted(list(scalars) + list(tables))}"
            )
        if pid in scalars:
            if isinstance(value, (list, dict)):
                raise SystemExit(f"{pid} is a scalar; got {type(value).__name__}")
            out[pid] = float(value)
            continue

        base = list(current.get(pid, []))
        n = len(tables[pid]["values"])
        if len(base) != n:
            raise SystemExit(
                f"{pid}: the resolved space has {n} elements but crop.xml holds {len(base)}. "
                f"The space was anchored on a different file — restage the run dir.")
        if isinstance(value, dict):
            merged = list(base)
            for key, v in value.items():
                idx = int(key)
                if not 0 <= idx < n:
                    raise SystemExit(f"{pid}: index {idx} out of range (table has {n} elements)")
                merged[idx] = float(v)
            out[pid] = merged
        elif isinstance(value, list):
            if len(value) != n:
                raise SystemExit(
                    f"{pid}: expected {n} values for this crop, got {len(value)}. "
                    f"Use an index-keyed object to change single elements."
                )
            out[pid] = [float(v) for v in value]
        else:
            raise SystemExit(f"{pid} is a table; got a scalar")
    return out


def changed_entries(proposal: dict, current: dict, atol: float = 1e-12) -> dict:
    """What the proposal actually alters: ``{pid: {"from":…, "to":…, "indices":[…]}}``."""
    out: dict[str, dict] = {}
    for pid, new in proposal.items():
        old = current.get(pid)
        if isinstance(new, list):
            old_list = list(old or [])
            idx = [i for i, v in enumerate(new)
                   if i >= len(old_list) or abs(float(v) - float(old_list[i])) > atol]
            if idx:
                out[pid] = {"from": old_list, "to": list(new), "indices": idx,
                            "n_elements_changed": len(idx)}
        else:
            if old is None or abs(float(new) - float(old)) > atol:
                out[pid] = {"from": old, "to": new, "indices": None, "n_elements_changed": 1}
    return out


# ---------------------------------------------------------------------------
# Constraint engine
# ---------------------------------------------------------------------------
@dataclass
class Violation:
    constraint: str
    parameter: str | None
    message: str
    severity: str = "error"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _interp_table(root, crop_name: str, prefix: str) -> tuple[list, list] | None:
    """``(dvs_grid, fractions)`` for a ``<prefix>DVS`` / ``<prefix>Fraction`` pair."""
    grid = read_param(root, crop_name, f"{prefix}DVS")
    frac = read_param(root, crop_name, f"{prefix}Fraction")
    if not isinstance(grid, list) or not isinstance(frac, list):
        return None
    n = min(len(grid), len(frac))
    return [float(x) for x in grid[:n]], [float(y) for y in frac[:n]]


def _partition_sums(root, crop_name: str, prefixes: list[str],
                    override: dict) -> dict[float, float]:
    """Above-ground allocation summed on the union DVS grid of the given tables.

    ``override`` supplies proposed fraction lists keyed by ``<prefix>Fraction``.
    Interpolating onto the union grid is the model-faithful reading: SIMPLACE
    evaluates each table independently at the current DVS, so the closure that
    matters is the sum of the *interpolated* values.
    """
    tables = []
    for prefix in prefixes:
        pair = _interp_table(root, crop_name, prefix)
        if pair is None:
            continue
        grid, frac = pair
        key = f"{prefix}Fraction"
        if key in override:
            proposed = [float(v) for v in override[key]]
            frac = proposed[:len(grid)] + frac[len(proposed):]
        tables.append((grid, frac))
    if not tables:
        return {}

    nodes = sorted({x for grid, _ in tables for x in grid})
    return {x: float(sum(np.interp(x, grid, frac) for grid, frac in tables)) for x in nodes}


def _monotone_violations(name: str, pid: str, values: list, direction: str,
                         tol: float, offset: int = 0) -> list[Violation]:
    out = []
    for i in range(1, len(values)):
        delta = float(values[i]) - float(values[i - 1])
        bad = (direction == "non_increasing" and delta > tol) or \
              (direction == "non_decreasing" and delta < -tol)
        if bad:
            out.append(Violation(
                name, pid,
                f"{pid}[{i + offset}]={values[i]:.6g} breaks {direction} against "
                f"[{i - 1 + offset}]={values[i - 1]:.6g}"))
    return out


def validate(proposal: dict, current: dict, space: dict, meta: dict,
             constraints: list, frozen: list[str], xml_path: Path,
             crop_name: str) -> list[Violation]:
    """Every rule the proposal must satisfy, evaluated against the *current* state.

    Returns the violations; an empty list means the proposal may be applied.
    """
    out: list[Violation] = []
    merged = {**current, **proposal}
    changed = changed_entries(proposal, current)
    root = crop_root(xml_path)

    # --- frozen parameters ---------------------------------------------------
    for pid in proposal:
        if pid in frozen:
            out.append(Violation("frozen", pid,
                                 f"{pid} is frozen for this target and may not be changed"))

    # --- bounds --------------------------------------------------------------
    outside = common.check_within_bounds(flatten(proposal), space)
    limits = common.bounds_of(space)
    for name in outside:
        low, high = limits[name]
        out.append(Violation("bounds", name.rsplit("_", 1)[0] if name not in space else name,
                             f"{name}={flatten(proposal)[name]:.6g} outside [{low:.6g}, {high:.6g}]"))

    # --- declared constraints ------------------------------------------------
    for rule in constraints or []:
        kind, cid = rule.get("type"), rule.get("id", rule.get("type", "?"))

        if kind == "table_smoothness":
            pid = rule["param"]
            if pid not in merged or not isinstance(merged[pid], list):
                continue
            values = [float(v) for v in merged[pid]]
            max_ratio = float(rule.get("max_ratio", 2.0))
            for i in range(1, len(values)):
                a, b = abs(values[i - 1]), abs(values[i])
                if min(a, b) <= ZERO:
                    continue
                ratio = max(a, b) / min(a, b)
                if ratio > max_ratio:
                    out.append(Violation(
                        cid, pid,
                        f"{pid}[{i-1}]={values[i-1]:.6g} -> [{i}]={values[i]:.6g} "
                        f"is a {ratio:.2f}x step (max {max_ratio})"))

        elif kind == "table_monotone":
            pid = rule["param"]
            if pid in merged and isinstance(merged[pid], list):
                out += _monotone_violations(cid, pid, merged[pid],
                                            rule.get("direction", "non_decreasing"),
                                            float(rule.get("tolerance", 0.0)))

        elif kind == "table_monotone_from":
            pid = rule["param"]
            grid = read_param(root, crop_name, rule["grid_param"])
            if pid in merged and isinstance(merged[pid], list) and isinstance(grid, list):
                from_x = float(rule["from_x"])
                keep = [i for i, x in enumerate(grid[:len(merged[pid])]) if float(x) >= from_x]
                if len(keep) > 1:
                    out += _monotone_violations(
                        cid, pid, [merged[pid][i] for i in keep],
                        rule.get("direction", "non_increasing"),
                        float(rule.get("tolerance", 0.0)), offset=keep[0])

        elif kind == "table_endpoint_order":
            pid = rule["param"]
            if pid not in merged or not isinstance(merged[pid], list) or not merged[pid]:
                continue
            values = [float(v) for v in merged[pid]]
            tol = float(rule.get("tolerance", 0.0))
            what = rule.get("rule", "last_le_max")
            # `max(values)` would include the last element itself and never fire.
            if what == "last_le_max" and len(values) > 1 and values[-1] > max(values[:-1]) + tol:
                out.append(Violation(
                    cid, pid,
                    f"{pid} last value {values[-1]:.6g} exceeds the maximum of the "
                    f"earlier nodes ({max(values[:-1]):.6g})"))
            if what == "last_le_first" and values[-1] > values[0] + tol:
                out.append(Violation(
                    cid, pid,
                    f"{pid} last value {values[-1]:.6g} exceeds the first {values[0]:.6g}"))

        elif kind == "partition_sum":
            prefixes = rule["tables"]
            tol = float(rule.get("tolerance", 0.005))
            before = _partition_sums(root, crop_name, prefixes, {})
            after = _partition_sums(root, crop_name, prefixes, proposal)
            for x, total in after.items():
                was_ok = abs(before.get(x, total) - 1.0) <= tol
                if was_ok and abs(total - 1.0) > tol:
                    out.append(Violation(
                        cid, None,
                        f"above-ground allocation sums to {total:.4f} at DVS={x:g} "
                        f"(was {before.get(x, float('nan')):.4f}); leaves + stems + "
                        f"storage organs must stay at 1"))

        elif kind == "max_changed":
            limit = int(rule.get("limit", 3))
            if len(changed) > limit:
                out.append(Violation(
                    cid, None,
                    f"{len(changed)} parameters changed ({', '.join(sorted(changed))}); "
                    f"at most {limit} per iteration so the effect stays attributable"))
            max_elements = rule.get("max_elements")
            if max_elements is not None:
                total = sum(c["n_elements_changed"] for c in changed.values())
                if total > int(max_elements):
                    out.append(Violation(
                        cid, None,
                        f"{total} individual values changed; at most {max_elements} per iteration"))

        elif kind == "relative_step":
            frac = float(rule.get("max_fraction", 0.5))
            for pid, change in changed.items():
                pairs = (zip(change["from"], change["to"]) if change["indices"] is not None
                         else [(change["from"], change["to"])])
                for i, (old, new) in enumerate(pairs):
                    if old is None or abs(float(old)) <= ZERO:
                        continue
                    moved = abs(float(new) - float(old)) / abs(float(old))
                    if moved > frac + 1e-9:
                        label = f"{pid}[{i}]" if change["indices"] is not None else pid
                        out.append(Violation(
                            cid, pid,
                            f"{label} moves {moved:.0%} ({old:.6g} -> {new:.6g}); "
                            f"max {frac:.0%} per iteration"))

        elif kind == "ordering":
            expr = rule.get("rule", "")
            for op in ("<=", ">=", "<", ">"):
                if op in expr:
                    lhs, rhs = (s.strip() for s in expr.split(op, 1))
                    a, b = merged.get(lhs), merged.get(rhs)
                    if a is None or b is None:
                        a = a if a is not None else read_param(root, crop_name, lhs)
                        b = b if b is not None else read_param(root, crop_name, rhs)
                    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                        break
                    ok = {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
                    if not ok:
                        out.append(Violation(cid, None,
                                             f"{lhs}={a:g} {op} {rhs}={b:g} is violated"))
                    break

    return out


# ---------------------------------------------------------------------------
# Closure locks
# ---------------------------------------------------------------------------
def apply_closure_locks(space: dict, meta: dict, constraints: list, frozen: list[str],
                        xml_path: Path, crop_name: str) -> dict:
    """Pin table elements that a closure constraint makes mathematically immovable.

    A ``partition_sum`` rule says leaves + stems + storage organs must stay at 1 at
    every development stage. An element can therefore only move if *some other
    member of the group* can move at the same stage to absorb the difference.

    Winter wheat is the case that forced this. Its above-ground partitioning is a
    step function — everything to stems until DVS 0.95, everything to the storage
    organ from DVS 1.0 on — so at every node past anthesis leaves and stems are
    structural zeros, already pinned at [0, 0]. Storage organs sit at 1.0 with a
    nominal range of [0.7, 1], which *looks* adjustable and is not: there is
    nothing left to take the other 0.3.

    Without this, an agent is handed a parameter it cannot legally change, and
    every proposal it makes is correctly rejected by the closure rule. It has no
    way to learn that from the rejection, so it burns the whole repair budget
    rediscovering the same wall. Collapsing the bounds tells it the truth up
    front.

    A member counts as movable at a node only if it is in *this target's* space
    (a frozen parameter cannot absorb anything either) and its bounds there are
    non-degenerate. Returns ``{parameter: [locked indices]}``.
    """
    root = crop_root(xml_path)
    multi = space.get("multi_value_params") or {}
    frozen_set = set(frozen or [])
    locks: dict[str, list[int]] = {}

    def movable(pid: str, i: int) -> bool:
        if pid in frozen_set or pid not in multi:
            return False
        values = multi[pid]["values"]
        return i < len(values) and (values[i]["high"] - values[i]["low"]) > ZERO

    for rule in constraints or []:
        if rule.get("type") != "partition_sum":
            continue
        tables = rule.get("tables") or []
        members = [f"{t}Fraction" for t in tables]
        grids = [read_param(root, crop_name, f"{t}DVS") for t in tables]

        # Only the shared-grid case is decidable element by element. When the
        # members sit on different DVS grids (spring barley), a change to any node
        # of one table moves the interpolated curve over a whole interval, so
        # "which node could compensate" has no clean answer — leave those alone
        # rather than over-lock and hide a legitimate lever.
        if any(g is None for g in grids) or len({tuple(g) for g in grids}) != 1:
            continue

        for pid in members:
            if pid not in multi:
                continue
            for i in range(len(grids[0])):
                if not movable(pid, i):
                    continue
                if any(movable(other, i) for other in members if other != pid):
                    continue
                # Nothing else can absorb a change here: pin it at its current value.
                current = read_param(root, crop_name, pid)
                value = float(current[i])
                multi[pid]["values"][i] = {"low": value, "high": value}
                locks.setdefault(pid, []).append(i)

    for pid, indices in locks.items():
        info = meta.get(pid)
        if info is None:
            continue
        info["locked_indices"] = indices
        info["locked_reason"] = (
            "no other member of the above-ground partitioning group can move at "
            "these development stages, so the closure constraint fixes them")
        values = multi[pid]["values"]
        info["movable"] = len(indices) < len(values)

    return locks


# ---------------------------------------------------------------------------
# Freeze guard
# ---------------------------------------------------------------------------
def frozen_ids(cfg: dict, target_cfg: dict) -> list[str]:
    """Resolve the target's ``frozen:`` groups to concrete parameter ids."""
    groups = cfg.get("frozen_groups", {}) or {}
    out: list[str] = []
    for group in target_cfg.get("frozen", []) or []:
        if group == "lai_calibrated":
            out += list((cfg["targets"].get("lai", {}).get("parameters") or {}))
        else:
            if group not in groups:
                raise SystemExit(f"unknown frozen group {group!r}; have {sorted(groups)}")
            out += list(groups[group])
    # A parameter that a later target needs as a structural counterweight (the stem
    # fraction keeps above-ground allocation closed) must be excused explicitly.
    excluded = set(target_cfg.get("frozen_exclude", []) or [])
    return sorted(set(out) - excluded)


def snapshot_frozen(xml_path: Path, crop_name: str, ids: list[str]) -> dict:
    """Exact current values of the frozen parameters, plus a digest."""
    root = crop_root(xml_path)
    values = read_params(root, crop_name, ids)
    payload = json.dumps(values, sort_keys=True)
    return {"crop_name": crop_name, "parameters": values,
            "digest": hashlib.sha256(payload.encode()).hexdigest(),
            "source": str(xml_path), "taken_at": utcnow()}


def verify_frozen(xml_path: Path, snapshot: dict) -> list[str]:
    """Differences between the XML and the frozen snapshot. Empty = unchanged.

    This is checked by re-reading the file that SIMPLACE will actually consume,
    after the parameters have been written — not by trusting that we only asked
    for non-frozen edits.
    """
    root = crop_root(xml_path)
    now = read_params(root, snapshot["crop_name"], list(snapshot["parameters"]))
    diffs = []
    for pid, was in snapshot["parameters"].items():
        is_now = now.get(pid)
        if isinstance(was, list) and isinstance(is_now, list):
            if len(was) != len(is_now) or any(
                    abs(float(a) - float(b)) > 1e-9 for a, b in zip(was, is_now)):
                diffs.append(f"{pid}: {was} -> {is_now}")
        elif isinstance(was, float) and isinstance(is_now, float):
            if abs(was - is_now) > 1e-9:
                diffs.append(f"{pid}: {was} -> {is_now}")
        elif was != is_now:
            diffs.append(f"{pid}: {was!r} -> {is_now!r}")
    return diffs


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass
class CalibSpec:
    """One agentic calibration target for one crop, fully resolved."""

    run: common.RunSpec          # staging / XML / SIMPLACE, reused from common.py
    cfg: dict
    target: str
    target_cfg: dict
    space: dict
    meta: dict
    frozen: list[str]
    constraints: list = field(default_factory=list)
    # Filled in by stage(): row/location counts of the staged project table.
    stage_info: dict = field(default_factory=dict)

    @property
    def crop(self) -> str:
        return self.run.crop

    @property
    def crop_name(self) -> str:
        return self.run.crop_name

    @property
    def crop_xml(self) -> Path:
        return self.run.crop_xml

    @property
    def ledger_dir(self) -> Path:
        d = LEDGER_ROOT / f"{self.crop}__{self.target}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def state_path(self) -> Path:
        return self.ledger_dir / "state.json"

    @property
    def ledger_path(self) -> Path:
        return self.ledger_dir / "ledger.jsonl"

    @property
    def frozen_path(self) -> Path:
        return self.ledger_dir / "frozen_snapshot.json"

    @property
    def protected_path(self) -> Path:
        """Where a protected target's already-optimized values are preserved."""
        return self.ledger_dir / "optimized_baseline.json"

    @property
    def protected(self) -> bool:
        return bool(self.target_cfg.get("protected"))

    @property
    def stopping(self) -> dict:
        return self.target_cfg.get("stopping", {}) or {}

    def iteration_dir(self, n: int) -> Path:
        d = self.ledger_dir / "iterations" / f"iter_{n:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_spec(config_path: Path, crop: str, target: str, device: str = "cluster",
              n_locations: int | None = None) -> CalibSpec:
    """Resolve one agentic calibration target.

    Builds on ``common.load_spec`` (which validates the crop, resolves the repo
    root, and stages the input profile) and then redirects the run to the agentic
    run dir with the agentic parameter space and subset.
    """
    cfg = load_calib_config(Path(config_path))
    if target not in cfg["targets"]:
        raise SystemExit(f"unknown target {target!r}; calibration.yaml has {sorted(cfg['targets'])}")
    tcfg = cfg["targets"][target]

    # common.load_spec resolves the crop, the repo root and the input profile
    # from the base config; everything the calibration layer owns is replaced below.
    base_cfg_path = Path(cfg.get("_base_config", OPTIM_DIR / "config.yaml"))
    run = common.load_spec(base_cfg_path, crop, target, device=device)

    subset = dict(tcfg.get("subset") or {})
    if n_locations is not None:
        subset["n_locations"] = n_locations

    crop_dir = run.crop_dir
    project_src = crop_dir / tcfg["project_csv"].format(crop=crop)
    if not project_src.exists():
        raise SystemExit(f"project table missing: {project_src}")

    # The staged per-location tables follow calibration.yaml's `inputs:`, which may
    # name a different profile than the base config's target of the same name.
    inputs = run.inputs
    profile_name = tcfg.get("inputs")
    if profile_name:
        profile = cfg["input_profiles"][profile_name]
        inputs = {}
        for role, rel in profile.items():
            path = crop_dir / rel.format(crop=crop)
            if not path.exists() and profile_name != "production":
                path = crop_dir / cfg["input_profiles"]["production"][role].format(crop=crop)
                print(f"  note: {profile_name} {role} table missing, using {path.name}")
            if not path.exists():
                raise SystemExit(f"required input missing: {path}")
            inputs[role] = path

    run = dataclasses.replace(
        run,
        run_dir=crop_dir / "runs_optim" / tcfg.get("run_subdir", f"calib_{target}"),
        exp_name=tcfg["exp_name"],
        project_src=project_src,
        inputs=inputs,
        subset=subset,
    )

    # The space must be resolved against the crop.xml this target will actually
    # mutate. Before the first staging that file does not exist yet, so fall back
    # to the production one — which is exactly what staging will copy in.
    xml_for_space = run.crop_xml if run.crop_xml.exists() else crop_dir / "data" / "crop" / "crop.xml"
    space, meta = resolve_space(xml_for_space, run.crop_name, tcfg.get("parameters") or {})
    frozen = frozen_ids(cfg, tcfg)
    constraints = tcfg.get("constraints") or []
    apply_closure_locks(space, meta, constraints, frozen, xml_for_space, run.crop_name)
    run = dataclasses.replace(run, parameters=space)

    return CalibSpec(
        run=run, cfg=cfg, target=target, target_cfg=tcfg,
        space=space, meta=meta,
        frozen=frozen,
        constraints=constraints,
    )


def refresh_space(spec: CalibSpec) -> CalibSpec:
    """Re-resolve bounds against the staged crop.xml (call after build_run_dir).

    Bounds are relative to the *starting* values, so they must be anchored once
    the run dir exists — in particular after a yield handoff replaces crop.xml
    with the LAI-calibrated one.
    """
    space, meta = resolve_space(spec.crop_xml, spec.crop_name,
                                spec.target_cfg.get("parameters") or {})
    apply_closure_locks(space, meta, spec.constraints, spec.frozen,
                        spec.crop_xml, spec.crop_name)
    spec.space, spec.meta = space, meta
    spec.run = dataclasses.replace(spec.run, parameters=space)
    return spec


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
def read_state(spec: CalibSpec) -> dict:
    if spec.state_path.exists():
        return json.loads(spec.state_path.read_text())
    return {}


def write_state(spec: CalibSpec, state: dict) -> None:
    spec.state_path.write_text(json.dumps(state, indent=2) + "\n")


def read_ledger(spec: CalibSpec) -> list[dict]:
    if not spec.ledger_path.exists():
        return []
    return [json.loads(line) for line in spec.ledger_path.read_text().splitlines() if line.strip()]


def append_ledger(spec: CalibSpec, record: dict) -> None:
    """Append one iteration. The ledger is never rewritten, only appended to."""
    with open(spec.ledger_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def next_iteration(spec: CalibSpec) -> int:
    records = read_ledger(spec)
    return max((r.get("iteration", -1) for r in records), default=-1) + 1


def best_record(spec: CalibSpec) -> dict | None:
    done = [r for r in read_ledger(spec)
            if r.get("status") == "completed" and r.get("objective") is not None]
    return min(done, key=lambda r: r["objective"]) if done else None


def stop_check(spec: CalibSpec) -> dict:
    """Whether the configured stopping criteria are met, and why."""
    rules = spec.stopping
    records = [r for r in read_ledger(spec) if r.get("status") == "completed"]
    if not records:
        return {"stop": False, "reason": "no completed iterations yet",
                "n_completed": 0, "best_objective": None, "since_improvement": 0}

    best = min(records, key=lambda r: r["objective"])
    order = sorted(records, key=lambda r: r["iteration"])
    since = 0
    for record in reversed(order):
        if record["iteration"] == best["iteration"]:
            break
        since += 1

    reasons = []
    if rules.get("max_iterations") and len(records) >= int(rules["max_iterations"]):
        reasons.append(f"max_iterations reached ({len(records)})")
    if rules.get("target_objective") is not None and best["objective"] <= float(rules["target_objective"]):
        reasons.append(f"objective {best['objective']:.4f} <= target {rules['target_objective']}")
    if rules.get("patience") and since >= int(rules["patience"]):
        reasons.append(f"no improvement for {since} iterations (patience {rules['patience']})")

    return {"stop": bool(reasons), "reason": "; ".join(reasons) or "criteria not met",
            "n_completed": len(records), "best_objective": best["objective"],
            "best_iteration": best["iteration"], "since_improvement": since}


# ---------------------------------------------------------------------------
# Objectives — reuse the existing loss functions unchanged
# ---------------------------------------------------------------------------
def objective_for(target: str):
    """``(process_result, loss_fn)`` for a target.

    Deliberately a thin re-export of ``objectives.py``: whoever proposes the
    parameters — Claude, a local Ollama agent, or a person typing
    ``calibrate.py run`` by hand — the objective is computed by the same code, or
    the ledgers cannot be compared.
    """
    import objectives
    return objectives.for_target(target)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def stage(spec: CalibSpec, rebuild: bool = False) -> dict:
    """Stage the isolated run dir (``common.build_run_dir``) and anchor the space.

    A study that has not started yet is re-anchored on the production ``crop.xml``.
    ``build_run_dir`` deliberately leaves an existing ``data/crop/`` alone — it has
    to, because iteration N must continue from what iteration N-1 wrote. But that
    also means a run dir left behind by an *earlier* study keeps its mutated
    parameters forever, so a fresh study would record someone else's endpoint as
    its baseline and anchor every relative bound on it. An empty ledger means
    nothing has been recorded yet, so there is nothing to preserve.

    The handoff is the one exception: it writes the LAI-calibrated crop.xml into
    the yield run dir *before* the yield ledger exists, and ``provenance.json``
    is how that intent is marked.
    """
    n_recorded = len(read_ledger(spec))
    if rebuild and n_recorded:
        # --rebuild re-copies crop.xml from production, so iteration N+1 would
        # start from the production values rather than from what iteration N
        # wrote. The ledger would keep counting as though nothing happened, and
        # the objective series would have an unexplained discontinuity in it.
        print(f"  WARNING: --rebuild resets crop.xml to the production file, but "
              f"{spec.crop}/{spec.target} already has {n_recorded} recorded "
              f"iteration(s).\n"
              f"           The next iteration will NOT continue from the last one. "
              f"If you meant to start over,\n"
              f"           archive the ledger first: mv {spec.ledger_dir} "
              f"{spec.ledger_dir}.bak")

    info = common.build_run_dir(spec.run, rebuild=rebuild)

    production = spec.run.crop_dir / "data" / "crop" / "crop.xml"
    fresh = not spec.ledger_path.exists() and not (spec.ledger_dir / "provenance.json").exists()
    if not rebuild and fresh and production.exists() and spec.crop_xml.exists():
        if production.read_bytes() != spec.crop_xml.read_bytes():
            shutil.copyfile(production, spec.crop_xml)
            print(f"  note: no iterations recorded for {spec.crop}/{spec.target} yet — "
                  f"reset the run-dir crop.xml from {production}")

    refresh_space(spec)
    return info


def ensure_frozen_snapshot(spec: CalibSpec) -> dict:
    """Load the freeze snapshot, taking it from the current XML on first use."""
    if spec.frozen_path.exists():
        return json.loads(spec.frozen_path.read_text())
    snapshot = snapshot_frozen(spec.crop_xml, spec.crop_name, spec.frozen)
    spec.frozen_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def ensure_protected_baseline(spec: CalibSpec) -> dict:
    """Preserve a protected target's already-optimized values, once.

    The phenology set was calibrated in an earlier phase and lives in the
    production ``crop.xml``. Before a recalibration is ever allowed to touch it,
    the starting values are written here so the original is recoverable from the
    ledger alone — independent of version control, and independent of whatever
    the recalibration then does.
    """
    if spec.protected_path.exists():
        return json.loads(spec.protected_path.read_text())
    snapshot = snapshot_frozen(spec.crop_xml, spec.crop_name, sorted(spec.meta))
    snapshot["note"] = (
        f"Values of the {spec.target} parameters as they stood before any agentic "
        f"iteration. Restore with: calibrate.py restore-optimized --crop {spec.crop} "
        f"--target {spec.target}")
    spec.protected_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def protected_drift(spec: CalibSpec) -> list[str]:
    """How the current XML differs from the preserved optimized values."""
    if not spec.protected_path.exists():
        return []
    return verify_frozen(spec.crop_xml, json.loads(spec.protected_path.read_text()))


def describe_space(spec: CalibSpec) -> list[dict]:
    """One row per calibratable parameter: value, bounds, meaning, role."""
    rows = []
    limits = common.bounds_of(spec.space)
    values = current_values(spec.crop_xml, spec.crop_name, spec.space) \
        if spec.crop_xml.exists() else {}
    for pid, info in spec.meta.items():
        if not info.get("present"):
            rows.append({"parameter": pid, "present": False, "note": info.get("note")})
            continue
        value = values.get(pid, info.get("current"))
        if isinstance(value, list):
            bounds = [list(limits.get(f"{pid}_{i}", (None, None))) for i in range(len(value))]
        else:
            bounds = list(limits.get(pid, (None, None)))
        # An element whose bounds have collapsed cannot move — either a structural
        # zero or a closure lock. Reporting it as changeable is what sends an agent
        # into a repair loop it cannot win.
        if isinstance(bounds[0], list):
            immovable = [i for i, b in enumerate(bounds)
                         if b[0] is not None and abs(b[1] - b[0]) <= ZERO]
            movable = len(immovable) < len(bounds)
        else:
            immovable = []
            movable = bounds[0] is None or abs(bounds[1] - bounds[0]) > ZERO

        rows.append({
            "parameter": pid, "present": True, "kind": info.get("kind", "scalar"),
            "value": value, "bounds": bounds,
            "controls": info.get("controls", []),
            "affects_lai": bool(info.get("affects_lai", False)),
            "coupled_with": info.get("coupled_with", []),
            "movable": movable,
            "immovable_indices": immovable,
            "locked_reason": info.get("locked_reason"),
            "meaning": " ".join((info.get("meaning") or "").split()),
        })
    return rows
