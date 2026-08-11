#!/usr/bin/env python
"""Scoring one iteration: every view of a stage, then the combined objective.

A stage owns one or more *views* (one model run scored against one observation
set). This module is the single place that knows how to turn a finished run into
``(loss, metrics, diagnostics, figures)`` for each of them, and how the stage's
weights fold those into the objective the ledger records.

    result = evaluation.evaluate(spec, iter_dir)
    result.objective        # what the calibration minimises
    result.views["lai"]     # the LAI loss, metrics, diagnostics and figures

Adding a view means adding a loss pair in ``objectives.py`` and one entry in
``EVALUATORS`` here. Nothing else in the calibration layer changes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_diagnostics as cd  # noqa: E402
import objectives  # noqa: E402


@dataclass
class ViewResult:
    view: str
    loss: float
    metrics: dict
    diagnostics: dict
    figures: list[str] = field(default_factory=list)


@dataclass
class Result:
    objective: float
    breakdown: dict                       # per view: loss, scale, weight, contribution
    views: dict[str, ViewResult]

    @property
    def metrics(self) -> dict:
        """Flat ``{"<view> · <metric>": value}`` for the ledger and the report."""
        out = {}
        for view, res in self.views.items():
            for key, value in (res.metrics or {}).items():
                out[f"{view} · {key}"] = value
        return out

    @property
    def diagnostics(self) -> dict:
        return {"objective": self.breakdown,
                **{view: res.diagnostics for view, res in self.views.items()}}

    @property
    def figures(self) -> list[str]:
        return [path for res in self.views.values() for path in res.figures]


# ---------------------------------------------------------------------------
# One view
# ---------------------------------------------------------------------------
def _phenology(run, out_dir: Path, title: str, reference=None):
    process_result, loss_fn = objectives.for_view("phenology")
    pairs = process_result(run)
    loss, metrics = loss_fn(pairs)
    points = cd.load_points(run)
    diagnostics = cd.phenology_diagnostics(pairs, points)
    pairs.to_csv(out_dir / "pairs.csv.gz", index=False, compression="gzip")
    figures = cd.phenology_plots(pairs, out_dir / "diagnostics", title, points=points)
    return loss, metrics, diagnostics, figures


def _lai(run, out_dir: Path, title: str, reference=None):
    process_result, loss_fn = objectives.for_view("lai")
    pairs = process_result(run)
    loss, metrics = loss_fn(pairs)
    diagnostics, seasons = cd.lai_diagnostics(pairs, objectives.DVS_BINS)
    pairs.to_csv(out_dir / "pairs.csv.gz", index=False, compression="gzip")
    if not seasons.empty:
        seasons.to_csv(out_dir / "season_shape.csv", index=False)
    figures = cd.lai_plots(pairs, seasons, objectives.DVS_BINS, out_dir / "diagnostics",
                           title, sim_daily=cd.load_lai_daily(run))
    return loss, metrics, diagnostics, figures


def _yield(run, out_dir: Path, title: str, reference: dict | None = None):
    """``reference`` is the crop's agronomic HI / biomass envelope, for attribution."""
    process_result, loss_fn = objectives.for_view("yield")
    loss, metrics = loss_fn(process_result(run))
    # The attribution frame carries the state variables behind the yield
    # (AGBiomass, maxLAI, TRANRF, NNI), so it is what gets stored and plotted.
    frame = cd.yield_attribution_frame(run)
    diagnostics = cd.yield_diagnostics(frame, reference)
    frame.to_csv(out_dir / "pairs.csv.gz", index=False, compression="gzip")
    figures = cd.yield_plots(frame, out_dir / "diagnostics", title)
    return loss, metrics, diagnostics, figures


EVALUATORS = {"phenology": _phenology, "lai": _lai, "yield": _yield}


def evaluate_view(spec, view: str, out_dir: Path) -> ViewResult:
    """Score one view of ``spec`` from the outputs currently in its ``out/``."""
    if view not in EVALUATORS:
        raise SystemExit(f"no evaluator for view {view!r}; have {sorted(EVALUATORS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss, metrics, diagnostics, figures = EVALUATORS[view](
        spec.runs[view], out_dir, f"{spec.crop} · {view}",
        (spec.cfg.get("crop_reference") or {}).get(spec.crop))
    return ViewResult(view=view, loss=float(loss), metrics=metrics,
                      diagnostics=diagnostics, figures=figures)


# ---------------------------------------------------------------------------
# The whole stage
# ---------------------------------------------------------------------------
def evaluate(spec, iter_dir: Path) -> Result:
    """Score every view of the stage and combine them into the objective.

    Each view writes its own artefacts under ``<iter_dir>/<view>/`` — the joined
    observed/simulated pairs, the figures — so a joint iteration keeps the two
    error pictures separable after the fact.
    """
    iter_dir = Path(iter_dir)
    views = {}
    for view in spec.views:
        views[view] = evaluate_view(spec, view, iter_dir / view)

    objective, breakdown = objectives.combine(
        spec.components, {view: res.loss for view, res in views.items()})
    return Result(objective=objective, breakdown=breakdown, views=views)
