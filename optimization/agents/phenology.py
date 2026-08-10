#!/usr/bin/env python
"""Phenology agent — validation by default, recalibration only when asked.

Phenology is already optimized for all five crops in this repository and the LAI
and yield stages treat those values as ground truth. So this agent has two modes:

  ``validate``     (default) run the optimized set, score it, report where the
                   flowering / duration residuals are structured. Proposes
                   nothing and cannot write a parameter — ``calibrate.py``
                   refuses without ``--allow-recalibration``.
  ``recalibrate``  the full loop, for a new crop or a changed observation set.
                   The pre-change values are preserved in
                   ``optimized_baseline.json`` and restorable with
                   ``calibrate.py restore-optimized``.

Keeping it available is the point: when someone later adds a sixth crop, or the
DWD phenology series is extended, the machinery is here and the optimized crops
are not at risk in the meantime.
"""
from __future__ import annotations

from .base import CalibrationAgent, Decision


class PhenologyAgent(CalibrationAgent):
    target = "phenology"
    agent_key = "phenology"
    prompt_file = "phenology.md"

    def __init__(self, *args, mode: str = "validate", **kwargs):
        if mode not in ("validate", "recalibrate"):
            raise ValueError(f"mode must be 'validate' or 'recalibrate', got {mode!r}")
        self.mode = mode
        # The two are the same switch seen from either side: recalibration is
        # exactly the permission to write.
        kwargs["allow_recalibration"] = (mode == "recalibrate")
        super().__init__(*args, **kwargs)

    def iterate(self) -> dict:
        if self.mode == "validate":
            status = self.status()
            if status.get("next_iteration", 0) > 0:
                return {"stopped_by_agent": True,
                        "reason": "validation only — the optimized set has already been scored; "
                                  "re-run with --mode recalibrate to change anything"}
            self.log("\n  validating the optimized phenology (nothing will be changed) ...")
            return self.execute(Decision(
                reason="validation of the already-optimized phenology parameters",
                analysis="No parameter change. Scores the optimized set against the "
                         "observations and reports where the residual is structured."))
        return super().iterate()

    def loop(self, max_iterations: int | None = None) -> dict:
        if self.mode == "validate":
            # One run, no decision, no model call.
            self.log("=" * 78)
            self.log(f"  phenology agent · {self.crop} · VALIDATION MODE")
            self.log(f"  The optimized parameters will be scored, not changed.")
            self.log("=" * 78)
            result = self.iterate()
            final = self.status()
            return {
                "crop": self.crop, "target": "phenology", "agent": "phenology",
                "mode": "validate", "result": result,
                "best": final.get("best"), "n_completed": final.get("n_completed"),
                "ledger_dir": final.get("ledger_dir"),
                "stopped_because": "validation run complete",
                "iterations_this_session": 0 if result.get("stopped_by_agent") else 1,
            }
        summary = super().loop(max_iterations)
        summary["mode"] = "recalibrate"
        return summary
