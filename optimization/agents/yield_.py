#!/usr/bin/env python
"""Yield calibration agent — district yields, after LAI is calibrated.

Stage 2. Starts from the LAI-calibrated ``crop.xml`` (via ``calibrate.py
handoff``) with the phenology *and* the LAI parameter set frozen, so it cannot
buy yield by undoing the canopy calibration. ``StemsPartitioningTableFraction``
is the deliberate exception — the partitioning closure needs a counterweight when
storage-organ allocation moves.

The module is named ``yield_`` because ``yield`` is a Python keyword.
"""
from __future__ import annotations

import json

from .base import AgentError, CalibrationAgent


class YieldAgent(CalibrationAgent):
    target = "yield"
    agent_key = "yield"
    prompt_file = "yield.md"

    def verify_lai(self) -> dict:
        """Has the yield calibration degraded LAI? Runs the LAI objective as it stands.

        Several yield parameters (RUE, KDIF, partitioning) legitimately move the
        canopy, so this is a regression guard, not a constraint — it is checked
        after the fact rather than blocking a proposal.
        """
        proc = self._calibrate("verify-lai", "--device", self.device)
        if self.verbose:
            print(proc.stdout, flush=True)
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("JSON "):
                return json.loads(line[5:])
        raise AgentError(f"verify-lai produced no verdict:\n{proc.stdout}\n{proc.stderr}")

    def loop(self, max_iterations: int | None = None) -> dict:
        summary = super().loop(max_iterations)
        if summary["iterations_this_session"]:
            self.log("\n  checking the LAI calibration for regression ...")
            try:
                summary["lai_regression"] = self.verify_lai()
            except AgentError as exc:
                # A failed guard must not erase the calibration work that preceded
                # it; report and carry on.
                summary["lai_regression"] = {"error": str(exc)}
                self.log(f"  LAI regression check could not run: {exc}")
        return summary
