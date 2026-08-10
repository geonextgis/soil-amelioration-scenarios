#!/usr/bin/env python
"""Diagnostics agent — reads a calibration and explains it. Proposes nothing.

The calibrators decide one parameter at a time under a strict output schema,
which is the right shape for a decision and the wrong shape for a review. This
agent answers the other question: given everything in the ledger, is the
calibration converging, what is the error pattern now, and is anything about the
history suspicious (thrashing, a parameter fighting another, an improvement that
came from the wrong mechanism)?

It has no ``execute`` path — the only tools it uses are ``status`` and
``history``, both read-only.
"""
from __future__ import annotations

import json

from .base import CalibrationAgent, compact, render_history, render_parameters

REVIEW_SCHEMA = """{
  "state":            "<converging | stalled | diverging | not started>",
  "error_pattern":    "<what is wrong with the simulation now, in plain terms>",
  "attribution":      "<which parameter(s) the evidence implicates, and why>",
  "history_review":   "<what the sequence of iterations shows; name any thrashing or
                        any improvement that looks like it came from the wrong mechanism>",
  "recommendation":   "<what you would try next, or why you would stop>",
  "confidence":       <0.0-1.0>,
  "concerns":         ["<anything that should be checked before trusting this calibration>"]
}"""


class AnalystAgent(CalibrationAgent):
    target = "lai"          # overridden per invocation
    agent_key = "analyst"
    prompt_file = "analyst.md"

    def __init__(self, crop: str, backend, target: str = "lai", **kwargs):
        self.target = target
        super().__init__(crop, backend, **kwargs)

    def context(self, status: dict, history: list[dict]) -> str:
        iteration, diagnostics = self.latest_diagnostics(history)
        return "\n".join([
            f"# Calibration review — {status['crop']} · {status['target']}",
            "",
            f"objective            {status.get('objective')} (lower is better)",
            f"iterations completed {status.get('n_completed')}",
            f"best                 {json.dumps(status.get('best'))}",
            f"stopping             {json.dumps(status.get('stopping'))}",
            f"frozen intact        {status.get('frozen', {}).get('intact')}",
            "",
            "## Current parameters",
            "",
            render_parameters(status.get("parameters") or []),
            "",
            "## Diagnostics of the most recent completed iteration"
            + (f" (iteration {iteration})" if iteration is not None else ""),
            "",
            json.dumps(compact(diagnostics, max_items=20), indent=2) if diagnostics
            else "  (none yet)",
            "",
            "## Full history",
            "",
            render_history(history),
            "",
            "## Your reply",
            "",
            "Reply with this JSON object and nothing else:",
            "",
            REVIEW_SCHEMA,
        ])

    def review(self) -> dict:
        status = self.status()
        history = self.history()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.context(status, history)}]
        review = self.backend.json_chat(messages)
        review["_scope"] = {"crop": self.crop, "target": self.target,
                            "n_completed": status.get("n_completed"),
                            "best": status.get("best")}
        return review
