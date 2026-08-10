#!/usr/bin/env python
"""The calibration loop a local agent runs, and the tools it is allowed to use.

One iteration is always the same ten steps, whichever target is being calibrated:

     1. inspect the current parameters, their bounds and their meaning
     2. inspect the history — every previous iteration and its outcome
     3. read the diagnostics of the last completed run
     4. attribute the error to a parameter (this is the model's job)
     5. propose a change, with a stated reason and expected effect
     6. pre-flight it (bounds, table shape, closure, blast radius) — free
     7. repair and retry if the pre-flight rejects it
     8. run SIMPLACE
     9. score it with the objective and re-derive the diagnostics
    10. compare against every previous iteration; stop or go again

Steps 1-3 and 6-10 are deterministic Python. The model does steps 4, 5 and 7 —
nothing else. It never edits a file, never chooses a bound, never touches the
frozen parameters, and cannot skip the pre-flight: every path to ``crop.xml``
goes through ``calibrate.py``, which validates and records regardless of who is
calling it.

That is the whole safety argument. The agent is a *proposer*; the guarantees live
one layer down and are the same ones a hand-typed proposal gets.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

OPTIM_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
CALIBRATE = OPTIM_DIR / "calibrate.py"

sys.path.insert(0, str(OPTIM_DIR))
import calib_common as cc  # noqa: E402

from . import llm as _llm  # noqa: E402


class AgentError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# What the model must return
# ---------------------------------------------------------------------------
PROPOSAL_SCHEMA = """{
  "analysis":        "<what the diagnostics show, in 2-4 sentences>",
  "hypothesis":      "<the single mechanism you think is responsible>",
  "parameter":       "<the parameter id you are changing>",
  "parameter_changes": { "<parameter id>": <new value, or {"<index>": <new value>} for one
                                            element of a table, or [full, new, table] > },
  "reason":          "<one line: the evidence that points at THIS parameter>",
  "reasoning":       "<why this direction and this magnitude; what you ruled out and why>",
  "expected_effect": "<what should change in the metrics if the hypothesis is right>",
  "confidence":      <0.0-1.0>,
  "stop":            <true only if no further change is justified>
}"""


@dataclass
class Decision:
    """One parameter proposal, as the model made it."""

    parameters: dict = field(default_factory=dict)
    reason: str = ""
    hypothesis: str = ""
    reasoning: str = ""
    expected_effect: str = ""
    analysis: str = ""
    confidence: float | None = None
    stop: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_reply(cls, reply: dict) -> "Decision":
        changes = reply.get("parameter_changes")
        if changes is None and reply.get("parameter") and "value" in reply:
            # Tolerated shorthand: some models answer with parameter/value instead
            # of the map. Accepting it costs nothing and saves a repair round-trip.
            changes = {reply["parameter"]: reply["value"]}
        if changes is None:
            changes = {}
        if not isinstance(changes, dict):
            raise AgentError(
                f"'parameter_changes' must be an object mapping parameter id -> value, "
                f"got {type(changes).__name__}")
        return cls(
            parameters=changes,
            reason=str(reply.get("reason", "")).strip(),
            hypothesis=str(reply.get("hypothesis", "")).strip(),
            reasoning=str(reply.get("reasoning", "")).strip(),
            expected_effect=str(reply.get("expected_effect", "")).strip(),
            analysis=str(reply.get("analysis", "")).strip(),
            confidence=reply.get("confidence"),
            stop=bool(reply.get("stop", False)),
            raw=reply,
        )


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------
def compact(obj, max_items: int = 14, max_depth: int = 6, _depth: int = 0):
    """Trim a diagnostics dict to something a 16k context can hold.

    Long per-year and per-location tables are the bulk of it and the tail carries
    no signal the head does not, so they are truncated with an explicit marker
    rather than silently — a model that sees ``"... 40 more"`` will not conclude
    the study only had 14 years.
    """
    if _depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: compact(v, max_items, max_depth, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) <= max_items:
            return [compact(v, max_items, max_depth, _depth + 1) for v in obj]
        head = [compact(v, max_items, max_depth, _depth + 1) for v in obj[:max_items]]
        return head + [f"... {len(obj) - max_items} more entries omitted"]
    return obj


def _wrap(text: str, width: int = 96, indent: str = "      ") -> str:
    return textwrap.fill(" ".join((text or "").split()), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def _tidy(value, digits: int = 6):
    """Round away float noise before it reaches the model.

    Relative bounds come out of the arithmetic as ``0.005579999999999999``. A
    small model will echo that back verbatim, and a proposal full of 16-digit
    noise is unreadable in the ledger and in the crop.xml diff.
    """
    if isinstance(value, list):
        return [_tidy(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def render_parameters(rows: list[dict]) -> str:
    out, blocked = [], []
    for row in rows:
        if not row.get("present"):
            blocked.append(f"  {row['parameter']} — not present for this crop")
            continue
        # A parameter with no movable element at all must not be offered. Listing
        # it separately, with the reason, is what stops an agent from spending its
        # whole repair budget proposing changes that cannot pass.
        if row.get("movable") is False:
            reason = row.get("locked_reason") or "its bounds are fixed for this crop"
            blocked.append(f"  {row['parameter']} — CANNOT BE CHANGED: {reason}")
            continue
        flags = []
        if row.get("affects_lai"):
            flags.append("ALSO MOVES LAI")
        if row.get("coupled_with"):
            flags.append("coupled with " + ", ".join(row["coupled_with"]))
        out.append(f"  {row['parameter']}")
        out.append(f"      value   {json.dumps(_tidy(row['value']))}")
        bounds = json.dumps(_tidy(row["bounds"]))
        out.append(f"      bounds  {bounds}" if len(bounds) < 88
                   else "      bounds\n" + _wrap(bounds, indent="        "))
        out.append(f"      controls {', '.join(row.get('controls') or []) or '-'}"
                   + (f"   [{'; '.join(flags)}]" if flags else ""))
        if row.get("immovable_indices"):
            out.append(f"      FIXED elements (do not propose these indices): "
                       f"{row['immovable_indices']}"
                       + (f" — {row['locked_reason']}" if row.get("locked_reason") else ""))
        out.append(_wrap(row.get("meaning", "")))

    if blocked:
        out.append("")
        out.append("  Not available to you for this crop — do not propose them:")
        out.extend(blocked)
    return "\n".join(out)


def render_history(records: list[dict]) -> str:
    if not records:
        return "  (nothing yet — this will be the baseline)"
    out = []
    for r in records:
        objective = ("FAILED" if r.get("objective") is None
                     else f"{r['objective']:.4f}")
        delta = ("" if r.get("delta_vs_best") is None
                 else f"  (delta vs best {r['delta_vs_best']:+.4f})")
        changed = r.get("parameters_changed") or {}
        out.append(f"  iteration {r['iteration']}: objective {objective}{delta}")
        out.append(f"      changed  {json.dumps(changed) if changed else 'nothing (baseline)'}")
        if r.get("previous_values"):
            out.append(f"      from     {json.dumps(r['previous_values'])}")
        if r.get("reason"):
            out.append(_wrap("reason: " + r["reason"]))
        if r.get("expected_effect"):
            out.append(_wrap("expected: " + r["expected_effect"]))
        outcome = ("improved" if r.get("improved") else
                   "new best but below the improvement threshold" if r.get("is_best") else
                   "did NOT improve")
        out.append(f"      outcome  {outcome}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class CalibrationAgent:
    """Base class. A subclass supplies a target, a prompt and its focus."""

    #: calibration target this agent owns
    target: str = ""
    #: filename under agents/prompts/
    prompt_file: str = ""
    #: key under llm.agents in calibration.yaml
    agent_key: str = ""

    def __init__(self, crop: str, backend, config_path: Path | None = None,
                 locations: int | None = None, device: str = "cluster",
                 allow_recalibration: bool = False, verbose: bool = True):
        self.crop = crop
        self.backend = backend
        self.config_path = Path(config_path or cc.DEFAULT_CALIB_CONFIG)
        self.locations = locations
        self.device = device
        self.allow_recalibration = allow_recalibration
        self.verbose = verbose
        self.cfg = cc.load_calib_config(self.config_path)
        self.llm_cfg = self.cfg.get("llm") or {}

    # -- plumbing ----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _calibrate(self, *argv: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        """Every model-side action goes through calibrate.py. There is no other path."""
        command = [sys.executable, str(CALIBRATE), *argv,
                   "--crop", self.crop, "--config", str(self.config_path)]
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    def _scope(self) -> list[str]:
        argv = ["--target", self.target, "--device", self.device]
        if self.locations:
            argv += ["--locations", str(self.locations)]
        return argv

    @staticmethod
    def _last_json_line(stdout: str) -> dict | None:
        """``calibrate.py`` ends a run report with a machine-readable ``JSON {...}`` line."""
        for line in reversed(stdout.splitlines()):
            if line.startswith("JSON "):
                try:
                    return json.loads(line[5:])
                except json.JSONDecodeError:
                    return None
        return None

    # -- tools -------------------------------------------------------------
    def status(self) -> dict:
        proc = self._calibrate("status", *self._scope(), "--json")
        if proc.returncode != 0:
            raise AgentError(f"calibrate.py status failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout[proc.stdout.index("{"):])

    def history(self) -> list[dict]:
        proc = self._calibrate("history", "--target", self.target, "--json")
        if proc.returncode != 0:
            raise AgentError(f"calibrate.py history failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout[proc.stdout.index("["):])

    def preflight(self, parameters: dict) -> dict:
        """Validate a proposal without running the model. Free, so always do it."""
        argv = ["run", *self._scope(), "--dry-run"]
        if parameters:
            argv += ["--params", json.dumps(parameters)]
        if self.allow_recalibration:
            argv.append("--allow-recalibration")
        proc = self._calibrate(*argv)
        try:
            return json.loads(proc.stdout[proc.stdout.index("{"):])
        except (ValueError, json.JSONDecodeError):
            raise AgentError(
                f"pre-flight produced no verdict (rc={proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}")

    def execute(self, decision: Decision) -> dict:
        """Apply, run SIMPLACE, score, record. Blocks for the whole simulation."""
        argv = ["run", *self._scope()]
        if decision.parameters:
            argv += ["--params", json.dumps(decision.parameters)]
        argv += ["--reason", decision.reason or "(no reason given)",
                 "--hypothesis", decision.hypothesis,
                 "--reasoning", self._reasoning_blob(decision),
                 "--expected-effect", decision.expected_effect]
        if self.allow_recalibration:
            argv.append("--allow-recalibration")

        proc = self._calibrate(*argv)
        if self.verbose:
            print(proc.stdout, flush=True)
        result = self._last_json_line(proc.stdout)
        if result is None:
            raise AgentError(
                f"iteration produced no result line (rc={proc.returncode}):\n"
                f"{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
        return result

    def _reasoning_blob(self, decision: Decision) -> str:
        """What goes into the ledger's ``agent_reasoning`` field."""
        parts = [f"[{self.agent_key} agent · {getattr(self.backend, 'model', '?')} "
                 f"via {self.backend.name}]"]
        if decision.analysis:
            parts.append("Analysis: " + decision.analysis)
        if decision.reasoning:
            parts.append("Reasoning: " + decision.reasoning)
        if decision.confidence is not None:
            parts.append(f"Confidence: {decision.confidence}")
        return "\n".join(parts)

    # -- prompt ------------------------------------------------------------
    def system_prompt(self) -> str:
        path = PROMPT_DIR / self.prompt_file
        if not path.exists():
            raise AgentError(f"prompt missing: {path}")
        return path.read_text()

    def latest_diagnostics(self, history: list[dict]) -> tuple[int | None, dict]:
        for record in reversed(history):
            if record.get("status") == "completed" and record.get("diagnostics"):
                return record["iteration"], record["diagnostics"]
        return None, {}

    def context(self, status: dict, history: list[dict]) -> str:
        iteration, diagnostics = self.latest_diagnostics(history)
        best = status.get("best")
        stopping = status.get("stopping") or {}
        max_changed = next((c.get("limit") for c in (status.get("constraints") or [])
                            if c.get("type") == "max_changed"), 3)

        return "\n".join([
            f"# Calibration state — {status['crop']} · {status['target']}",
            "",
            f"objective function     {status.get('objective')}   (lower is better)",
            f"iterations completed   {status.get('n_completed')}",
            f"next iteration         {status.get('next_iteration')}",
            f"best so far            " + (
                "none yet" if not best
                else f"iteration {best['iteration']}, objective {best['objective']:.4f}"),
            f"stopping               {stopping.get('reason')}",
            f"simulation scope       {status.get('project_rows')} rows / "
            f"{status.get('project_locations')} locations, {status.get('subset')}",
            # Listed in full, never truncated. The frozen set is sorted, so any cut
            # would drop the tail — which is where the thermal-time parameters live,
            # and those are exactly the ones a model reaches for when the timing of
            # a feature is wrong. It has to see that they are not available.
            f"frozen (untouchable)   {len(status.get('frozen', {}).get('parameters', []))} "
            f"parameters, listed here so you do not propose them:",
            _wrap(", ".join(status.get("frozen", {}).get("parameters", [])), indent="    "),
            "",
            "## Parameters you may change",
            "",
            render_parameters(status.get("parameters") or []),
            "",
            "## Constraints your proposal must satisfy",
            "",
            json.dumps(status.get("constraints") or [], indent=2),
            "",
            f"You may change at most {max_changed} parameter(s) in one iteration.",
            "",
            "## Diagnostics of the most recent completed iteration"
            + (f" (iteration {iteration})" if iteration is not None else ""),
            "",
            json.dumps(compact(diagnostics), indent=2) if diagnostics
            else "  (none yet)",
            "",
            "## Everything that has already been tried",
            "",
            render_history(history),
            "",
            "## Your reply",
            "",
            "Reply with this JSON object and nothing else:",
            "",
            PROPOSAL_SCHEMA,
        ])

    # -- decision ----------------------------------------------------------
    def decide(self, context: str) -> Decision:
        """Ask the model, then pre-flight; hand any rejection back and retry.

        The repair loop is what makes a small local model usable here. A 14B model
        will get a table index or a step size wrong now and then; the pre-flight
        catches it for free and the exact violation text is a far better correction
        signal than any amount of prompt tuning.
        """
        attempts = int(self.llm_cfg.get("max_repair_attempts", 3))
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": context}]

        last_error = "unknown"
        for attempt in range(attempts + 1):
            try:
                reply = self.backend.json_chat(messages)
                decision = Decision.from_reply(reply)
            except (_llm.LLMError, AgentError) as exc:
                if isinstance(exc, _llm.LLMUnavailable):
                    raise
                last_error = str(exc)
                self.log(f"    reply unusable ({last_error}); asking again "
                         f"[{attempt + 1}/{attempts}]")
                messages += [{"role": "assistant", "content": "(unparseable)"},
                             {"role": "user",
                              "content": f"That reply could not be used: {last_error}\n\n"
                                         f"Reply again with ONLY this JSON object:\n"
                                         f"{PROPOSAL_SCHEMA}"}]
                continue

            if decision.stop or not decision.parameters:
                return decision

            verdict = self.preflight(decision.parameters)
            if verdict.get("valid") and not verdict.get("blocked_by_protection"):
                return decision

            problems = [f"[{v['constraint']}] {v['message']}"
                        for v in verdict.get("violations", [])]
            if verdict.get("blocked_by_protection"):
                problems.append(
                    "this target is protected and the run was not started with "
                    "--allow-recalibration, so its parameters cannot be changed")
            last_error = "; ".join(problems) or "rejected without a reason"
            self.log(f"    pre-flight rejected the proposal [{attempt + 1}/{attempts}]:")
            for problem in problems:
                self.log(f"      {problem}")

            messages += [
                {"role": "assistant", "content": json.dumps(decision.raw)},
                {"role": "user", "content":
                    "That proposal was rejected before the model was run:\n"
                    + "\n".join(f"  - {p}" for p in problems)
                    + "\n\nPropose a change that satisfies every constraint. Keep the same "
                      "hypothesis if you still believe it — adjust the magnitude, the "
                      "index, or pick the parameter that can actually express it. Reply "
                      "with the JSON object only."},
            ]

        raise AgentError(
            f"no usable proposal after {attempts} repair attempt(s). Last problem: {last_error}")

    # -- loop --------------------------------------------------------------
    def iterate(self) -> dict:
        """One full iteration: read state, decide, run, score, record."""
        status = self.status()
        history = self.history()

        if status.get("next_iteration", 0) == 0:
            self.log(f"\n  iteration 0 — baseline (nothing changed), establishing the "
                     f"objective to beat")
            return self.execute(Decision(
                reason="baseline: the current parameter set, unchanged",
                hypothesis="", expected_effect="",
                analysis="Reference run. Every later iteration is compared against this."))

        drift = status.get("frozen", {}).get("drift") or []
        if drift:
            raise AgentError(
                "frozen parameters have drifted; refusing to continue:\n  "
                + "\n  ".join(drift))

        self.log(f"\n  iteration {status['next_iteration']} — asking "
                 f"{getattr(self.backend, 'model', '?')} to decide ...")
        decision = self.decide(self.context(status, history))

        if decision.stop:
            self.log(f"  the agent stopped: {decision.reason or decision.analysis}")
            return {"stopped_by_agent": True, "reason": decision.reason,
                    "analysis": decision.analysis}
        if not decision.parameters:
            raise AgentError("the agent returned no parameter change and did not stop")

        self.log(f"  proposal: {json.dumps(decision.parameters)}")
        self.log(f"  reason:   {decision.reason}")
        return self.execute(decision)

    def loop(self, max_iterations: int | None = None) -> dict:
        """Iterate until the configured stopping criteria are met."""
        status = self.status()
        configured = ((self.cfg["targets"][self.target].get("stopping") or {})
                      .get("max_iterations", 25))
        budget = max_iterations if max_iterations is not None else configured

        self.log("=" * 78)
        self.log(f"  {self.agent_key} agent · {self.crop} · {self.target}")
        self.log(f"  model      {getattr(self.backend, 'model', '?')} ({self.backend.name})")
        self.log(f"  budget     {budget} iteration(s) this session")
        self.log(f"  ledger     {status.get('ledger_dir')}")
        self.log("=" * 78)

        completed, stopped = [], "iteration budget exhausted"
        for _ in range(budget):
            status = self.status()
            if status["stopping"]["stop"] and status["n_completed"] > 0:
                stopped = status["stopping"]["reason"]
                break
            result = self.iterate()
            if result.get("stopped_by_agent"):
                stopped = f"the agent stopped: {result.get('reason')}"
                break
            completed.append(result)
            if result.get("stop"):
                stopped = result.get("stop_reason", "stopping criteria met")
                break

        final = self.status()
        summary = {
            "crop": self.crop, "target": self.target,
            "agent": self.agent_key,
            "model": getattr(self.backend, "model", None),
            "iterations_this_session": len(completed),
            "stopped_because": stopped,
            "best": final.get("best"),
            "n_completed": final.get("n_completed"),
            "ledger_dir": final.get("ledger_dir"),
            "frozen_intact": final.get("frozen", {}).get("intact"),
        }
        self.log("\n" + "=" * 78)
        self.log(f"  done — {summary['iterations_this_session']} iteration(s) this session")
        self.log(f"  stopped because: {stopped}")
        if summary["best"]:
            self.log(f"  best: iteration {summary['best']['iteration']}  "
                     f"objective {summary['best']['objective']:.4f}")
        self.log(f"  frozen phenology intact: {summary['frozen_intact']}")
        self.log("=" * 78)
        return summary
