#!/usr/bin/env python
"""Run the crop-model calibration with local LLM agents over Ollama.

Everything stays on this machine. The agent asks a local model which parameter to
change, the proposal goes through the same validation, freeze guard and ledger as
a hand-written one, SIMPLACE runs on the cluster, and the result comes back as
the next iteration's evidence. There is no sampler and no external service.

    python optimization/agentic.py check
    python optimization/agentic.py validate-phenology --crop winter_wheat
    python optimization/agentic.py run   --crop winter_wheat --target lai --iterations 8
    python optimization/agentic.py step  --crop winter_wheat --target lai
    python optimization/agentic.py propose --crop winter_wheat --target lai
    python optimization/agentic.py review  --crop winter_wheat --target lai

The calibration order is phenology (already done — validate only), then LAI, then
``calibrate.py handoff``, then yield. ``run`` respects the stopping criteria in
calibration.yaml; ``--iterations`` caps the session on top of them.

Nothing here can write to ``simplace/<crop>/data/crop/crop.xml``. Promoting a
finished calibration into production stays a deliberate human step:
``calibrate.py promote --crop <crop> --target <target> --yes``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OPTIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIM_DIR))

import agents as ag  # noqa: E402
import calib_common as cc  # noqa: E402


def _backend(args, agent_key: str):
    cfg = cc.load_calib_config(Path(args.config))
    return ag.build_backend(cfg.get("llm") or {}, agent_key,
                            model=getattr(args, "model", None))


def _agent(args, target: str | None = None):
    target = target or args.target
    backend = _backend(args, target)
    kwargs = dict(crop=args.crop, backend=backend, config_path=Path(args.config),
                  locations=getattr(args, "locations", None),
                  device=getattr(args, "device", "cluster"))
    if target == "phenology":
        kwargs["mode"] = getattr(args, "mode", "validate")
    agent = ag.AGENTS[target](**kwargs)

    ok, detail = backend.available()
    if not ok:
        raise SystemExit(
            f"\n  local model not usable: {detail}\n\n"
            f"  Check the setup with:  python optimization/agentic.py check\n")
    return agent


# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    """Is a local model server reachable, and are the configured models pulled?"""
    cfg = cc.load_calib_config(Path(args.config))
    llm_cfg = cfg.get("llm") or {}
    print(f"\n  provider   {llm_cfg.get('provider', 'ollama')}")

    keys = sorted((llm_cfg.get("agents") or {}))
    installed, problem, probed, missing = [], None, False, []
    for key in keys:
        backend = ag.build_backend(llm_cfg, key)
        if not probed:
            probed = True
            print(f"  host       {getattr(backend, 'host', '-')}")
            try:
                installed = backend.models()
                print(f"  models     {', '.join(installed) if installed else '(none pulled)'}")
            except ag.LLMError as exc:
                problem = str(exc)
                print(f"  models     UNREACHABLE — {problem}")
        ok, detail = backend.available()
        print(f"    {key:<10s} {getattr(backend, 'model', '-'):<28s} "
              f"{'OK' if ok else 'NOT USABLE'}")
        if not ok:
            missing.append(key)
            if installed:
                print(f"    {'':<10s} {detail}")

    if problem:
        print("\n  No local model server answered. To use the local agents:")
        print("    1. install Ollama         https://ollama.com/download")
        print("    2. start it               ollama serve")
        print("    3. pull a model           ollama pull qwen2.5:14b-instruct")
        print("    4. re-run                 python optimization/agentic.py check")
        print("\n  If Ollama runs on another host, point OLLAMA_HOST at it or set")
        print("  llm.host in optimization/calibration.yaml.\n")
        return 1

    if missing:
        print(f"\n  {len(missing)} agent(s) have no usable model: {', '.join(missing)}")
        print("  Pull the configured models, or point llm.agents.<agent>.model at one")
        print("  of the models listed above.\n")
        return 1
    print("\n  ready.\n")
    return 0


def cmd_run(args) -> int:
    agent = _agent(args)
    summary = agent.loop(max_iterations=args.iterations)
    print("\nJSON " + json.dumps(summary, default=str))
    return 0


def cmd_step(args) -> int:
    """Exactly one iteration, then stop. The unit to use when watching it work."""
    agent = _agent(args)
    result = agent.iterate()
    print("\nJSON " + json.dumps(result, default=str))
    return 0


def cmd_propose(args) -> int:
    """Ask for a decision and pre-flight it, but do not run the model.

    The cheap way to see what a model would do — and to compare two models on the
    same calibration state — without spending a SLURM run on it.
    """
    agent = _agent(args)
    status = agent.status()
    history = agent.history()
    if status.get("next_iteration", 0) == 0:
        print("\n  no completed iteration yet — there is nothing to reason from.")
        print(f"  Run the baseline first:  python optimization/agentic.py step "
              f"--crop {args.crop} --target {args.target}\n")
        return 1

    decision = agent.decide(agent.context(status, history))
    verdict = ({} if not decision.parameters
               else agent.preflight(decision.parameters))
    payload = {
        "crop": args.crop, "target": args.target,
        "model": getattr(agent.backend, "model", None),
        "would_change": decision.parameters,
        "reason": decision.reason,
        "hypothesis": decision.hypothesis,
        "analysis": decision.analysis,
        "reasoning": decision.reasoning,
        "expected_effect": decision.expected_effect,
        "confidence": decision.confidence,
        "stop": decision.stop,
        "preflight": verdict,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_review(args) -> int:
    backend = _backend(args, "analyst")
    ok, detail = backend.available()
    if not ok:
        raise SystemExit(f"\n  local model not usable: {detail}\n")
    analyst = ag.AnalystAgent(args.crop, backend, target=args.target,
                              config_path=Path(args.config),
                              locations=getattr(args, "locations", None))
    print(json.dumps(analyst.review(), indent=2, default=str))
    return 0


def cmd_validate_phenology(args) -> int:
    """Score the already-optimized phenology. Changes nothing, needs no model."""
    agent = ag.PhenologyAgent(
        crop=args.crop, backend=ag.MockBackend(replies=[]), mode="validate",
        config_path=Path(args.config), locations=getattr(args, "locations", None),
        device=args.device)
    summary = agent.loop()
    print("\nJSON " + json.dumps(summary, default=str))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def shared(parser, target: bool = True, scope: bool = True):
        parser.add_argument("--crop", default="winter_wheat")
        parser.add_argument("--config", default=str(cc.DEFAULT_CALIB_CONFIG))
        if target:
            parser.add_argument("--target", default="lai",
                                choices=["phenology", "lai", "yield"])
            parser.add_argument("--model", help="override the configured Ollama model")
        if scope:
            parser.add_argument("--device", default="cluster", choices=["cluster", "local"])
            parser.add_argument("--locations", type=int,
                                help="override the subset size (smaller = faster iterations)")
        return parser

    p = sub.add_parser("check", help="is a local model server reachable and ready?")
    p.add_argument("--config", default=str(cc.DEFAULT_CALIB_CONFIG))
    p.set_defaults(func=cmd_check)

    p = shared(sub.add_parser("run", help="iterate until the stopping criteria are met"))
    p.add_argument("--iterations", type=int,
                   help="cap this session (default: the target's max_iterations)")
    p.add_argument("--mode", default="validate", choices=["validate", "recalibrate"],
                   help="phenology only: recalibrate permits parameter writes")
    p.set_defaults(func=cmd_run)

    p = shared(sub.add_parser("step", help="one iteration, then stop"))
    p.add_argument("--mode", default="validate", choices=["validate", "recalibrate"])
    p.set_defaults(func=cmd_step)

    p = shared(sub.add_parser("propose", help="get a decision without running the model"))
    p.add_argument("--mode", default="validate", choices=["validate", "recalibrate"])
    p.set_defaults(func=cmd_propose)

    p = shared(sub.add_parser("review", help="read-only analysis of the calibration so far"))
    p.set_defaults(func=cmd_review)

    p = shared(sub.add_parser(
        "validate-phenology",
        help="score the optimized phenology without changing it (no model needed)"),
        target=False)
    p.set_defaults(func=cmd_validate_phenology)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
