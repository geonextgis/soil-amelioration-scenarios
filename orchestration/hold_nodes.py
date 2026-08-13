#!/usr/bin/env python
"""Hold a set of nodes across many runs instead of re-queueing for each one.

Both drivers in this repo submit their own SLURM jobs per run and release the
nodes when it ends, so the next run goes back to the end of the queue. For the
scenario matrix that is solved by the campaign script (one sbatch, every
experiment inside it). Calibration cannot work that way: an agent picks the next
parameter *between* runs, on the login node, so the allocation has to outlive the
process that uses it.

This holds one with ``salloc --no-shell`` — the allocation exists, nothing runs
in it, and it stays until released or its walltime expires. The job id is
recorded in ``.simplace_allocation.json`` at the repo root, which
``simplace_runner_cluster.py`` reads on its own: every subsequent run attaches
its work to those nodes with ``srun --jobid=`` instead of submitting. Nothing
else has to be told about it — not calibrate.py, not the agents, not a config
file — and each run still costs exactly one queue wait less than it did.

    python orchestration/hold_nodes.py hold --nodes 40 --walltime 08:00:00
    python optimization/calibrate.py run --crop maize --target growth   # attaches
    python optimization/calibrate.py run --crop maize --target growth   # attaches
    python orchestration/hold_nodes.py status
    python orchestration/hold_nodes.py release

The nodes are idle between runs and still charged to you, so hold them for a
working session, not overnight. `status` prints what is left.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOCATION_FILE = ".simplace_allocation.json"


def session_path() -> Path:
    return REPO_ROOT / ALLOCATION_FILE


def read_session() -> dict | None:
    p = session_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def job_state(job_id: str) -> str:
    r = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""


def job_field(job_id: str, name: str) -> str:
    r = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", name],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""


def slurm_defaults() -> dict:
    """SLURM settings from the calibration config, falling back to the scenario one.

    Calibration is the workflow that needs a held allocation most, so its sizing
    (40 nodes, 8 tasks) is the natural default for `hold`.
    """
    for cfg_path in (REPO_ROOT / "optimization" / "config.yaml",
                     REPO_ROOT / "orchestration" / "experiments.yaml"):
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            if cfg.get("slurm"):
                return cfg["slurm"]
    return {}


def cmd_hold(args) -> int:
    existing = read_session()
    if existing and job_state(existing.get("jobid", "")) == "RUNNING":
        print(f"already holding allocation {existing['jobid']} "
              f"({job_field(existing['jobid'], '%D')} node(s), "
              f"{job_field(existing['jobid'], '%L')} left).")
        print("Release it first, or use --jobid on the runner to target another.")
        return 1

    s = slurm_defaults()
    nodes = args.nodes or int(s.get("num_nodes", 40))
    tasks = args.tasks_per_node or int(s.get("num_tasks_per_node", 8))
    cpus_per_node = int(s.get("cpus_per_node", 80))
    cpus_per_task = max(1, cpus_per_node // tasks)
    partition = args.partition or s.get("partition", "compute")

    cmd = ["salloc", "--no-shell",
           f"--nodes={nodes}",
           f"--ntasks-per-node={tasks}",
           f"--cpus-per-task={cpus_per_task}",
           f"--partition={partition}",
           f"--time={args.walltime}",
           f"--job-name={args.name}"]
    if args.mem_per_cpu:
        cmd.append(f"--mem-per-cpu={args.mem_per_cpu}")

    print(" ".join(cmd))
    print(f"waiting for {nodes} node(s) on {partition} ...", flush=True)
    # salloc reports the granted allocation on stderr and returns once it exists.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out.strip())
    m = re.search(r"Granted job allocation (\d+)", out)
    if proc.returncode != 0 or not m:
        print("could not hold an allocation.", file=sys.stderr)
        return 1

    job_id = m.group(1)
    session = {
        "jobid": job_id,
        "nodes": nodes,
        "tasks_per_node": tasks,
        "cpus_per_task": cpus_per_task,
        "partition": partition,
        "walltime": args.walltime,
        "held_at": datetime.now().isoformat(timespec="seconds"),
        "held_by": os.environ.get("USER", "?"),
    }
    session_path().write_text(json.dumps(session, indent=2) + "\n")

    print(f"\nholding allocation {job_id}: {nodes} node(s) x {tasks} task(s) x "
          f"{cpus_per_task} cpu(s) = {nodes * tasks} concurrent SIMPLACE steps")
    print(f"  recorded in:  {session_path()}")
    print(f"  runs attach automatically; nothing else needs configuring")
    print(f"  release with: python orchestration/hold_nodes.py release")
    return 0


def cmd_status(args) -> int:
    session = read_session()
    if not session:
        print("no allocation held (no .simplace_allocation.json). "
              "Runs will submit their own jobs.")
        return 0
    job_id = str(session.get("jobid", ""))
    state = job_state(job_id)
    if state != "RUNNING":
        print(f"recorded allocation {job_id} is {state or 'gone'} — runs will "
              f"submit their own jobs again.")
        print(f"clear the stale record with: python orchestration/hold_nodes.py release")
        return 0
    print(f"allocation {job_id}: {state}, {job_field(job_id, '%D')} node(s), "
          f"{job_field(job_id, '%L')} remaining (of {session.get('walltime')})")
    print(f"  nodes:  {job_field(job_id, '%N')}")
    print(f"  slots:  {session.get('nodes')} x {session.get('tasks_per_node')} = "
          f"{int(session.get('nodes', 0)) * int(session.get('tasks_per_node', 1))}")
    print(f"  held:   {session.get('held_at')} by {session.get('held_by')}")
    return 0


def cmd_release(args) -> int:
    session = read_session()
    if not session:
        print("nothing recorded to release.")
        return 0
    job_id = str(session.get("jobid", ""))
    state = job_state(job_id)
    if state == "RUNNING":
        r = subprocess.run(["scancel", job_id], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"scancel failed: {r.stderr.strip()}", file=sys.stderr)
            return 1
        print(f"released allocation {job_id}.")
    else:
        print(f"allocation {job_id} was already {state or 'gone'}.")
    session_path().unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hold", help="hold nodes until released")
    h.add_argument("--nodes", type=int, help="default: slurm.num_nodes from optimization/config.yaml")
    h.add_argument("--tasks-per-node", type=int, help="default: slurm.num_tasks_per_node")
    h.add_argument("--partition", help="default: slurm.partition")
    h.add_argument("--walltime", default="08:00:00",
                   help="how long to hold the nodes (default 08:00:00)")
    h.add_argument("--mem-per-cpu", default="1100M",
                   help="per-CPU memory; without it the first step on a node "
                        "claims all of it and its siblings stall (default 1100M)")
    h.add_argument("--name", default="simplace_hold", help="SLURM job name")
    h.set_defaults(func=cmd_hold)

    s = sub.add_parser("status", help="show the held allocation and time left")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("release", help="cancel the allocation and clear the record")
    r.set_defaults(func=cmd_release)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
