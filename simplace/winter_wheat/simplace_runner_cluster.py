"""Drive one SIMPLACE experiment on the cluster.

Two execution modes, selected by ``--mode`` (default ``auto``):

``sbatch``
    The original behaviour. Splits the experiment's project CSV into
    ``num_nodes`` line ranges and submits one SLURM job per range, then polls
    ``squeue`` until they finish. Every experiment therefore acquires and
    releases its own nodes: the next experiment goes back to the end of the
    queue and waits for resources all over again.

``alloc``
    Runs against an allocation that already exists. Nothing is submitted; the
    work is launched as ``srun`` job steps, so the nodes stay held. Two ways in:

    * *Inside* the allocation — the campaign job produced by
      ``orchestration/generate.py`` calls this once per experiment in sequence,
      and the allocation is released only after the last one finishes.
    * *Attached to* an allocation held by someone else, via ``--jobid`` (or the
      session written by ``orchestration/hold_nodes.py hold``). Steps are placed
      with ``srun --jobid=``, so a driver that lives on the login node — the
      calibration loop, where an agent picks parameters between runs — can reuse
      one set of nodes across many iterations instead of re-queueing each time.

``auto``
    ``alloc`` when running inside a SLURM allocation (``SLURM_JOB_ID`` set) or
    when a held allocation is configured and still running; ``sbatch``
    otherwise. Keeps the login-node entrypoint unchanged when no nodes are held.

Exit codes: ``0`` success, ``1`` at least one chunk failed, ``3`` not enough
walltime left in the allocation to start this experiment (the campaign script
stops cleanly on this so the run can be resumed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

# Not enough allocation time left to start this experiment. Distinct from a
# real failure so the campaign driver can stop cleanly instead of reporting an
# error for every remaining experiment.
EXIT_OUT_OF_TIME = 3

# Cores per compute node, used only to derive cpus-per-task when the
# allocation does not state it. Overridable via `cluster.cpus_per_node`.
DEFAULT_CPUS_PER_NODE = 80

# Where `orchestration/hold_nodes.py hold` records the allocation it is holding.
# A file, not just an env var: the calibration loop runs one iteration per shell
# (an agent decides the next parameter in between), so nothing survives in the
# environment from one run to the next.
ALLOCATION_FILE = ".simplace_allocation.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def split_range(start: int, end: int, num_splits: int):
    """
    Splits the range [start, end] into `num_splits` nearly equal subranges.

    Args:
        start (int): Start of the range (inclusive).
        end (int): End of the range (inclusive).
        num_splits (int): Number of subranges to split into.

    Returns:
        List[Tuple[int, int]]: List of (start, end) tuples for each subrange.
    """
    indices = np.linspace(start, end + 1, num_splits + 1, dtype=int)
    return [(indices[i], indices[i + 1] - 1) for i in range(num_splits) if indices[i] <= indices[i + 1] - 1]


def read_location_starts(input_csv: str, loc_col: str = "vLocationID", sep: str = ";"):
    """1-based data-line numbers where vLocationID changes, plus total data lines.

    SIMPLACE writes output per location (``<vLocationID>_yearly.csv``) into one shared
    dir, truncating on open. If a location's rows are split across two SIMPLACE
    invocations they clobber each other, so work must be split ONLY on location
    boundaries. Assumes the CSV is sorted so each location is one contiguous block.
    """
    starts, total, prev = [], 0, None
    with open(input_csv) as f:
        header = f.readline().rstrip("\n").split(sep)
        ci = header.index(loc_col)
        for line in f:
            total += 1
            loc = line.split(sep)[ci]
            if loc != prev:
                starts.append(total)
                prev = loc
    return starts, total


def split_on_locations(loc_starts, end_line: int, num_splits: int):
    """Split into <=num_splits contiguous line ranges, never cutting a location block.

    ``loc_starts`` are the 1-based line numbers where each location's block begins;
    ``end_line`` is the last data line of the region being split. Locations are spread
    as evenly as possible across the splits (rows-per-location is ~constant, so this
    also balances rows).
    """
    nloc = len(loc_starts)
    if nloc == 0:
        return []
    idx = np.linspace(0, nloc, num_splits + 1, dtype=int)
    ranges = []
    for k in range(num_splits):
        a, b = idx[k], idx[k + 1]
        if a >= b:
            continue
        s = int(loc_starts[a])
        e = int(loc_starts[b]) - 1 if b < nloc else int(end_line)
        ranges.append((s, e))
    return ranges


def load_cluster_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['cluster']


def prepare_dirs(config: dict) -> dict:
    """Resolve the paths both modes need and create them."""
    work_dir = config['work_dir']
    paths = {
        "work": work_dir,
        "out": os.path.join(work_dir, "out", config['exp_name']),
        "out_zip": os.path.join(work_dir, "out_zip"),
        "project": os.path.join(work_dir, "projects"),
        "log": os.path.join(work_dir, "log"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def singularity_command(config: dict, paths: dict, first: int, last: int):
    """The container invocation for one contiguous line range of the project CSV."""
    bind_paths = ",".join([
        f"{paths['work']}:/simplace/SIMPLACE_WORK",
        f"{paths['out']}:/outputs",
        f"{config['mount_data']}:/data",
        f"{paths['project']}:/projects",
        f"{paths['log']}:/simplace/log",
    ])
    execdir = "/simplace"
    simplace_workdir = "/simplace/SIMPLACE_WORK"
    cmd = [
        "singularity", "run", "-B", bind_paths, config['singularity_image'],
        f"{execdir}/simplace", "run",
        f"-s={simplace_workdir}/{config['solution']}",
        f"-p={simplace_workdir}/{config['project']}",
        "-t=CLUSTER",
        f"-l={first}-{last}",
    ]
    if str(config.get('debug', 'false')).lower() != "true":
        cmd.append("-loglevel=ERROR")
    return cmd


def work_ranges(config: dict, num_splits: int):
    """Split the configured region of the project CSV into <=num_splits ranges."""
    loc_starts, total_lines = read_location_starts(config['input_csv'])
    start = config.get("start_line", 1)
    end = config.get("end_line", total_lines)
    sel = [s for s in loc_starts if start <= s <= end]
    return split_on_locations(sel, end, num_splits), loc_starts


def completion_marker(config: dict) -> Path:
    """Marker written once every chunk of an experiment has exited 0.

    Only alloc mode writes it — it is the only mode that sees each chunk's exit
    code. It is what lets a campaign that ran out of walltime be resubmitted
    without redoing the experiments it already finished.
    """
    return Path(config['work_dir']) / f".completed_{config['exp_name']}"


# --- mode: sbatch (one allocation per experiment) ---------------------------

def run_simplace(config_path: str) -> int:
    config = load_cluster_config(config_path)
    paths = prepare_dirs(config)

    NUM_JOBS = config['num_nodes']
    NUM_TASKS = config['num_tasks_per_node']
    cpus_per_node = int(config.get('cpus_per_node', DEFAULT_CPUS_PER_NODE))

    # === Split work on LOCATION boundaries ===
    # Output is per-location and written into one shared dir, so a location must be
    # handled by exactly one SIMPLACE invocation (else its file gets overwritten).
    job_ranges, loc_starts = work_ranges(config, NUM_JOBS)
    submitted_jobs = []

    for i, j in job_ranges:
        job_name = f"SIM_{i}_{j}"
        sel_job = [s for s in loc_starts if i <= s <= j]
        subranges = split_on_locations(sel_job, j, NUM_TASKS)
        num_procs_per_task = int(cpus_per_node / NUM_TASKS)
        srun_commands = []
        for i_sub, j_sub in subranges:
            cmd_str = " ".join(singularity_command(config, paths, i_sub, j_sub))
            srun_commands.append(f"srun -n1 -c {num_procs_per_task} {cmd_str} &")

        # Combine all into a Slurm batch script
        srunc = '\n'.join(srun_commands)
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --partition={config['partition']}
#SBATCH --time={config['walltime']}
#SBATCH --output={paths['log']}/simplace-%j.out

{srunc}
wait
"""
        slurm_file = f"{paths['work']}/.tmp_simplace_job_{i}_{j}.sh"
        with open(slurm_file, "w") as f:
            f.write(slurm_script)

        result = subprocess.run(["sbatch", slurm_file], capture_output=True, text=True)
        os.remove(slurm_file)
        if result.returncode == 0:
            slurm_job_id = result.stdout.split()[-1]
            print(f"Submitted SLURM job {slurm_job_id} for range {i}-{j}")
            submitted_jobs.append(slurm_job_id)
        else:
            print(f"Failed to submit SLURM job for range {i}-{j}")
            print(result.stderr)

    # === Monitor job completion ===
    print(f"Waiting for {len(submitted_jobs)} job(s) to finish...")

    def is_job_active(job_id):
        result = subprocess.run(["squeue", "-h", "-j", job_id], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip() != ""

    while True:
        still_running = [job_id for job_id in submitted_jobs if is_job_active(job_id)]
        if not still_running:
            print("All jobs completed.")
            break
        else:
            print(f"Jobs still running: {', '.join(still_running)}")
            time.sleep(10)

    # No completion marker here: this mode only waits for the jobs to leave the
    # queue, it never learns whether they succeeded, so a marker would let a
    # resumed campaign skip an experiment that actually failed.
    return 0


# --- mode: alloc (steps inside an allocation that is already held) ----------

def inside_usable_allocation() -> bool:
    """True when our own environment describes an allocation we can run steps in.

    ``SLURM_JOB_ID`` alone is not enough. A shell opened inside a long-lived
    ``salloc`` (the agent driving calibration often lives in one) inherits the
    job id without the geometry variables sbatch/srun export — and that
    allocation is usually not the one the work belongs on anyway. Requiring the
    node count keeps this true only for the case it is meant for: the campaign
    job, where the batch script *is* the allocation.
    """
    return bool(os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES"))


def job_state(job_id: str) -> str:
    """SLURM state of a job, or "" if it is not in the queue any more."""
    r = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""


def held_allocation_id(explicit: str | None = None) -> str | None:
    """Job id of an allocation held for us to attach steps to, if any.

    Explicit ``--jobid`` wins, then ``SIMPLACE_ALLOC_JOBID``, then the session
    file written by ``hold_nodes.py``. A recorded allocation that is no longer
    RUNNING is reported and ignored — silently falling through to a mode that
    behaves differently is worse than saying so.
    """
    jid = explicit or os.environ.get("SIMPLACE_ALLOC_JOBID")
    source = "--jobid" if explicit else "SIMPLACE_ALLOC_JOBID"
    if not jid:
        session = REPO_ROOT / ALLOCATION_FILE
        if not session.exists():
            return None
        try:
            jid = str(json.loads(session.read_text())["jobid"])
        except (ValueError, KeyError, OSError) as exc:
            print(f"ignoring unreadable {session}: {exc}")
            return None
        source = str(session)

    jid = str(jid).strip()
    state = job_state(jid)
    if state != "RUNNING":
        print(f"held allocation {jid} (from {source}) is "
              f"{state or 'gone'}, not RUNNING — ignoring it.")
        return None
    return jid


def scontrol_geometry(job_id: str):
    """(nodes, tasks, cpus_per_task) of an allocation we are not running inside."""
    r = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read allocation {job_id}: {r.stderr.strip()}")

    def field(name, default=0):
        m = re.search(rf"(?:^|\s){re.escape(name)}=(\d+)", r.stdout)
        return int(m.group(1)) if m else default

    return field("NumNodes"), field("NumTasks"), field("CPUs/Task")


def allocation_geometry(config: dict, job_id: str | None = None):
    """Nodes / tasks-per-node / cpus-per-task of the allocation we will use.

    The allocation is the authority — it is what actually bounds concurrency —
    so its own values win over the per-experiment `num_nodes` in config.yaml,
    which only describes how wide the sbatch mode would spread the work.
    """
    if job_id:
        nodes, tasks, cpus_per_task = scontrol_geometry(job_id)
        if not nodes:
            raise SystemExit(f"allocation {job_id} reports no nodes")
        tasks_per_node = max(1, tasks // nodes) if tasks else int(config['num_tasks_per_node'])
    else:
        nodes = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
        if not nodes:
            raise SystemExit(
                "alloc mode needs a SLURM allocation: run inside one (the campaign "
                "script from orchestration/generate.py), attach to one held by "
                "orchestration/hold_nodes.py, or use --mode sbatch."
            )
        nodes = int(nodes)
        tasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        tasks_per_node = int(tasks_per_node) if tasks_per_node else int(config['num_tasks_per_node'])
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)

    if not cpus_per_task:
        cpus_per_node = int(config.get('cpus_per_node', DEFAULT_CPUS_PER_NODE))
        cpus_per_task = max(1, cpus_per_node // tasks_per_node)
    return nodes, tasks_per_node, cpus_per_task


def parse_walltime(text: str) -> int:
    """SLURM time string -> seconds.

    Handles the forms squeue %L and `--time` produce: D-HH:MM:SS, D-HH:MM,
    D-HH, HH:MM:SS, MM:SS and a bare integer, which SLURM reads as *minutes*.
    """
    text = str(text).strip()
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
        # After a day field the remainder counts down from hours: "1-6" is 30h.
        parts = [int(p) for p in text.split(":")] if text else [0]
        while len(parts) < 3:
            parts.append(0)         # HH -> HH:00:00, HH:MM -> HH:MM:00
    else:
        parts = [int(p) for p in text.split(":")] if text else [0]
        if len(parts) == 1:
            parts = [0, parts[0], 0]                # bare integer = minutes
        while len(parts) < 3:
            parts.insert(0, 0)                      # MM:SS -> 00:MM:SS
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def remaining_seconds(job_id: str | None = None):
    """Seconds left in an allocation, or None if it cannot be determined."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    r = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%L"],
                       capture_output=True, text=True)
    left = r.stdout.strip().splitlines()
    if r.returncode != 0 or not left or left[0] in ("", "UNLIMITED", "INVALID"):
        return None
    try:
        return parse_walltime(left[0])
    except ValueError:
        return None


def run_in_allocation(config_path: str, min_remaining: str | None = None,
                      chunks_per_slot: int = 1, max_steps: int | None = None,
                      job_id: str | None = None) -> int:
    """Run one experiment as srun steps in an allocation that is already held.

    The allocation is not released here — that is the whole point. Work is cut
    into ``nodes * tasks_per_node * chunks_per_slot`` location-aligned chunks and
    fed through a pool of exactly ``nodes * tasks_per_node`` concurrent srun
    steps, so every slot of the allocation stays busy until the experiment ends.

    ``job_id`` attaches the steps to an allocation this process is not running
    inside (``srun --jobid=``); without it the steps go to our own allocation.
    """
    config = load_cluster_config(config_path)
    paths = prepare_dirs(config)
    exp = config['exp_name']

    nodes, tasks_per_node, cpus_per_task = allocation_geometry(config, job_id)
    slots = max(1, nodes * tasks_per_node)

    # Every concurrent step is one srun client process living on the batch node,
    # so a very wide allocation concentrates hundreds of them there. Capping
    # trades throughput for headroom on that one node.
    cap = max_steps or config.get('max_concurrent_steps')
    if cap and int(cap) < slots:
        print(f"[{exp}] capping concurrency at {cap} of {slots} slot(s)", flush=True)
        slots = int(cap)

    # Refuse to start work the allocation cannot finish: a chunk killed at the
    # walltime leaves a half-written per-location output file that looks valid.
    budget = parse_walltime(min_remaining or config.get('min_remaining')
                            or config.get('walltime', '01:00:00'))
    alloc_id = job_id or os.environ.get("SLURM_JOB_ID")
    left = remaining_seconds(job_id)
    if left is not None and left < budget:
        print(f"[{exp}] only {left}s left in allocation {alloc_id}, "
              f"need {budget}s — not starting.", flush=True)
        return EXIT_OUT_OF_TIME

    ranges, _ = work_ranges(config, slots * max(1, chunks_per_slot))
    if not ranges:
        print(f"[{exp}] nothing to run (empty line selection).", flush=True)
        return 1

    attached = " (attached)" if job_id else ""
    print(f"[{exp}] {len(ranges)} chunk(s) over {slots} slot(s) "
          f"({nodes} node(s) x {tasks_per_node} task(s), {cpus_per_task} cpu(s)/task) "
          f"in allocation {alloc_id}{attached}", flush=True)

    started = time.time()

    def run_chunk(rng):
        first, last = rng
        cmd = ["srun", "--exclusive", "--exact",
               "-N1", "-n1", f"-c{cpus_per_task}",
               f"--job-name=SIM_{exp}_{first}_{last}",
               *([f"--jobid={job_id}"] if job_id else []),
               *singularity_command(config, paths, first, last)]
        log_path = os.path.join(paths['log'], f"step_{exp}_{first}_{last}.out")
        with open(log_path, "w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        return rng, rc, log_path

    failures, done = [], 0
    step = max(1, len(ranges) // 10)
    with ThreadPoolExecutor(max_workers=slots) as pool:
        futures = [pool.submit(run_chunk, rng) for rng in ranges]
        for fut in as_completed(futures):
            rng, rc, log_path = fut.result()
            done += 1
            if rc != 0:
                failures.append((rng, rc, log_path))
                print(f"[{exp}] chunk {rng[0]}-{rng[1]} FAILED rc={rc}  {log_path}",
                      flush=True)
            if done % step == 0 or done == len(ranges):
                print(f"[{exp}] {done}/{len(ranges)} chunks done "
                      f"({time.time() - started:.0f}s elapsed)", flush=True)

    elapsed = time.time() - started
    if failures:
        print(f"[{exp}] FAILED — {len(failures)}/{len(ranges)} chunk(s) "
              f"after {elapsed:.0f}s", flush=True)
        return 1

    print(f"[{exp}] completed {len(ranges)} chunk(s) in {elapsed:.0f}s", flush=True)
    completion_marker(config).write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} mode=alloc "
        f"job={os.environ.get('SLURM_JOB_ID')} chunks={len(ranges)} "
        f"seconds={elapsed:.0f}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="run dir config.yaml (the `cluster:` block)")
    ap.add_argument("--mode", choices=("auto", "sbatch", "alloc"), default="auto",
                    help="auto (default): alloc inside or attached to a held "
                         "allocation, else sbatch")
    ap.add_argument("--jobid", help="alloc mode: attach steps to this allocation "
                                    "instead of submitting (default: the one "
                                    "recorded by hold_nodes.py, if it is running)")
    ap.add_argument("--min-remaining", metavar="HH:MM:SS",
                    help="alloc mode: refuse to start with less allocation time "
                         "left than this (default: the config's walltime)")
    ap.add_argument("--chunks-per-slot", type=int, default=1,
                    help="alloc mode: cut this many work chunks per slot; >1 "
                         "trades more srun steps for a shorter tail (default 1)")
    ap.add_argument("--max-steps", type=int,
                    help="alloc mode: cap concurrent srun steps (default: one "
                         "per allocation slot, nodes x tasks-per-node)")
    ap.add_argument("--skip-completed", action="store_true",
                    help="exit 0 immediately if this experiment already has a "
                         "completion marker (resuming an interrupted campaign)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    print(args.config)

    # Precedence: an explicit --jobid, then our own allocation if we are running
    # in one (the campaign job — its steps belong to it, never to some session
    # file left at the repo root), then an allocation held for us, then submit.
    inside = inside_usable_allocation() and not args.jobid
    held = None if inside else held_allocation_id(args.jobid)

    mode = args.mode
    if mode == "auto":
        mode = "alloc" if (inside or held) else "sbatch"
    elif mode == "alloc" and not inside and not held:
        # Explicitly asked for alloc with nothing to attach to. held_allocation_id
        # already explained any rejection; falling back to sbatch would quietly
        # ignore the instruction and submit jobs the caller did not ask for.
        print("no allocation to run in; not falling back to sbatch.")
        return 1

    if args.skip_completed:
        marker = completion_marker(load_cluster_config(args.config))
        if marker.exists():
            print(f"already completed, skipping ({marker})")
            return 0

    if mode == "alloc":
        return run_in_allocation(args.config, args.min_remaining,
                                 args.chunks_per_slot, args.max_steps, held)
    return run_simplace(args.config)


if __name__ == "__main__":
    sys.exit(main())
