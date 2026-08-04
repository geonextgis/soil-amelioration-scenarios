import os
import math
import subprocess
import uuid
import datetime
import time
import yaml
from pathlib import Path
import sys
import numpy as np

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


def run_simplace(config_path: str):
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = config['cluster']

    # === Configuration ===
    EXP_NAME = config['exp_name']
    INPUT_CSV = config['input_csv']
    MOUNT_DATA = config['mount_data']
    WORK_DIR = config['work_dir']
    MOUNT_WORK = WORK_DIR
    MOUNT_OUT = os.path.join(WORK_DIR, "out", EXP_NAME)
    OUT_ZIP = os.path.join(WORK_DIR, "out_zip")
    MOUNT_PROJECT = os.path.join(WORK_DIR, "projects")
    SIMPLACE_LOG = os.path.join(WORK_DIR, "log")

    SOLUTION = config['solution']
    PROJECT = config['project']
    SINGULARITY_IMAGE = config['singularity_image']
    DEBUG = str(config.get('debug', 'false')).lower()
    TESTRUN = str(config.get('testrun', 'false')).lower()

    NUM_JOBS = config['num_nodes']
    NUM_TASKS = config['num_tasks_per_node']
    PARTITION = config['partition']
    WALLTIME = config['walltime']

    os.makedirs(MOUNT_OUT, exist_ok=True)
    os.makedirs(OUT_ZIP, exist_ok=True)
    os.makedirs(MOUNT_PROJECT, exist_ok=True)
    os.makedirs(SIMPLACE_LOG, exist_ok=True)

    # === Split work on LOCATION boundaries ===
    # Output is per-location and written into one shared dir, so a location must be
    # handled by exactly one SIMPLACE invocation (else its file gets overwritten).
    loc_starts, total_lines = read_location_starts(INPUT_CSV)

    start = config.get("start_line", 1)
    end = config.get("end_line", total_lines)
    sel = [s for s in loc_starts if start <= s <= end]
    job_ranges = split_on_locations(sel, end, NUM_JOBS)
    submitted_jobs = []

    for i, j in job_ranges:
        job_name = f"SIM_{i}_{j}"
        outdir_run = f"{MOUNT_OUT}"
        os.makedirs(outdir_run, exist_ok=True)

        bind_paths = ",".join([
            f"{MOUNT_WORK}:/simplace/SIMPLACE_WORK",
            f"{outdir_run}:/outputs",
            f"{MOUNT_DATA}:/data",
            f"{MOUNT_PROJECT}:/projects",
            f"{SIMPLACE_LOG}:/simplace/log"
        ])

        execdir = "/simplace"
        simplace_workdir = "/simplace/SIMPLACE_WORK"
        sel_job = [s for s in loc_starts if i <= s <= j]
        subranges = split_on_locations(sel_job, j, NUM_TASKS)
        num_procs_per_task = int(80/NUM_TASKS)
        srun_commands = []
        for idx, (i_sub, j_sub) in enumerate(subranges):
            singularity_cmd = [
                "singularity", "run", "-B", bind_paths, SINGULARITY_IMAGE,
                f"{execdir}/simplace", "run",
                f"-s={simplace_workdir}/{SOLUTION}",
                f"-p={simplace_workdir}/{PROJECT}",
                "-t=CLUSTER",
                f"-l={i_sub}-{j_sub}"
            ]
            if DEBUG != "true":
                singularity_cmd.append("-loglevel=ERROR")
            cmd_str = " ".join(singularity_cmd)#
            srun_commands.append(f"srun -n1 -c {num_procs_per_task} {cmd_str} &")

        # Combine all into a Slurm batch script
        srunc = '\n'.join(srun_commands)
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task=80
#SBATCH --partition={PARTITION}
#SBATCH --time={WALLTIME}
#SBATCH --output={SIMPLACE_LOG}/simplace-%j.out

{srunc}
wait
"""
        slurm_file = f"{WORK_DIR}/.tmp_simplace_job_{i}_{j}.sh"
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
            
            
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simplace_runner.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(config_path)
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    run_simplace(config_path)
