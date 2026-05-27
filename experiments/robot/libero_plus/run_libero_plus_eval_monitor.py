#!/usr/bin/env python3
"""Monitor and resubmit LIBERO+ eval SLURM job until rollout completion."""

import argparse
import glob
import re
import subprocess
import time
from pathlib import Path


def run_cmd(cmd):
    """Run shell command and return CompletedProcess."""
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def read_job_id(job_id_file: Path):
    if not job_id_file.exists():
        return None
    content = job_id_file.read_text().strip()
    if not content:
        return None
    if not content.isdigit():
        return None
    return content


def write_job_id(job_id_file: Path, job_id: str):
    job_id_file.write_text(job_id + "\n")


def is_job_running(job_id: str) -> bool:
    # Use squeue [PID] as requested.
    proc = run_cmd(["squeue", "-j", job_id, "-h", "-o", "%A"])
    if proc.returncode != 0:
        return False
    return any(line.strip() == job_id for line in proc.stdout.splitlines())


def find_active_matching_jobs(script_path: Path, run_id: str, id_note: str):
    """Return active job ids matching this monitor target (R/PD/CG)."""
    proc = run_cmd(["squeue", "-h", "-o", "%A|%T|%o"])
    if proc.returncode != 0:
        return []

    script_name = script_path.name
    matches = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        jid, state, command = parts
        if state not in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"}:
            continue
        # Match the exact target command pattern passed via sbatch.
        if script_name in command and f" {run_id} " in f" {command} " and id_note in command:
            matches.append(jid.strip())
    return matches


def count_rollout_files(rollout_dir: Path) -> int:
    pattern = str(rollout_dir / "*.npy")
    return len(glob.glob(pattern))


def submit_job(script_path: Path, run_id: str, id_note: str) -> str:
    proc = run_cmd(["sbatch", str(script_path), run_id, id_note])
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    match = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {proc.stdout.strip()}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a LIBERO+ eval SLURM job. If running, wait until completion. "
            "If not running, check rollout file count; if incomplete, resubmit."
        )
    )
    parser.add_argument("--run-id", required=True, help="RUN_ID argument passed to run_libero_plus_eval.sh")
    parser.add_argument("--id-note", default="libero_plus_eval", help="ID_NOTE argument passed to run_libero_plus_eval.sh")
    parser.add_argument(
        "--rollout-dir",
        default="./rollouts/OpenVLA-OFT+/libero_object/change_spawn_False_train_False",
        help="Rollout base directory (without /run_<run-id>).",
    )
    parser.add_argument("--expected-files", type=int, default=2518, help="Required number of rollout .npy files.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval while waiting.")
    parser.add_argument(
        "--job-id-file",
        default="./libero_plus_eval_last_jobid.txt",
        help="Path to a file used to persist the last submitted SLURM job ID.",
    )
    parser.add_argument(
        "--script-path",
        default="./run_libero_plus_eval.sh",
        help="Path to the sbatch script to submit.",
    )
    parser.add_argument("--job-id", default=None, help="Optional SLURM PID (job id). Overrides --job-id-file if set.")
    args = parser.parse_args()

    script_path = Path(args.script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    rollout_dir = Path(args.rollout_dir).resolve() / f"run_{args.run_id}"
    job_id_file = Path(args.job_id_file).resolve()

    while True:
        job_id = args.job_id or read_job_id(job_id_file)
        if job_id and is_job_running(job_id):
            print(f"[monitor] job {job_id} is running; waiting {args.poll_seconds}s...")
            time.sleep(args.poll_seconds)
            continue

        # Safety net: even if job-id tracking is stale, do not resubmit while
        # another matching eval job is active.
        matching_jobs = find_active_matching_jobs(script_path, args.run_id, args.id_note)
        if matching_jobs:
            print(
                f"[monitor] found active matching job(s) {', '.join(matching_jobs)}; "
                f"waiting {args.poll_seconds}s..."
            )
            # Keep tracking the most recent matching job.
            newest = matching_jobs[-1]
            write_job_id(job_id_file, newest)
            args.job_id = newest
            time.sleep(args.poll_seconds)
            continue

        file_count = count_rollout_files(rollout_dir)
        print(f"[monitor] rollout dir: {rollout_dir}")
        print(f"[monitor] found {file_count}/{args.expected_files} .npy files")
        if file_count >= args.expected_files:
            print("[monitor] rollout complete. stopping.")
            break

        print("[monitor] rollout incomplete and no running job. submitting a new job...")
        new_job_id = submit_job(script_path, args.run_id, args.id_note)
        write_job_id(job_id_file, new_job_id)
        args.job_id = new_job_id
        print(f"[monitor] submitted job {new_job_id}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
