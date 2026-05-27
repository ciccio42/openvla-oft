#!/usr/bin/env python3
"""Verify LIBERO object rollout completion and generate rerun script for incomplete experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_ROLLOUT_ROOT = Path(
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object"
)
DEFAULT_OUTPUT_SH = Path(
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/run_libero_eval_resume_incomplete.sh"
)


def parse_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Expected True/False, got: {text}")


def count_trajectories(run_dir: Path) -> tuple[int, str]:
    npz_count = len(list(run_dir.glob("*.npz")))
    if npz_count > 0:
        return npz_count, "npz"
    npy_count = len(list(run_dir.glob("*.npy")))
    return npy_count, "npy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--expected-trajectories", type=int, default=500)
    parser.add_argument("--output-sh", type=Path, default=DEFAULT_OUTPUT_SH)
    args = parser.parse_args()

    rollout_root = args.rollout_root
    expected = args.expected_trajectories
    incomplete = []

    folder_re = re.compile(r"^change_spawn_(True|False)_train_(True|False)$")
    run_re = re.compile(r"^run_(\d+)$")

    for exp_dir in sorted(rollout_root.glob("change_spawn_*_train_*")):
        match = folder_re.match(exp_dir.name)
        if not exp_dir.is_dir() or match is None:
            continue

        change_spawn = parse_bool(match.group(1))
        train_spawn_distribution = parse_bool(match.group(2))

        run_dirs = sorted([p for p in exp_dir.glob("run_*") if p.is_dir()])
        if not run_dirs:
            incomplete.append((exp_dir.name, 0, "none", change_spawn, train_spawn_distribution, 0))
            continue

        for run_dir in run_dirs:
            run_match = run_re.match(run_dir.name)
            if run_match is None:
                continue

            run_number = int(run_match.group(1))
            count, ext = count_trajectories(run_dir)
            if count < expected:
                incomplete.append(
                    (exp_dir.name, count, ext, change_spawn, train_spawn_distribution, run_number)
                )

    lines = [
        "#!/bin/bash",
        "",
        "#SBATCH -A did_robot_learning_359",
        "#SBATCH --partition=aiq",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --ntasks=1",
        "#SBATCH --nodes=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --export=ALL",
        "",
        "export MUJOCO_PY_MUJOCO_PATH=\"/home/rsofnc000/.mujoco/mujoco210\"",
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin",
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia",
        "",
        "export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO",
        "export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/transformers-openvla-oft",
        "export LIBERO_CONFIG_PATH=\"/mnt/beegfs/frosa/.libero\"",
        "",
        "ID_NOTE=parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio",
        "MODEL_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/open_vla/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--${ID_NOTE}--135000_chkpt",
        "",
    ]

    if incomplete:
        lines.append("echo \"Found incomplete rollout experiments. Re-running only missing experiments...\"")
        for exp_name, count, ext, change_spawn, train_spawn_distribution, run_number in incomplete:
            lines.append(
                f"echo \"Re-running {exp_name}/run_{run_number}: {count}/{expected} trajectories ({ext})\""
            )
            lines.append(
                "srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_libero_eval.py \\"
            )
            lines.append("    --pretrained_checkpoint ${MODEL_PATH} \\")
            lines.append("    --num_images_in_input 2 \\")
            lines.append("    --use_proprio True \\")
            lines.append("    --wandb_entity \"francescorosa97\" \\")
            lines.append("    --wandb_project \"Open_VLA_OFT_finetune\" \\")
            lines.append("    --run_id_note ${ID_NOTE} \\")
            lines.append("    --task_suite_name \"libero_object\" \\")
            lines.append(f"    --run_number {run_number} \\")
            lines.append(f"    --change_spawn {str(change_spawn)} \\")
            lines.append(f"    --spawn_train_distribution {str(train_spawn_distribution)} \\")
            lines.append("    --debug False")
            lines.append("")
    else:
        lines.append("echo \"All rollout experiments are complete (>= expected trajectories).\"")

    args.output_sh.write_text("\n".join(lines) + "\n")
    args.output_sh.chmod(0o755)

    print(f"Rollout root: {rollout_root}")
    if incomplete:
        print("Incomplete experiments:")
        for exp_name, count, ext, _, _, run_number in incomplete:
            print(f"  - {exp_name}/run_{run_number}: {count}/{expected} ({ext})")
    else:
        print("All experiments complete.")
    print(f"Generated rerun script: {args.output_sh}")


if __name__ == "__main__":
    main()
