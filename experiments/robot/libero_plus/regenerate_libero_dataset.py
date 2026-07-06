"""Backward-compatible wrapper. Use regenerate_libero_plus_dataset.py instead."""

from experiments.robot.libero_plus.regenerate_libero_plus_dataset import main
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        required=True,
    )
    parser.add_argument("--libero_raw_data_dir", type=str, required=True)
    parser.add_argument("--libero_target_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
