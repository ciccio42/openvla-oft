#!/usr/bin/env python3
"""Compute success rate and x-y gripper coverage heatmap for rollout runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


TABLE_SIZE_M = (0.7, 0.7)  # (x, y) table size in meters
PX_RESOLUTION_CM = 0.5


def extract_success_from_name(file_path: Path) -> bool | None:
    match = re.search(r"success=(True|False)", file_path.name)
    if not match:
        return None
    return match.group(1) == "True"


def extract_task_from_name(file_path: Path) -> str:
    match = re.search(r"task=([^.]*)", file_path.name)
    if not match:
        return "unknown_task"
    return match.group(1)


def load_states(rollout_file: Path) -> list:
    """Load state list from .npy or .npz rollout file."""
    if rollout_file.suffix == ".npy":
        data = np.load(rollout_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            payload = data.item()
            return payload.get("states", payload.get("state", []))
        return []

    if rollout_file.suffix == ".npz":
        data = np.load(rollout_file, allow_pickle=True)
        if "states" in data:
            return data["states"].tolist()
        if "state" in data:
            return data["state"].tolist()
        return []

    return []


def collect_xy_points_from_files(rollout_files: list[Path]) -> np.ndarray:
    xy_points = []
    for rollout_file in rollout_files:
        states = load_states(rollout_file)
        for s in states:
            if len(s) >= 2:
                xy_points.append([float(s[0]), float(s[1])])
    if not xy_points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(xy_points, dtype=np.float32)


def build_heatmap(xy_points: np.ndarray) -> np.ndarray:
    table_size_cm = np.array(TABLE_SIZE_M) * 100.0
    table_size_px = (table_size_cm / PX_RESOLUTION_CM).astype(np.int32)
    table_map = np.zeros((table_size_px[0], table_size_px[1]), dtype=np.float32)

    if xy_points.size == 0:
        return table_map

    px = (xy_points * 100.0 / PX_RESOLUTION_CM).astype(np.int32)
    px[:, 0] = table_map.shape[0] // 2 + px[:, 0]
    px[:, 1] = table_map.shape[1] // 2 + px[:, 1]

    valid = (
        (px[:, 0] >= 0)
        & (px[:, 0] < table_map.shape[0])
        & (px[:, 1] >= 0)
        & (px[:, 1] < table_map.shape[1])
    )
    px = px[valid]
    for x, y in px:
        table_map[x, y] += 1.0
    return table_map


def save_heatmap(table_map: np.ndarray, output_path: Path, title: str) -> None:
    y_min, y_max = -30, 30
    x_min, x_max = -35, 35
    table_size_cm = np.array(TABLE_SIZE_M) * 100.0

    y_min_px = int((y_min + table_size_cm[0] / 2) / PX_RESOLUTION_CM)
    y_max_px = int((y_max + table_size_cm[0] / 2) / PX_RESOLUTION_CM)
    x_min_px = int((x_min + table_size_cm[1] / 2) / PX_RESOLUTION_CM)
    x_max_px = int((x_max + table_size_cm[1] / 2) / PX_RESOLUTION_CM)
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlabel("Y Axis (cm)")
    ax.set_ylabel("X Axis (cm)")

    vmax = float(np.max(cropped_map)) if np.max(cropped_map) > 0 else 1.0
    norm = mcolors.LogNorm(vmin=1, vmax=vmax)
    im = ax.imshow(cropped_map, cmap="plasma", origin="upper", norm=norm)
    ax.invert_xaxis()

    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / PX_RESOLUTION_CM))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / PX_RESOLUTION_CM))
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(np.arange(x_min, x_max, 10))
    ax.set_yticklabels(np.arange(y_min, y_max, 10))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def analyze_run(run_dir: Path, output_dir: Path) -> dict:
    rollout_files = sorted(list(run_dir.glob("*.npz")) + list(run_dir.glob("*.npy")))
    success_flags = [extract_success_from_name(p) for p in rollout_files]
    success_flags = [s for s in success_flags if s is not None]
    total = len(success_flags)
    successes = sum(success_flags)
    success_rate = (100.0 * successes / total) if total > 0 else 0.0

    xy_points = collect_xy_points_from_files(rollout_files)
    table_map = build_heatmap(xy_points)
    heatmap_path = output_dir / f"{run_dir.name}_gripper_xy_heatmap.png"
    save_heatmap(table_map, heatmap_path, f"Gripper XY Coverage - {run_dir.name}")

    # Per-task heatmaps
    task_to_files: dict[str, list[Path]] = {}
    for rollout_file in rollout_files:
        task_name = extract_task_from_name(rollout_file)
        task_to_files.setdefault(task_name, []).append(rollout_file)

    task_heatmaps = []
    for task_name, files in sorted(task_to_files.items()):
        task_xy = collect_xy_points_from_files(files)
        task_map = build_heatmap(task_xy)
        task_heatmap_path = output_dir / f"{run_dir.name}__task_{task_name}_gripper_xy_heatmap.png"
        pretty_task_name = task_name.replace("_", " ")
        save_heatmap(task_map, task_heatmap_path, f"Gripper XY - {run_dir.name} - {pretty_task_name}")
        task_heatmaps.append((task_name, len(files), str(task_heatmap_path)))

    return {
        "run_name": run_dir.name,
        "num_trajectories": len(rollout_files),
        "num_success_parsed": total,
        "num_successes": int(successes),
        "success_rate_percent": success_rate,
        "heatmap_path": str(heatmap_path),
        "task_heatmaps": task_heatmaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-root",
        type=Path,
        required=True,
        help="Path containing run_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for heatmaps and summary. Default: <rollout-root>/analysis",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    
    args = parser.parse_args()

    if args.debug:
        print(f"Debug mode enabled. Arguments: {args}")
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")
        

    rollout_root = args.rollout_root
    output_dir = args.output_dir or (rollout_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in rollout_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run_* directories found in: {rollout_root}")

    results = []
    for run_dir in run_dirs:
        result = analyze_run(run_dir, output_dir)
        results.append(result)
        print(
            f"{result['run_name']}: success {result['num_successes']}/{result['num_success_parsed']} "
            f"({result['success_rate_percent']:.2f}%), heatmap={result['heatmap_path']}"
        )

    summary_path = output_dir / "summary_success_rates.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"{r['run_name']}\ttrajectories={r['num_trajectories']}\t"
                f"success={r['num_successes']}/{r['num_success_parsed']}\t"
                f"success_rate={r['success_rate_percent']:.4f}\t"
                f"heatmap={r['heatmap_path']}\n"
            )
            for task_name, count, task_heatmap in r["task_heatmaps"]:
                f.write(
                    f"  task={task_name}\ttrajectories={count}\theatmap={task_heatmap}\n"
                )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
