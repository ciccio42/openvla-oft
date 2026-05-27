#!/usr/bin/env python3
"""Generate rollout spawn heatmaps (bin + target objects) from LIBERO object rollouts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


TABLE_SIZE_M = (0.7, 0.7)  # (height_y, width_x)
PX_RESOLUTION_CM = 0.5
CROP_Y_MIN_MAX_CM = (-30, 30)
CROP_X_MIN_MAX_CM = (-35, 35)
DEFAULT_ROLLOUT_ROOT = Path(
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/experiments/robot/libero/rollouts/libero_object"
)
TASK_FROM_FILENAME_RE = re.compile(r"task=([^.]*)")


def normalize_task_name(task_name: str) -> str:
    task_name = task_name.strip().lower()
    task_name = re.sub(r"_(table|tb)_\d+$", "", task_name)
    task_name = re.sub(r"_add_\d+$", "", task_name)
    return task_name


def extract_task_name(payload: dict, npy_file: Path) -> str | None:
    task_name = payload.get("original_task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name

    task_name = payload.get("task_name")
    if isinstance(task_name, str) and task_name.strip():
        return task_name

    task_name = payload.get("task_command")
    if isinstance(task_name, str) and task_name.strip():
        return task_name

    match = TASK_FROM_FILENAME_RE.search(npy_file.name)
    if match:
        return match.group(1)
    return None


def get_spawn_value(payload: dict, key_candidates: list[str]):
    for key in key_candidates:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def to_xyz_points(value) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[None, :]
    if arr.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return arr[:, :3]


def build_density_map(points_xy: np.ndarray) -> np.ndarray:
    table_size_cm = np.array(TABLE_SIZE_M) * 100.0
    table_size_px = (table_size_cm / PX_RESOLUTION_CM).astype(np.int32)
    table_map = np.zeros((table_size_px[0], table_size_px[1]), dtype=np.float64)

    if points_xy.size == 0:
        return table_map

    px_points = (points_xy * 100.0 / PX_RESOLUTION_CM).astype(np.int32)
    px_points[:, 0] = table_map.shape[0] // 2 + px_points[:, 0]
    px_points[:, 1] = table_map.shape[1] // 2 + px_points[:, 1]

    valid = (
        (px_points[:, 0] >= 0)
        & (px_points[:, 0] < table_map.shape[0])
        & (px_points[:, 1] >= 0)
        & (px_points[:, 1] < table_map.shape[1])
    )
    px_points = px_points[valid]

    for x, y in px_points:
        table_map[x, y] += 1
    return table_map


def crop_map(table_map: np.ndarray) -> np.ndarray:
    table_size_cm = np.array(TABLE_SIZE_M) * 100.0
    y_min, y_max = CROP_Y_MIN_MAX_CM
    x_min, x_max = CROP_X_MIN_MAX_CM

    y_min_px = int((y_min + table_size_cm[0] / 2.0) / PX_RESOLUTION_CM)
    y_max_px = int((y_max + table_size_cm[0] / 2.0) / PX_RESOLUTION_CM)
    x_min_px = int((x_min + table_size_cm[1] / 2.0) / PX_RESOLUTION_CM)
    x_max_px = int((x_max + table_size_cm[1] / 2.0) / PX_RESOLUTION_CM)

    return table_map[y_min_px:y_max_px, x_min_px:x_max_px]


def render_heatmap(cropped_map: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.max(cropped_map) if np.max(cropped_map) > 0 else 1

    masked = np.ma.masked_where(cropped_map == 0, cropped_map)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, cmap=cmap, origin="upper", vmin=1, vmax=vmax)
    ax.invert_xaxis()

    x_min, x_max = CROP_X_MIN_MAX_CM
    y_min, y_max = CROP_Y_MIN_MAX_CM

    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / PX_RESOLUTION_CM))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / PX_RESOLUTION_CM))
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(np.arange(x_min, x_max, 10))
    ax.set_yticklabels(np.arange(y_min, y_max, 10))

    ax.set_title(title)
    ax.set_xlabel("Y Axis (cm)")
    ax.set_ylabel("X Axis (cm)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Density")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def concat_points(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def concat_points_3d(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def merge_task_point_maps(dst: dict[str, list[np.ndarray]], src: dict[str, list[np.ndarray]]) -> None:
    for task_key, chunks in src.items():
        if chunks:
            dst[task_key].extend(chunks)


def compute_spawn_stats(chunks: list[np.ndarray]) -> dict:
    pts = concat_points_3d(chunks)
    if pts.size == 0:
        return {
            "num_rollouts": len(chunks),
            "num_points": 0,
            "mean_xyz": [None, None, None],
            "std_xyz": [None, None, None],
        }
    return {
        "num_rollouts": len(chunks),
        "num_points": int(pts.shape[0]),
        "mean_xyz": np.mean(pts, axis=0).tolist(),
        "std_xyz": np.std(pts, axis=0).tolist(),
    }


def parse_rollouts(npy_files: list[Path]):
    bin_xy_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    target_xy_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    bin_xyz_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    target_xyz_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    used_files = 0
    missing_task_name = 0
    missing_spawn_fields = 0

    for npy_file in npy_files:
        try:
            payload = np.load(npy_file, allow_pickle=True).item()
        except Exception as exc:
            print(f"[WARN] Failed to load {npy_file}: {exc}")
            continue

        if not isinstance(payload, dict):
            print(f"[WARN] Skipping non-dict payload: {npy_file}")
            continue

        task_name = extract_task_name(payload, npy_file)
        if not isinstance(task_name, str) or not task_name.strip():
            missing_task_name += 1
            continue

        task_key = normalize_task_name(task_name)
        bin_raw = get_spawn_value(payload, ["bin_position", "bin_positions", "bin_pos"])
        target_raw = get_spawn_value(
            payload, ["target_object_positions", "target_positions", "target_object_pos"]
        )
        bin_xyz = to_xyz_points(bin_raw)
        target_xyz = to_xyz_points(target_raw)
        bin_xy = bin_xyz[:, :2] if bin_xyz.size > 0 else np.empty((0, 2), dtype=np.float64)
        target_xy = target_xyz[:, :2] if target_xyz.size > 0 else np.empty((0, 2), dtype=np.float64)

        if bin_xy.size == 0 and target_xy.size == 0:
            missing_spawn_fields += 1

        if bin_xy.size > 0:
            bin_xy_by_task[task_key].append(bin_xy)
            bin_xyz_by_task[task_key].append(bin_xyz)
        if target_xy.size > 0:
            target_xy_by_task[task_key].append(target_xy)
            target_xyz_by_task[task_key].append(target_xyz)

        used_files += 1

    return (
        bin_xy_by_task,
        target_xy_by_task,
        bin_xyz_by_task,
        target_xyz_by_task,
        used_files,
        missing_task_name,
        missing_spawn_fields,
    )


def write_task_outputs(task_keys: set[str], bin_xy_by_task, target_xy_by_task, bin_xyz_by_task, target_xyz_by_task, output_dir: Path) -> tuple[int, int]:
    written_heatmaps = 0
    written_stats = 0

    for task_key in sorted(task_keys):
        task_dir = output_dir / task_key
        bin_points = concat_points(bin_xy_by_task.get(task_key, []))
        target_points = concat_points(target_xy_by_task.get(task_key, []))

        if bin_points.size > 0 or target_points.size > 0:
            all_points = np.concatenate([arr for arr in [bin_points, target_points] if arr.size > 0], axis=0)
            combined_map = crop_map(build_density_map(all_points))
            render_heatmap(
                cropped_map=combined_map,
                title=f"Command:'{task_key.replace('_', ' ')}' ", # remove '_' characters for better readability
                save_path=task_dir / "combined_bin_target_heatmap.png",
            )
            written_heatmaps += 1

        payload = {
            "original_task_name": task_key,
            "bin_position": compute_spawn_stats(bin_xyz_by_task.get(task_key, [])),
            "target_object_positions": compute_spawn_stats(target_xyz_by_task.get(task_key, [])),
        }
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "spawn_stats.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        written_stats += 1

    return written_heatmaps, written_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LIBERO object rollout spawn heatmaps from .npy files")
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder under rollout-root (e.g. change_spawn_False_train_False)",
    )
    parser.add_argument(
        "--combine-train-comparison",
        action="store_true",
        help=(
            "If set, combines rollouts from the two experiments "
            "'change_spawn_True_train_True' and 'change_spawn_True_train_False' "
            "and writes one plot per command for the given run index."
        ),
    )
    parser.add_argument("--run-idx", type=int, required=True, help="Run index used to select run_[RUN_IDX]")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <rollout-root>/<experiment>/analysis/spawn_heatmaps/run_[RUN_IDX])",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, will print additional debug information during processing",
    )
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    if args.combine_train_comparison:
        experiments = ["change_spawn_True_train_True", "change_spawn_True_train_False"]
    else:
        if not args.experiment:
            raise ValueError("`--experiment` is required unless `--combine-train-comparison` is set.")
        experiments = [args.experiment]

    total_used_files = 0
    total_npy_files = 0
    total_missing_task_name = 0
    total_missing_spawn_fields = 0

    bin_xy_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    target_xy_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    bin_xyz_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    target_xyz_by_task: dict[str, list[np.ndarray]] = defaultdict(list)

    resolved_run_dirs: list[Path] = []

    for experiment in experiments:
        run_dir = (args.rollout_root / experiment / f"run_{args.run_idx}").resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        npy_files = sorted(run_dir.glob("*.npy"))
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {run_dir}")

        (
            exp_bin_xy_by_task,
            exp_target_xy_by_task,
            exp_bin_xyz_by_task,
            exp_target_xyz_by_task,
            used_files,
            missing_task_name,
            missing_spawn_fields,
        ) = parse_rollouts(npy_files)

        merge_task_point_maps(bin_xy_by_task, exp_bin_xy_by_task)
        merge_task_point_maps(target_xy_by_task, exp_target_xy_by_task)
        merge_task_point_maps(bin_xyz_by_task, exp_bin_xyz_by_task)
        merge_task_point_maps(target_xyz_by_task, exp_target_xyz_by_task)

        total_used_files += used_files
        total_npy_files += len(npy_files)
        total_missing_task_name += missing_task_name
        total_missing_spawn_fields += missing_spawn_fields
        resolved_run_dirs.append(run_dir)

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif args.combine_train_comparison:
        output_dir = (
            args.rollout_root
            / "analysis"
            / "spawn_heatmaps"
            / "train_true_false_combined"
            / f"run_{args.run_idx}"
        ).resolve()
    else:
        output_dir = (args.rollout_root / experiments[0] / "analysis" / "spawn_heatmaps" / f"run_{args.run_idx}").resolve()

    task_keys = set(bin_xy_by_task.keys()) | set(target_xy_by_task.keys())
    written_heatmaps, written_stats = write_task_outputs(
        task_keys=task_keys,
        bin_xy_by_task=bin_xy_by_task,
        target_xy_by_task=target_xy_by_task,
        bin_xyz_by_task=bin_xyz_by_task,
        target_xyz_by_task=target_xyz_by_task,
        output_dir=output_dir,
    )

    print(f"Run folder(s): {resolved_run_dirs}")
    print(f"Loaded {total_used_files}/{total_npy_files} rollout files")
    print(f"Discovered {len(task_keys)} tasks")
    print(f"Files missing task name: {total_missing_task_name}")
    print(f"Files with no bin/target spawn fields: {total_missing_spawn_fields}")
    print(f"Heatmaps written: {written_heatmaps}")
    print(f"Stats JSON written: {written_stats}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
