#!/usr/bin/env python3
"""Generate LIBERO rollout heatmaps from .npy episode files.

For each `original_task_name`, the script builds heatmaps from the XY positions of:
- `bin_position`
- `target_object_positions`

It also reads task classification JSON and creates an additional set of heatmaps
for tasks that belong to the `Objects Layout` category.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


TABLE_SIZE_M = (0.7, 0.7)  # (height_y, width_x)
PX_RESOLUTION_CM = 0.5
CROP_Y_MIN_MAX_CM = (-30, 30)
CROP_X_MIN_MAX_CM = (-35, 35)


def normalize_task_name(task_name: str) -> str:
    """Normalize task names so table-index variants map to one task instruction."""
    task_name = task_name.strip().lower()
    task_name = re.sub(r"_(table|tb)_\d+$", "", task_name)
    task_name = re.sub(r"_add_\d+$", "", task_name)
    return task_name


def safe_filename(name: str) -> str:
    """Keep name readable while avoiding filesystem-invalid characters."""
    invalid = set('<>:"/\\|?*')
    return "".join("_" if (c in invalid or ord(c) < 32) else c for c in name).strip()


def to_xy_points(value) -> np.ndarray:
    """Convert rollout field into Nx2 array (x, y)."""
    if value is None:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return np.empty((0, 2), dtype=np.float64)
        arr = arr[None, :]
    return arr[:, :2]


def to_xyz_points(value) -> np.ndarray:
    """Convert rollout field into Nx3 array (x, y, z)."""
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
    """Build a table-space density map from Nx2 positions in meters."""
    table_size_cm = np.array(TABLE_SIZE_M) * 100.0
    table_size_px = (table_size_cm / PX_RESOLUTION_CM).astype(np.int32)
    table_map = np.zeros((table_size_px[0], table_size_px[1]), dtype=np.float64)

    if points_xy.size == 0:
        return table_map

    px_points = (points_xy * 100.0 / PX_RESOLUTION_CM).astype(np.int32)

    # Map table-centered coordinates to image coordinates.
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

    # Render zero-density cells as white and use linear scale for density.
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


def load_objects_layout_tasks(classification_json: Path) -> Tuple[Set[str], Set[str]]:
    with classification_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    objects_layout_tasks_raw: Set[str] = set()
    objects_layout_tasks_normalized: Set[str] = set()
    for _, task_list in data.items():
        if not isinstance(task_list, list):
            continue
        for item in task_list:
            if not isinstance(item, dict):
                continue
            if item.get("category") == "Objects Layout":
                name = item.get("name")
                if isinstance(name, str):
                    objects_layout_tasks_raw.add(name.strip().lower())
                    objects_layout_tasks_normalized.add(normalize_task_name(name))
    return objects_layout_tasks_raw, objects_layout_tasks_normalized


def discover_rollout_files(rollout_dir: Path) -> List[Path]:
    return sorted(rollout_dir.rglob("*.npy"))


def is_objects_layout_task_name(
    task_name: str,
    objects_layout_tasks_raw: Set[str],
    objects_layout_tasks_normalized: Set[str],
) -> bool:
    task_name_l = task_name.strip().lower()
    return (
        task_name_l in objects_layout_tasks_raw
        # or normalize_task_name(task_name_l) in objects_layout_tasks_normalized
    )


def parse_rollouts(
    npy_files: Iterable[Path],
    objects_layout_tasks_raw: Set[str],
    objects_layout_tasks_normalized: Set[str],
) -> Tuple[
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, int],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, Dict[str, Path]],
    int,
]:
    bin_xy_by_task: Dict[str, List[np.ndarray]] = defaultdict(list)
    target_xy_by_task: Dict[str, List[np.ndarray]] = defaultdict(list)
    bin_xyz_by_task: Dict[str, List[np.ndarray]] = defaultdict(list)
    target_xyz_by_task: Dict[str, List[np.ndarray]] = defaultdict(list)
    bin_xy_by_task_object_layout: Dict[str, List[np.ndarray]] = defaultdict(list)
    target_xy_by_task_object_layout: Dict[str, List[np.ndarray]] = defaultdict(list)
    bin_xyz_by_task_object_layout: Dict[str, List[np.ndarray]] = defaultdict(list)
    target_xyz_by_task_object_layout: Dict[str, List[np.ndarray]] = defaultdict(list)
    object_layout_videos_by_task: Dict[str, Dict[str, Path]] = defaultdict(dict)
    dd_count_by_task: Dict[str, int] = defaultdict(int)
    used_files = 0

    for npy_file in npy_files:
        try:
            payload = np.load(npy_file, allow_pickle=True).item()
        except Exception as exc:
            print(f"[WARN] Failed to load {npy_file}: {exc}")
            continue

        if not isinstance(payload, dict):
            print(f"[WARN] Skipping non-dict payload: {npy_file}")
            continue

        task_name = payload.get("original_task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            print(f"[WARN] Missing original_task_name in {npy_file}")
            continue

        rollout_task_name = payload.get("task_name")
        is_object_layout_rollout = (
            isinstance(rollout_task_name, str)
            and is_objects_layout_task_name(
                rollout_task_name,
                objects_layout_tasks_raw=objects_layout_tasks_raw,
                objects_layout_tasks_normalized=objects_layout_tasks_normalized,
            )
        )
        
        print(f"[INFO] Processing {rollout_task_name}")

        task_key = normalize_task_name(task_name)
        bin_xyz = to_xyz_points(payload.get("bin_position"))
        target_xyz = to_xyz_points(payload.get("target_object_positions"))
        bin_xy = bin_xyz[:, :2] if bin_xyz.size > 0 else np.empty((0, 2), dtype=np.float64)
        target_xy = target_xyz[:, :2] if target_xyz.size > 0 else np.empty((0, 2), dtype=np.float64)

        if bin_xy.size > 0:
            bin_xy_by_task[task_key].append(bin_xy)
            bin_xyz_by_task[task_key].append(bin_xyz)
            if is_object_layout_rollout:
                bin_xy_by_task_object_layout[task_key].append(bin_xy)
                bin_xyz_by_task_object_layout[task_key].append(bin_xyz)
        if target_xy.size > 0:
            target_xy_by_task[task_key].append(target_xy)
            target_xyz_by_task[task_key].append(target_xyz)
            if is_object_layout_rollout:
                target_xy_by_task_object_layout[task_key].append(target_xy)
                target_xyz_by_task_object_layout[task_key].append(target_xyz)
                if isinstance(rollout_task_name, str):
                    video_path = npy_file.with_suffix(".mp4")
                    if video_path.exists() and rollout_task_name not in object_layout_videos_by_task[task_key]:
                        object_layout_videos_by_task[task_key][rollout_task_name] = video_path

        # Keep track of DD presence/count even if not used for plotting.
        if payload.get("DD") is not None:
            dd_count_by_task[task_key] += 1

        used_files += 1

    return (
        bin_xy_by_task,
        target_xy_by_task,
        bin_xyz_by_task,
        target_xyz_by_task,
        dd_count_by_task,
        bin_xy_by_task_object_layout,
        target_xy_by_task_object_layout,
        bin_xyz_by_task_object_layout,
        target_xyz_by_task_object_layout,
        object_layout_videos_by_task,
        used_files,
    )


def concat_points(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def concat_points_3d(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    out = []
    for arr in chunks:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            if a.shape[0] >= 3:
                a = a[None, :3]
            else:
                continue
        elif a.ndim >= 2:
            if a.shape[1] >= 3:
                a = a[:, :3]
            else:
                continue
        out.append(a)
    if not out:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(out, axis=0)


def compute_spawn_stats(chunks: List[np.ndarray]) -> dict:
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


def write_task_stats_json(
    task_keys: Iterable[str],
    bin_by_task: Dict[str, List[np.ndarray]],
    target_by_task: Dict[str, List[np.ndarray]],
    output_dir: Path,
    filename: str,
) -> int:
    written = 0
    for task_key in sorted(set(task_keys)):
        payload = {
            "original_task_name": task_key,
            "bin_position": compute_spawn_stats(bin_by_task.get(task_key, [])),
            "target_object_positions": compute_spawn_stats(target_by_task.get(task_key, [])),
        }
        task_dir = output_dir / task_key
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        written += 1
    return written


def generate_task_heatmaps(task_keys: Iterable[str], bin_by_task: Dict[str, List[np.ndarray]], target_by_task: Dict[str, List[np.ndarray]], output_dir: Path, label_prefix: str) -> int:
    written = 0
    for task_key in sorted(set(task_keys)):
        bin_points = concat_points(bin_by_task.get(task_key, []))
        target_points = concat_points(target_by_task.get(task_key, []))

        if bin_points.size == 0 and target_points.size == 0:
            continue

        task_dir = output_dir / task_key
        all_points = np.concatenate([arr for arr in [bin_points, target_points] if arr.size > 0], axis=0)
        combined_map = crop_map(build_density_map(all_points))
        render_heatmap(
            cropped_map=combined_map,
            title=f"{label_prefix} | {task_key} | bin+target",
            save_path=task_dir / "combined_bin_target_heatmap.png",
        )
        written += 1

    return written


def copy_object_layout_videos(
    videos_by_original_task: Dict[str, Dict[str, Path]],
    output_dir: Path,
) -> int:
    copied = 0
    for original_task_name, task_to_video in videos_by_original_task.items():
        task_dir = output_dir / original_task_name / "videos"
        task_dir.mkdir(parents=True, exist_ok=True)
        for task_name, src in sorted(task_to_video.items()):
            dst = task_dir / f"{safe_filename(task_name)}.mp4"
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-task rollout heatmaps from LIBERO rollout .npy files")
    parser.add_argument("--rollout_dir", type=Path, help="Path to rollout run folder (contains .npy files)")
    parser.add_argument(
        "--classification-json",
        type=Path,
        default=Path("/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero/benchmark/task_classification.json"),
        help="Path to task_classification.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root directory (default: <rollout_dir>/heatmaps)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging and no file writing",
    )

    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        print("Debug mode enabled. Waiting for debugger to attach on port 5678...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("Debugger attached. Continuing execution.")
    
    rollout_dir = args.rollout_dir.resolve()
    if not rollout_dir.exists() or not rollout_dir.is_dir():
        raise FileNotFoundError(f"Invalid rollout_dir: {rollout_dir}")

    output_dir = (args.output_dir.resolve() if args.output_dir else rollout_dir / "heatmaps")
    output_all = output_dir / "all_tasks"
    output_objects_layout = output_dir / "objects_layout"

    npy_files = discover_rollout_files(rollout_dir)
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {rollout_dir}")

    objects_layout_tasks_raw, objects_layout_tasks_normalized = load_objects_layout_tasks(
        args.classification_json
    )
    (
        bin_xy_by_task,
        target_xy_by_task,
        bin_xyz_by_task,
        target_xyz_by_task,
        dd_count_by_task,
        bin_xy_by_task_object_layout,
        target_xy_by_task_object_layout,
        bin_xyz_by_task_object_layout,
        target_xyz_by_task_object_layout,
        object_layout_videos_by_task,
        used_files,
    ) = parse_rollouts(
        npy_files=npy_files,
        objects_layout_tasks_raw=objects_layout_tasks_raw,
        objects_layout_tasks_normalized=objects_layout_tasks_normalized,
    )
    all_tasks = set(bin_xy_by_task.keys()) | set(target_xy_by_task.keys())
    objects_layout_in_rollout = sorted(
        set(bin_xy_by_task_object_layout.keys()) | set(target_xy_by_task_object_layout.keys())
    )

    written_all = generate_task_heatmaps(
        task_keys=all_tasks,
        bin_by_task=bin_xy_by_task,
        target_by_task=target_xy_by_task,
        output_dir=output_all,
        label_prefix="All Tasks",
    )

    written_obj = generate_task_heatmaps(
        task_keys=objects_layout_in_rollout,
        bin_by_task=bin_xy_by_task_object_layout,
        target_by_task=target_xy_by_task_object_layout,
        output_dir=output_objects_layout,
        label_prefix="Objects Layout",
    )

    written_stats_all = write_task_stats_json(
        task_keys=all_tasks,
        bin_by_task=bin_xyz_by_task,
        target_by_task=target_xyz_by_task,
        output_dir=output_all,
        filename="spawn_stats.json",
    )
    written_stats_obj = write_task_stats_json(
        task_keys=objects_layout_in_rollout,
        bin_by_task=bin_xyz_by_task_object_layout,
        target_by_task=target_xyz_by_task_object_layout,
        output_dir=output_objects_layout,
        filename="spawn_stats.json",
    )
    copied_object_layout_videos = copy_object_layout_videos(
        videos_by_original_task=object_layout_videos_by_task,
        output_dir=output_objects_layout,
    )

    print(f"Loaded {used_files}/{len(npy_files)} rollout files.")
    print(f"Discovered {len(all_tasks)} tasks with valid positions.")
    print(f"Tasks in 'Objects Layout': {len(objects_layout_in_rollout)}")
    print(f"DD present in {sum(dd_count_by_task.values())} episodes (tracked, not plotted).")
    print(f"Heatmaps written: {written_all} (all tasks), {written_obj} (Objects Layout)")
    print(f"Stats JSON written: {written_stats_all} (all tasks), {written_stats_obj} (Objects Layout)")
    print(f"Videos copied (Objects Layout): {copied_object_layout_videos}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
