"""
run_libero_trajectory_heatmap.py

Runs inference with different command variations (default, l1, l2, l3) and 
generates trajectory heatmaps comparing how the model behaves with different commands.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

import torch
from contextlib import contextmanager

if not hasattr(torch.serialization, 'safe_globals'):
    @contextmanager
    def _safe_globals_compat(allowed_types):
        yield
    torch.serialization.safe_globals = _safe_globals_compat
    print("[INFO] Applied PyTorch 2.2+ compatibility patch for LIBERO")

from libero.libero import benchmark

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.openvla_utils import (
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Import from run_libero_eval for reuse
from run_libero_eval import (
    TaskSuite,
    TASK_MAX_STEPS,
    validate_config,
    initialize_model,
    check_unnorm_key,
    prepare_observation,
    process_action,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Heatmap settings
TABLE_SIZE = (0.7, 0.7)  # meters
COMMAND_LEVELS = ['default', 'l1', 'l2', 'l3']


@dataclass
class HeatmapConfig:
    # fmt: off
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8

    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 20  # Episodes per task per command level
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    #################################################################################################################
    # Heatmap-specific parameters
    #################################################################################################################
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/trajectory_heatmaps"
    save_videos: bool = False
    command_levels: str = "all"  # "all", "default", "l1", "l2", "l3", or comma-separated like "l1,l2"
    single_task_id: Optional[int] = None  # If specified (0-indexed), only run this task
    single_task_name: Optional[str] = None  # Alternative: specify by name (e.g., "turn on the stove")

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 42
    debug: bool = False
    # fmt: on


def run_episode_and_collect_trajectory(
    cfg: HeatmapConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
):
    """Run a single episode and collect the trajectory (end-effector positions)."""
    
    # Reset environment
    env.reset()
    
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    trajectory = []  # List of [x, y, z] positions
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Collect end-effector position
            eef_pos = obs["robot0_eef_pos"].copy()
            trajectory.append(eef_pos)

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_vla_action(
                    cfg,
                    model,
                    processor,
                    observation,
                    task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        logger.error(f"Episode error: {e}")

    return success, np.array(trajectory), replay_images


def generate_heatmap(trajectories, task_name, command, level, output_dir):
    """Generate a single heatmap for a set of trajectories."""
    px_resolution = 0.5  # cm
    table_size_cm = np.array(TABLE_SIZE) * 100
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)

    table_map = np.zeros((table_size_px[0], table_size_px[1]))

    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue
        traj_xy = trajectory[:, :2]
        px_traj = (traj_xy * 100 / px_resolution).astype(np.int32)
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]
        
        px_traj = px_traj[
            (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
            (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
        ]
        
        for x, y in px_traj:
            table_map[x, y] += 1

    # Crop
    y_min, y_max = -30, 30
    x_min, x_max = -35, 35
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(f"[{level.upper()}] Command: '{command}'", fontsize=11)
    plt.xlabel("Y Axis (cm)")
    plt.ylabel("X Axis (cm)")

    norm = mcolors.LogNorm(vmin=1, vmax=np.max(cropped_map) if np.max(cropped_map) > 0 else 1)
    im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
    ax.invert_xaxis()

    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
    tick_labels_x = np.arange(x_min, x_max, 10)
    tick_labels_y = np.arange(y_min, y_max, 10)
    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    level_dir = os.path.join(output_dir, level)
    os.makedirs(level_dir, exist_ok=True)
    safe_name = task_name.replace(" ", "_").lower()
    save_path = os.path.join(level_dir, f"{safe_name}_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to {save_path}")
    return cropped_map


def generate_comparison_heatmap(all_trajectories, task_name, commands, output_dir):
    """Generate side-by-side comparison heatmap for all command levels."""
    px_resolution = 0.5
    table_size_cm = np.array(TABLE_SIZE) * 100
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)

    y_min, y_max = -30, 30
    x_min, x_max = -35, 35
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)

    levels = list(all_trajectories.keys())
    n_levels = len(levels)
    
    if n_levels == 0:
        return

    fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 6))
    if n_levels == 1:
        axes = [axes]

    fig.suptitle(f"Task: '{task_name}'\nModel Trajectories with Command Variations", fontsize=12)

    global_max = 1
    all_cropped_maps = {}

    # First pass: compute heatmaps
    for level in levels:
        trajectories = all_trajectories[level]
        table_map = np.zeros((table_size_px[0], table_size_px[1]))

        for trajectory in trajectories:
            if len(trajectory) == 0:
                continue
            traj_xy = trajectory[:, :2]
            px_traj = (traj_xy * 100 / px_resolution).astype(np.int32)
            px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]
            px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]
            
            px_traj = px_traj[
                (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
                (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
            ]
            
            for x, y in px_traj:
                table_map[x, y] += 1

        cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]
        all_cropped_maps[level] = cropped_map
        global_max = max(global_max, np.max(cropped_map))

    # Second pass: plot
    norm = mcolors.LogNorm(vmin=1, vmax=global_max)

    for idx, level in enumerate(levels):
        ax = axes[idx]
        cropped_map = all_cropped_maps[level]
        command = commands.get(level, task_name)

        im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
        ax.invert_xaxis()
        ax.set_title(f"[{level.upper()}]\n\"{command}\"", fontsize=9)
        ax.set_xlabel("Y Axis (cm)")
        ax.set_ylabel("X Axis (cm)")

        ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
        ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
        tick_labels_x = np.arange(x_min, x_max, 10)
        tick_labels_y = np.arange(y_min, y_max, 10)
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(tick_labels_x)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels(tick_labels_y)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Trajectory Density (log scale)")

    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    safe_name = task_name.replace(" ", "_").lower()
    save_path = os.path.join(comparison_dir, f"{safe_name}_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Saved comparison heatmap to {save_path}")


def run_task_with_variations(
    cfg: HeatmapConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    levels_to_test: list,
):
    """Run a task with all command variations and collect trajectories."""
    
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Get environments and commands for each level
    all_trajectories = {}  # {level: [trajectory1, trajectory2, ...]}
    commands = {}  # {level: command_string}
    results = {}  # {level: {'successes': int, 'episodes': int}}

    for level in levels_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_id}: Testing level {level.upper()}")
        logger.info(f"{'='*60}")

        # Get environment with appropriate command
        if level == 'default':
            env, task_description, original_description = get_libero_env(
                task,
                change_command=False,
                command_level=None,
                resolution=cfg.env_img_res
            )
        else:
            env, task_description, original_description = get_libero_env(
                task,
                change_command=True,
                command_level=level,
                resolution=cfg.env_img_res
            )

        commands[level] = task_description
        logger.info(f"Command: {task_description}")

        trajectories = []
        successes = 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"{level}"):
            initial_state = initial_states[episode_idx]

            success, trajectory, replay_images = run_episode_and_collect_trajectory(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
            )

            trajectories.append(trajectory)
            if success:
                successes += 1

            # Save video if requested
            if cfg.save_videos:
                # Costruisci percorso personalizzato
                video_dir = os.path.join(
                    os.path.dirname(cfg.output_dir),
                    f"heatmap_{cfg.task_suite_name.lower().replace('_', '_')}_seed_{cfg.seed}",
                    'videos',
                    level
                )
                
                save_rollout_video(
                    {'image': replay_images},
                    episode_idx,
                    success=success,
                    task_description=task_description,
                    log_file=None,
                    change_command=(level != 'default'),
                    command_level=level if level != 'default' else None,
                    custom_video_dir=video_dir
                )

        all_trajectories[level] = trajectories
        results[level] = {'successes': successes, 'episodes': cfg.num_trials_per_task}
        
        logger.info(f"Level {level}: {successes}/{cfg.num_trials_per_task} successes")

        env.close()

    return original_description, all_trajectories, commands, results


@draccus.wrap()
def main(cfg: HeatmapConfig):
    """Main function to run inference and generate trajectory heatmaps."""
    
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # Validate and setup
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Initialize model
    logger.info("Initializing model...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine which levels to test
    if cfg.command_levels == "all":
        levels_to_test = COMMAND_LEVELS
    else:
        levels_to_test = [l.strip() for l in cfg.command_levels.split(",")]

    logger.info(f"Testing command levels: {levels_to_test}")

    # Determine which tasks to run
    if cfg.single_task_id is not None:
        # User specified task by ID (0-indexed)
        task_ids = [cfg.single_task_id]
        logger.info(f"Running SINGLE TASK: ID {cfg.single_task_id}")
        
    elif cfg.single_task_name is not None:
        # User specified task by name - find matching task
        task_ids = []
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            if cfg.single_task_name.lower() in task.language.lower():
                task_ids.append(task_id)
        
        if not task_ids:
            raise ValueError(f"No task found matching name: '{cfg.single_task_name}'")
        
        if len(task_ids) > 1:
            logger.warning(f"Multiple tasks match '{cfg.single_task_name}': {task_ids}")
            logger.warning(f"Using first match: task {task_ids[0]}")
            task_ids = [task_ids[0]]
        
        logger.info(f"Running SINGLE TASK: '{cfg.single_task_name}' (ID {task_ids[0]})")
        
    else:
        # Run all tasks
        task_ids = list(range(num_tasks))
        logger.info(f"Running ALL {num_tasks} tasks")

    # Store all results
    all_results = {}
    all_commands = {}

    # Process each task
    for task_id in task_ids:
        logger.info(f"\n{'#'*80}")
        logger.info(f"TASK {task_id + 1}/{num_tasks}")
        logger.info(f"{'#'*80}")

        task_name, all_trajectories, commands, results = run_task_with_variations(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            levels_to_test,
        )

        # Generate individual heatmaps for each level
        for level in levels_to_test:
            if level in all_trajectories:
                generate_heatmap(
                    all_trajectories[level],
                    task_name,
                    commands[level],
                    level,
                    cfg.output_dir
                )

        # Generate comparison heatmap
        generate_comparison_heatmap(all_trajectories, task_name, commands, cfg.output_dir)

        # Store results
        all_results[task_name] = results
        all_commands[task_name] = commands

    # Save summary
    summary = {
        'config': {
            'model': str(cfg.pretrained_checkpoint),
            'task_suite': cfg.task_suite_name,
            'num_trials_per_task': cfg.num_trials_per_task,
            'command_levels': levels_to_test,
            'seed': cfg.seed,
            'single_task_id': cfg.single_task_id,
            'single_task_name': cfg.single_task_name,
        },
        'commands': all_commands,
        'results': all_results,
    }

    summary_path = os.path.join(cfg.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Print final summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Success Rates by Command Level")
    print("=" * 100)
    
    header = f"{'Task':<50}"
    for level in levels_to_test:
        header += f" | {level.upper():>10}"
    print(header)
    print("-" * 100)

    for task_name, task_results in all_results.items():
        row = f"{task_name[:48]:<50}"
        for level in levels_to_test:
            if level in task_results:
                sr = task_results[level]['successes'] / task_results[level]['episodes']
                row += f" | {sr:>9.1%}"
            else:
                row += f" | {'N/A':>10}"
        print(row)

    print("=" * 100)

    logger.info(f"\nAll heatmaps saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
