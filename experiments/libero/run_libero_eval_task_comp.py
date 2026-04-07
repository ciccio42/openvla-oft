"""
run_libero_eval_task_comp.py

Evaluates a trained policy on custom LIBERO task composition scenarios (task_comp_l1).
Tests task-level generalization: the model must apply known primitives to new object/target 
combinations never seen during training.

Custom tasks (all share the libero_goal scene):
  1. Put the plate on the top of the cabinet
  2. Put the plate on the stove
  3. Put the cream cheese on the top of the cabinet
  4. Put the cream cheese on the plate
  5. Open the top layer of the drawer and put the cream cheese inside
"""

import json
import logging
import os
import sys
import gc
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm

from libero.libero import get_libero_path
from libero.libero.benchmark import Task

from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # openvla-oft/
sys.path.insert(0, str(project_root))

from experiments.libero.utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
    extract_command_from_bddl,
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

from libero.libero.envs import OffScreenRenderEnv


# ============================================================================
# Task Composition L1 - Custom Task Definitions
# ============================================================================

# Each custom task maps to an original libero_goal task that shares the same
# manipulated object, so we can reuse init states from that original task.
# All libero_goal tasks share the identical scene layout.

TASK_COMP_L1_TASKS = [
    {
        # Training: push plate (push_plate_to_stove) + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place plate → cabinet (new object-target pair)
        "bddl_file": "put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: push plate (push_plate_to_stove) + put on stove (bowl→stove)
        # Composition: pick-place plate → stove (new object-target pair)
        "bddl_file": "put_the_plate_on_the_stove_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: cream_cheese→bowl + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place cream_cheese → cabinet (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: cream_cheese→bowl + bowl→plate
        # Composition: pick-place cream_cheese → plate (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_the_plate_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: open drawer + bowl inside
        # Composition: open drawer + cream_cheese inside (swaps object, same primitive)
        "bddl_file": "open_the_top_drawer_and_put_the_cream_cheese_inside_task_comp_l1.bddl",
        "init_states_from": "open_the_top_drawer_and_put_the_bowl_inside",
    },
]


# ============================================================================
# Task Composition L2 - Custom Task Definitions
# ============================================================================

TASK_COMP_L2_TASKS = [
    {
        # L2: open MIDDLE drawer + put bowl inside (chain: open + pick-place)
        "bddl_file": "open_the_middle_drawer_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "open_the_middle_drawer_of_the_cabinet",
    },
    {
        # L2: put bowl on stove + turn on stove (chain: pick-place + manipulation)
        "bddl_file": "put_the_bowl_on_the_stove_task_comp_l2.bddl",
        "init_states_from": "put_the_bowl_on_the_stove",
    },
    {
        # L2: put cream cheese in bowl + put bowl on plate (chain: 2 pick-place)
        "bddl_file": "put_the_cream_cheese_in_the_bowl_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # L2: push plate to stove front + put bowl on plate (chain: push + pick-place)
        "bddl_file": "push_the_plate_to_the_front_of_the_stove_task_comp_l2.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # L2: put cream cheese in bowl + put bowl on top of cabinet (chain: 2 pick-place)
        "bddl_file": "put_the_bowl_on_top_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
]

TASK_COMP_REGISTRY = {
    "l1": TASK_COMP_L1_TASKS,
    "l2": TASK_COMP_L2_TASKS,
}


# Define max steps
TASK_MAX_STEPS = 500


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


@dataclass
class GenerateConfig:
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
    # LIBERO environment parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL       # Used for unnorm_key lookup
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 42
    debug: bool = False
    comp_level: str = "l1"

    # Task subset (for splitting across nodes)
    task_start: int = 0
    task_end: int = -1  # -1 means all tasks
    # fmt: on


# ============================================================================
# Helper Functions (reused from run_libero_eval.py)
# ============================================================================

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = get_vla(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head loaded separately")
        except (AssertionError, FileNotFoundError):
            logger.warning("Action head not found as separate file, assuming integrated in model")
            action_head = None

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    logger.info(f"Available norm_stats keys: {list(model.norm_stats.keys())}")
    logger.info(f"cfg.unnorm_key from config: '{cfg.unnorm_key}'")

    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        logger.info(f"Using user-specified unnorm_key: {cfg.unnorm_key}")
        return

    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    run_id = f"EVAL-task_comp_{cfg.comp_level}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        import wandb
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


# ============================================================================
# Custom Task Loading
# ============================================================================

def load_custom_tasks(comp_level: str = "l1"):
    """
    Build Task NamedTuples and load init_states for each task_comp task.

    Returns:
        list of dicts, each with keys:
            - 'task': Task NamedTuple
            - 'init_states': tensor of initial states
            - 'task_description': str from BDDL (:language ...)
            - 'bddl_path': absolute path to BDDL file
    """
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    init_dir = os.path.join(get_libero_path("init_states"), "libero_goal")

    custom_tasks = []
    for task_def in TASK_COMP_REGISTRY[comp_level]:
        bddl_filename = task_def["bddl_file"]
        init_from = task_def["init_states_from"]

        bddl_path = os.path.join(bddl_dir, bddl_filename)
        assert os.path.exists(bddl_path), f"BDDL file not found: {bddl_path}"

        # Extract language from BDDL
        task_description = extract_command_from_bddl(bddl_path)
        assert task_description is not None, f"Could not extract language from {bddl_path}"

        # Build Task NamedTuple
        task_name = bddl_filename.replace(".bddl", "")
        task = Task(
            name=task_name,
            language=task_description,
            problem="Libero",
            problem_folder="libero_goal",
            bddl_file=bddl_filename,
            init_states_file=f"{init_from}.pruned_init",
        )

        # Load init states from the mapped original task
        init_states_path = os.path.join(init_dir, f"{init_from}.pruned_init")
        assert os.path.exists(init_states_path), f"Init states not found: {init_states_path}"
        init_states = torch.load(init_states_path, weights_only=False)

        custom_tasks.append({
            "task": task,
            "init_states": init_states,
            "task_description": task_description,
            "bddl_path": bddl_path,
        })

    return custom_tasks


def create_env_from_bddl(bddl_path, resolution=256):
    """Create LIBERO environment directly from a BDDL file path."""
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


# ============================================================================
# Episode & Task Execution
# ============================================================================

def run_episode(
    cfg, env, task_description, model, resize_size,
    processor=None, action_head=None, proprio_projector=None,
    noisy_action_projector=None, initial_state=None, log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) != NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    replay_states = []
    max_steps = TASK_MAX_STEPS

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            replay_states.append(obs["robot0_eef_pos"])

            if len(action_queue) == 0:
                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                if t < 50:
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, replay_states


def run_custom_task(
    cfg, task_info, task_idx, num_tasks, model, resize_size,
    processor=None, action_head=None, proprio_projector=None,
    noisy_action_projector=None, total_episodes=0, total_successes=0,
    log_file=None,
):
    """Run evaluation for a single custom task_comp_l1 task."""
    task = task_info["task"]
    init_states = task_info["init_states"]
    task_description = task_info["task_description"]
    bddl_path = task_info["bddl_path"]

    # Create environment from custom BDDL
    env = create_env_from_bddl(bddl_path, resolution=cfg.env_img_res)

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_idx + 1}/{num_tasks} (Task Composition L1)", log_file)
    log_message(f"BDDL: {task.bddl_file}", log_file)
    log_message(f"Command: {task_description}", log_file)
    log_message(f"Init states from: {task.init_states_file}", log_file)
    log_message("=" * 80, log_file)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        initial_state = init_states[episode_idx]

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images, replay_states = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector,
            noisy_action_projector, initial_state, log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            {'image': replay_images, 'states': replay_states},
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            dataset_name="task_comp_l1",
            run=cfg.run_id_note,
            custom_video_dir=f"/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/task_composition/openvla-oft/task_comp_{cfg.comp_level}/run_{cfg.run_id_note}",
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)", log_file)
    log_message(f"Running total success rate: {total_success_rate:.4f}", log_file)

    if cfg.use_wandb:
        import wandb
        wandb.log({
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })

    # Cleanup
    try:
        env.close()
        log_message("Environment closed successfully", log_file)
    except Exception as e:
        log_message(f"Warning: Error closing environment: {e}", log_file)
    gc.collect()

    return total_episodes, total_successes, task_description, task_success_rate, task_episodes


# ============================================================================
# Results Printing
# ============================================================================

def print_results_table(task_results, all_results):
    """Print a summary table of task composition L1 results."""
    print("\n" + "=" * 100)
    print("TASK COMPOSITION L1 - RESULTS TABLE")
    print("=" * 100)

    print(f"{'Task':<60} | {'Success Rate':>20} | {'Episodes':>8}")
    print("-" * 100)

    for task_name, result in task_results.items():
        sr = result['success_rate']
        eps = result['episodes']
        successes = int(sr * eps)
        print(f"{task_name:<60} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")

    print("-" * 100)
    overall_sr = all_results['success_rate']
    overall_succ = all_results['total_successes']
    overall_eps = all_results['total_episodes']
    print(f"{'OVERALL':<60} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}")
    print("=" * 100)


# ============================================================================
# Main Entry Point
# ============================================================================

@draccus.wrap()
def eval_task_comp(cfg: GenerateConfig) -> float:
    """Evaluate trained policy on task composition L1 scenarios."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # Validate
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set seed
    set_seed_everywhere(cfg.seed)

    # Initialize model
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Load custom tasks
    all_custom_tasks = load_custom_tasks(cfg.comp_level)
    total_num_tasks = len(all_custom_tasks)

    # Select task subset
    task_end = cfg.task_end if cfg.task_end >= 0 else total_num_tasks
    task_start = cfg.task_start
    custom_tasks = all_custom_tasks[task_start:task_end]
    num_tasks = len(custom_tasks)

    log_message(f"Loaded {total_num_tasks} total task composition {cfg.comp_level.upper()} tasks", None)
    log_message(f"Running task subset [{task_start}:{task_end}] ({num_tasks} tasks)", None)
    for i, ct in enumerate(custom_tasks):
        log_message(f"  [{task_start + i}] {ct['task_description']} ({ct['task'].bddl_file})", None)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    log_message("=" * 80, log_file)
    log_message(f"TASK COMPOSITION {cfg.comp_level.upper()} EVALUATION", log_file)
    log_message(f"Model: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Num tasks: {num_tasks}", log_file)
    log_message("=" * 80, log_file)

    # Run evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}

    for task_idx in tqdm.tqdm(range(num_tasks), desc=f"Task Comp {cfg.comp_level.upper()}"):
        total_episodes, total_successes, task_name, task_sr, task_eps = run_custom_task(
            cfg, custom_tasks[task_idx], task_idx, num_tasks,
            model, resize_size, processor, action_head,
            proprio_projector, noisy_action_projector,
            total_episodes, total_successes, log_file,
        )

        task_results[task_name] = {
            'success_rate': task_sr,
            'episodes': task_eps,
        }

    # Final results
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    all_results = {
        'success_rate': final_success_rate,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
    }

    log_message("=" * 80, log_file)
    log_message("FINAL RESULTS - TASK COMPOSITION L1:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message("=" * 80, log_file)

    if cfg.use_wandb:
        import wandb
        wandb.log({
            "success_rate/task_comp_l1_overall": final_success_rate,
            "num_episodes/task_comp_l1_overall": total_episodes,
        })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    # Print results table
    print_results_table(task_results, all_results)

    return final_success_rate


if __name__ == "__main__":
    eval_task_comp()
