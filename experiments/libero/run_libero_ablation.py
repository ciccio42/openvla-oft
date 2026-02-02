"""
run_libero_ablation.py

Ablation study for LIBERO tasks with keyword-only commands.
Supports multiple tasks via --ablation_task_id parameter.
Automatically loads custom BDDL files if available.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Task suite constants
class TaskSuite(str, Enum):
    LIBERO_GOAL = "libero_goal"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_GOAL: 200,
}


def get_ablation_tasks(task_id: int) -> dict:
    """Get ablation tasks configuration for specific task ID."""
    
    ABLATION_CONFIGS = {
        # Task 8: "Turn on the stove" (task_id=7, 0-indexed)
        7: {
            "task_name": "Turn on the stove",
            "tests": {
                "stove1": {
                    "bddl_file": "turn_on_the_stove_ablation_stove1.bddl",
                    "expected_command": "stove"
                },
                "stove2": {
                    "bddl_file": "turn_on_the_stove_ablation_stove2.bddl",
                    "expected_command": "bowl stove"
                },
                "stove3": {
                    "bddl_file": "turn_on_the_stove_ablation_stove3.bddl",
                    "expected_command": "plate stove"
                },
                "stove4": {
                    "bddl_file": "turn_on_the_stove_ablation_stove4.bddl",
                    "expected_command": "Turn on"
                },
            }
        },
        
        # Task 9: "Put the bowl on the plate" (task_id=8, 0-indexed)
        8: {
            "task_name": "Put the bowl on the plate",
            "tests": {
                "bowl_plate1": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate1.bddl",
                    "expected_command": "bowl"
                },
                "bowl_plate2": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate2.bddl",
                    "expected_command": "plate"
                },
                "bowl_plate3": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate3.bddl",
                    "expected_command": "bowl plate"
                },
                "bowl_plate4": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate4.bddl",
                    "expected_command": "Put plate"
                },
                "bowl_plate5": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate5.bddl",
                    "expected_command": "plate bowl"
                },
            }
        },
    }
    
    if task_id not in ABLATION_CONFIGS:
        raise ValueError(f"No ablation configuration found for task_id={task_id}. "
                        f"Available task IDs: {list(ABLATION_CONFIGS.keys())}")
    
    return ABLATION_CONFIGS[task_id]


@dataclass
class GenerateConfig:
    # fmt: off
    
    #################################################################################################################
    # Ablation-specific parameters
    #################################################################################################################
    ablation_task_id: int = 7  # Task ID to run ablation on (0-indexed, default=7 for Task 8)
    
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
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
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
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name == TaskSuite.LIBERO_GOAL, "Ablation only works with libero_goal!"
    
    # Validate that ablation config exists for this task
    try:
        get_ablation_tasks(cfg.ablation_task_id)
    except ValueError as e:
        raise ValueError(f"Invalid ablation_task_id: {e}")


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = get_vla(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head caricato separatamente")
        except (AssertionError, FileNotFoundError) as e:
            logger.warning("Action head non trovato come file separato")
            logger.warning("Assumo sia integrato nel modello (checkpoint LoRA)")
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

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig, task_name: str):
    """Set up logging to file and optionally to wandb."""
    safe_task_name = task_name.replace(" ", "_").lower()
    run_id = f"ABLATION-Task{cfg.ablation_task_id+1}-{safe_task_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


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


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    
    # Reset environment
    env.reset()
    
    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               f"{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
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
                if t < 50:
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_ablation_task(
    cfg: GenerateConfig,
    task_key: str,
    task_info: dict,
    task_suite,
    task,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run evaluation for a single ablation task."""
    task_id = cfg.ablation_task_id
    
    log_message("=" * 80, log_file)
    log_message(f"ABLATION TASK: {task_key.upper()}", log_file)
    log_message(f"BDDL File: {task_info['bddl_file']}", log_file)
    log_message("=" * 80, log_file)
    
    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment with ABLATION BDDL (if exists)
    # Pass ablation_bddl_file to get_libero_env to load custom BDDL
    env, task_description, original_description = get_libero_env(
        task,
        ablation_bddl_file=task_info['bddl_file'],
        resolution=cfg.env_img_res
    )
    
    # Use command extracted from BDDL (or fallback to config if BDDL not found)
    ablation_command = task_description
    
    log_message(f"Original Task {task_id+1} Command: {original_description}", log_file)
    log_message(f"Ablation Command (from BDDL): '{ablation_command}'", log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Ablation {task_key}"):

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = original_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode with ABLATION COMMAND
        success, replay_images = run_episode(
            cfg,
            env,
            ablation_command,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        if success:
            task_successes += 1

        # Save replay video
        save_rollout_video(
            {'image': replay_images},
            task_episodes,
            success=success,
            task_description=f"ablation_task{task_id+1}_{task_key}_{ablation_command.replace(' ', '_')}",
            log_file=log_file,
            change_command=True,
            command_level="ablation"
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {task_episodes}", log_file)
        log_message(f"# successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log({
            f"success_rate/ablation_{task_key}": task_success_rate,
            f"num_episodes/ablation_{task_key}": task_episodes,
        })

    return task_success_rate, task_episodes, task_successes


def print_ablation_results(results, task_name: str, ablation_tests: dict):
    """Print summary table."""
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY RESULTS - Task: {task_name}")
    print("=" * 100)
    print(f"{'Test':<20} | {'Command':<20} | {'Success Rate':>12} | {'Episodes':>15}")
    print("-" * 100)
    
    for task_key, result in results.items():
        cmd = ablation_tests[task_key]['expected_command']
        sr = result['success_rate']
        succ = result['successes']
        total = result['episodes']
        print(f"{task_key:<20} | {cmd:<20} | {sr:>11.1%} | {succ:>6}/{total:<7}")
    
    print("=" * 100)
    
    # Average
    avg_sr = sum(r['success_rate'] for r in results.values()) / len(results)
    total_succ = sum(r['successes'] for r in results.values())
    total_eps = sum(r['episodes'] for r in results.values())
    print(f"{'AVERAGE':<20} | {'':<20} | {avg_sr:>11.1%} | {total_succ:>6}/{total_eps:<7}")
    print("=" * 100)


@draccus.wrap()
def eval_ablation(cfg: GenerateConfig) -> float:
    """Main function for ablation study."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Get ablation configuration for specified task
    ablation_config = get_ablation_tasks(cfg.ablation_task_id)
    task_name = ablation_config['task_name']
    ablation_tests = ablation_config['tests']

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    # Get task
    task = task_suite.get_task(cfg.ablation_task_id)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_name)
    
    log_message("=" * 80, log_file)
    log_message(f"ABLATION STUDY: Task {cfg.ablation_task_id+1} Keyword Shortcut Analysis", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Task Name: {task_name}", log_file)
    log_message(f"Base Command: {task.language}", log_file)
    log_message(f"Model: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Trials per ablation: {cfg.num_trials_per_task}", log_file)
    log_message(f"Ablation tests: {list(ablation_tests.keys())}", log_file)
    log_message("=" * 80, log_file)

    # Run all ablation tasks
    results = {}
    for task_key, task_info in ablation_tests.items():
        sr, episodes, successes = run_ablation_task(
            cfg,
            task_key,
            task_info,
            task_suite,
            task,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )
        
        results[task_key] = {
            'success_rate': sr,
            'episodes': episodes,
            'successes': successes
        }

    # Print summary
    print_ablation_results(results, task_name, ablation_tests)
    
    # Log final results
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    for task_key, result in results.items():
        cmd = ablation_tests[task_key]['expected_command']
        log_message(f"{task_key} ('{cmd}'): {result['success_rate']:.1%} ({result['successes']}/{result['episodes']})", log_file)
    avg_sr = sum(r['success_rate'] for r in results.values()) / len(results)
    log_message(f"\nAVERAGE: {avg_sr:.1%}", log_file)
    log_message("=" * 80, log_file)

    if cfg.use_wandb:
        wandb.log({"average_success_rate": avg_sr})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return avg_sr


if __name__ == "__main__":
    eval_ablation()