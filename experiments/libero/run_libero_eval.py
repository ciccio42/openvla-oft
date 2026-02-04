"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
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
project_root = current_file.parent.parent.parent  # Va a openvla-oft/
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


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 200,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    
    #################################################################################################################
    # Command variation parameters
    #################################################################################################################
    change_command: bool = False                     # Use synonym command variations
    command_level: Optional[str] = None              # Command level: 'l1', 'l2', 'l3', 'all', or None
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                    # Random Seed (for reproducibility)

    debug: bool = False  
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load model
    model = get_vla(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed - RESO OPZIONALE per checkpoint con LoRA
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head caricato separatamente")
        except (AssertionError, FileNotFoundError) as e:
            logger.warning("Action head non trovato come file separato")
            logger.warning("Assumo sia integrato nel modello (checkpoint LoRA)")
            action_head = None

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Debug: print available keys and current unnorm_key
    logger.info(f"Available norm_stats keys: {list(model.norm_stats.keys())}")
    logger.info(f"cfg.unnorm_key from config: '{cfg.unnorm_key}'")
    
    # Use the unnorm_key from config if already specified, otherwise use task_suite_name
    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        # User specified a valid unnorm_key, use it
        logger.info(f"Using user-specified unnorm_key: {cfg.unnorm_key}")
        return
    
    # Initialize unnorm_key from task_suite_name
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    # Add command level to run_id if specified
    if cfg.change_command and cfg.command_level:
        run_id += f"--{cfg.command_level}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
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
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
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
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
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
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_states = []  # Store end-effector positions
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
            # Store end-effector position (x, y, z)
            replay_states.append(obs["robot0_eef_pos"])

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
                # DEBUG: Print first action to check if actions are varying
                if t < 50:  # Only print first few steps
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")  # Print first 4 dims
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

    return success, replay_images, replay_states

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description, original_description = get_libero_env(
        task, 
        change_command=cfg.change_command,
        command_level=cfg.command_level,
        resolution=cfg.env_img_res
    )
    
    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id + 1}/{task_suite.n_tasks}", log_file)
    log_message(f"Original Command: {original_description}", log_file)
    
    if cfg.change_command and cfg.command_level:
        log_message(f"Command Level: {cfg.command_level.upper()}", log_file)
        log_message(f"Variation Command: {task_description}", log_file)
        if task_description == original_description:
            log_message(" WARNING: Variation same as original - check BDDL file", log_file)
    else:
        log_message(f"Command Level: DEFAULT", log_file)
    
    log_message("=" * 80, log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, replay_states = run_episode(
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
            log_file,
        )

        # Update counters
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
            change_command=cfg.change_command,
            command_level=cfg.command_level,
            run=27012026
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, task_description, task_success_rate, task_episodes

def print_results_table(task_results, command_levels, all_results):
    """Print a summary table of results by task and command level."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)
    
    # Get all task names (from first level)
    first_level = command_levels[0]
    level_name = first_level if first_level is not None else "default"
    task_names = list(task_results[level_name].keys())
    
    # Prepare column headers
    level_names = [l if l is not None else "default" for l in command_levels]
    
    if len(level_names) == 1:
        # Single level: show task name and success rate
        print(f"{'Task':<50} | {'Success Rate':>12} | {'Episodes':>8}")
        print("-" * 100)
        
        for task_name in task_names:
            result = task_results[level_names[0]][task_name]
            sr = result['success_rate']
            eps = result['episodes']
            print(f"{task_name:<50} | {sr:>11.1%} | {eps:>8}")
        
        print("-" * 100)
        overall_sr = all_results[level_names[0]]['success_rate']
        overall_eps = all_results[level_names[0]]['total_episodes']
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} | {overall_eps:>8}")
    
    else:
        # Multiple levels: show task name and success rate for each level
        header = f"{'Task':<40}"
        for level_name in level_names:
            header += f" | {level_name.upper():>12}"
        print(header)
        print("-" * (41 + len(level_names) * 16))
        
        for task_name in task_names:
            row = f"{task_name:<40}"
            for level_name in level_names:
                if task_name in task_results[level_name]:
                    sr = task_results[level_name][task_name]['success_rate']
                    row += f" | {sr:>11.1%}"
                else:
                    row += f" | {'N/A':>12}"
            print(row)
        
        print("-" * (41 + len(level_names) * 16))
        
        # Overall row
        overall_row = f"{'OVERALL':<40}"
        for level_name in level_names:
            sr = all_results[level_name]['success_rate']
            overall_row += f" | {sr:>11.1%}"
        print(overall_row)
    
    print("=" * 100)
    
    # Summary statistics
    print("\nSUMMARY BY COMMAND LEVEL:")
    print("-" * 60)
    for level_name in level_names:
        result = all_results[level_name]
        sr = result['success_rate']
        succ = result['total_successes']
        total = result['total_episodes']
        print(f"  {level_name.upper():>15}: {sr:.1%} ({succ}/{total} episodes)")
    print("=" * 100)

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine which command levels to test
    if cfg.change_command and cfg.command_level == "all":
        command_levels = [None, "l1", "l2", "l3"]  
        log_message_prefix = "Testing all command levels (default, l1, l2, l3)"
    elif cfg.change_command and cfg.command_level == "all_no_default":
        command_levels = ["l1", "l2", "l3"]  
        log_message_prefix = "Testing command levels (l1, l2, l3) without default"
    elif cfg.change_command and cfg.command_level == "default":
        command_levels = [None]  
        log_message_prefix = "Testing default command only"
    elif cfg.change_command and cfg.command_level is not None:
        command_levels = [cfg.command_level]
        log_message_prefix = f"Testing command level: {cfg.command_level}"
    else:
        command_levels = [None]  # Default only
        log_message_prefix = "Testing with default commands"

    # Store results for all levels
    all_results = {}
    task_results = {}  # Dizionario per risultati per task

    # Loop over command levels
    for level in command_levels:
        # Update config for current level
        current_level_name = level if level is not None else "default"
        cfg.command_level = level
        cfg.change_command = (level is not None)
        
        # Setup logging for this level
        log_file, local_log_filepath, run_id = setup_logging(cfg)
        
        log_message("=" * 80, log_file)
        log_message(f"EVALUATING: {current_level_name.upper()}", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Task suite: {cfg.task_suite_name}", log_file)
        log_message(log_message_prefix, log_file)

        # Inizializza dizionario per questo livello
        task_results[current_level_name] = {}

        # Start evaluation for this level
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks), desc=f"Level {current_level_name}"):
            # Cattura anche i risultati per task
            total_episodes, total_successes, task_name, task_sr, task_eps = run_task(
                cfg,
                task_suite,
                task_id,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
            )
            
            # Salva risultati per task
            task_results[current_level_name][task_name] = {
                'success_rate': task_sr,
                'episodes': task_eps
            }

        # Calculate final success rate for this level
        final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        all_results[current_level_name] = {
            'success_rate': final_success_rate,
            'total_episodes': total_episodes,
            'total_successes': total_successes
        }

        # Log final results for this level
        log_message("=" * 80, log_file)
        log_message(f"RESULTS FOR {current_level_name.upper()}:", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Total episodes: {total_episodes}", log_file)
        log_message(f"Total successes: {total_successes}", log_file)
        log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
        log_message("=" * 80, log_file)

        # Log to wandb if enabled
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{current_level_name}": final_success_rate,
                    f"num_episodes/{current_level_name}": total_episodes,
                }
            )
            wandb.save(local_log_filepath)

        # Close log file for this level
        if log_file:
            log_file.close()

    # Genera e stampa tabella riepilogativa
    print_results_table(task_results, command_levels, all_results)

    # Return the success rate of the last tested level (or average if all)
    if len(command_levels) > 1:
        avg_success_rate = sum(r['success_rate'] for r in all_results.values()) / len(all_results)
        return avg_success_rate
    else:
        return final_success_rate


if __name__ == "__main__":
    eval_libero()