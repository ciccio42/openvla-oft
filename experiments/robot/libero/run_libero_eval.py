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
import glob
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
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
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
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
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    run_number: int = 0                                  # Run number (for logging purposes)
    debug: bool = False  
    change_spawn: bool = True  # Whether to change spawn region of target object in the environment
    spawn_train_distribution: bool = False  # Whether to use the training spawn distribution for the target object
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
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

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
    # Initialize unnorm_key
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
    def _resample_target_xy_within_region_once(env):
        """Force a fresh target XY sample inside floor_target_object_region for this episode."""
        parsed = getattr(env.env, "parsed_problem", None)
        if not parsed:
            return
        regions = parsed.get("regions", {})
        target_region = regions.get("floor_target_object_region", {})
        ranges = target_region.get("ranges", [])
        if not ranges:
            return
        # LIBERO stores one rectangle as [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = ranges[0]
        x = np.random.uniform(min(xmin, xmax), max(xmin, xmax))
        y = np.random.uniform(min(ymin, ymax), max(ymin, ymax))

        target_name = None
        for obj_name in getattr(env, "obj_of_interest", []):
            if "basket_1" not in obj_name:
                target_name = obj_name
                break
        if target_name is None:
            return
        target_obj = env.env.objects_dict.get(target_name, None)
        if target_obj is None or not getattr(target_obj, "joints", None):
            return

        qpos = env.sim.data.get_joint_qpos(target_obj.joints[0]).copy()
        if len(qpos) < 7:
            return
        qpos[0] = x
        qpos[1] = y
        env.sim.data.set_joint_qpos(target_obj.joints[0], qpos)
        env.sim.forward()

    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        # env.set_init_state(initial_state)
        obs =  env.reset() #env.get_observation()
        if cfg.change_spawn:
            _resample_target_xy_within_region_once(env)
            obs = env.env._get_observations()
        # fix the position of the bin
        # due to problem with the initialization of the environment
        # For test with different spawn regions this is not a problem
        if 'basket_1_pos' in obs.keys():
            basket_pos = [0.005, 0.261, 0.035]
            basket_quat = [0.000, 0.000, 0.000, 1.000]  # [x, y, z, w]
            # env.sim.data.set_joint_qpos()
            env.sim.data.set_joint_qpos(env.env.objects_dict['basket_1'].joints[0], 
                                        np.concatenate((basket_pos, basket_quat)))
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
            
    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_traj = dict()
    replay_traj['image'] = []
    replay_traj['task_command'] = task_description
    replay_traj['actions'] = []
    replay_traj['states'] = []
    replay_traj['target_object_positions'] = []
    replay_traj['target_object_name'] = []
    replay_traj['bin_position'] = []
    
    
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

            if t == cfg.num_steps_wait:
                # Capture stabilized spawn information once, before policy actions start.
                if 'basket_1_pos' in obs:
                    basket_pos = obs['basket_1_pos']
                    log_message(f"Basket position after stabilization: {basket_pos}", log_file)
                    replay_traj['bin_position'].append(list(basket_pos))

                for obj_name in getattr(env, "obj_of_interest", []):
                    if 'basket_1' in obj_name:
                        continue
                    key = f"{obj_name}_pos"
                    if key in obs:
                        obj_pos = obs[key]
                        log_message(f"Target object {obj_name} position after stabilization: {obj_pos}", log_file)
                        replay_traj['target_object_positions'].append(list(obj_pos))
                        replay_traj['target_object_name'].append(obj_name)
                        break

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_traj['image'].append(img)
            
            # Append state to replay trajectory
            replay_traj['states'].append(observation['state'].tolist())

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            replay_traj['actions'].append(action.tolist())
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            pil_img = Image.fromarray(obs['agentview_image'])
            pil_img.save(os.path.join(cfg.local_log_dir, f"step.png"))
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_traj

def _collect_spawn_positions_from_obs(obs, env):
    """Extract bin and target-object positions from observation after stabilization."""
    bin_position = None
    target_positions = []
    target_names = []

    if isinstance(obs, dict):
        if "basket_1_pos" in obs:
            bin_position = np.asarray(obs["basket_1_pos"], dtype=np.float32).reshape(-1)[:3].tolist()

        obj_of_interest = getattr(env, "obj_of_interest", [])
        for obj_name in obj_of_interest:
            if "basket_1" in obj_name:
                continue
            key = f"{obj_name}_pos"
            if key in obs:
                pos = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                if pos.shape[0] >= 3:
                    target_positions.append(pos[:3].tolist())
                    target_names.append(obj_name)
                    break

    return {
        "bin_position": [bin_position] if bin_position is not None else [],
        "target_object_positions": target_positions,
        "target_object_name": target_names,
    }


def enrich_episode_with_spawn_info(
    episode_file,
    env,
    cfg: GenerateConfig,
    initial_states,
    all_initial_states,
    task_description: str,
    log_file=None,
):
    """Enrich one saved rollout .npy with bin/target object spawn positions."""
    npy_path = str(episode_file)
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        log_message(f"Failed to load rollout for enrichment: {npy_path} ({e})", log_file)
        return

    # Parse episode id from filename to replay matching init-state setup.
    try:
        episode_idx = int(os.path.basename(npy_path).split("episode=")[-1].split("--")[0])
    except Exception:
        log_message(f"Could not parse episode id from filename, skipping enrichment: {npy_path}", log_file)
        return

    # If already present, keep existing values unless explicitly empty.
    has_bin = "bin_position" in data and data["bin_position"] is not None and len(data["bin_position"]) > 0
    has_target = (
        "target_object_positions" in data
        and data["target_object_positions"] is not None
        and len(data["target_object_positions"]) > 0
    )
    if has_bin and has_target:
        return

    # Match initialization path used during rollout.
    if cfg.initial_states_path == "DEFAULT":
        initial_state = initial_states[episode_idx]
    else:
        initial_states_task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{episode_idx}"
        try:
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(
                    f"Skipping enrichment for episode {episode_idx}: failed expert demo in initial states.",
                    log_file,
                )
                return
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
        except Exception as e:
            log_message(f"Initial-state lookup failed for episode {episode_idx}: {e}", log_file)
            return

    if cfg.change_spawn:
        initial_state = None

    # Recreate scene and wait for stabilization.
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    enrich = _collect_spawn_positions_from_obs(obs, env)
    data["bin_position"] = enrich["bin_position"]
    data["target_object_positions"] = enrich["target_object_positions"]
    data["target_object_name"] = enrich["target_object_name"]
    np.save(npy_path, data)
    log_message(f"Enriched rollout with spawn info: {npy_path}", log_file)


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
    # load default initial states
    
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    
    # Initialize environment and get task description
    env, task_description = get_libero_env(task, 
                                           cfg.model_family, 
                                           resolution=cfg.env_img_res,
                                           change_spawn=cfg.change_spawn,
                                           train_spawn_distribution=cfg.spawn_train_distribution,
                                           env_seed=cfg.seed + task_id)
    
    # get the episode already recorded in the environment
    rollout_dir = f"./rollouts/{cfg.task_suite_name}/change_spawn_{cfg.change_spawn}_train_{cfg.spawn_train_distribution}/run_{cfg.run_number}"
    episode_full_list = glob.glob(os.path.join(rollout_dir, "*.npy"))
    completed_episode_ids = set()
    for episode in episode_full_list:
        try:
            completed_episode_ids.add(int(episode.split("episode=")[-1].split("--")[0]))
            
            # enrich current npy with target object and bin positions
            enrich_episode_with_spawn_info(
                episode_file=episode,
                env=env,
                cfg=cfg,
                initial_states=initial_states,
                all_initial_states=all_initial_states,
                task_description=task_description,
                log_file=log_file,
            )
            
        except Exception:
            # Ignore malformed filenames and keep evaluating.
            continue
    
    
    
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        if total_episodes in completed_episode_ids:
            log_message(f"Skipping episode {total_episodes} as it already exists in {rollout_dir}", log_file)
            total_episodes += 1
            continue

        # When spawn randomization is enabled, rebuild env per episode so a new BDDL spawn
        # configuration is sampled each rollout (instead of once per task).
        if cfg.change_spawn:
            episode_env_seed = cfg.seed + (task_id * 100000) + episode_idx
            try:
                env.close()
            except Exception:
                pass
            env, task_description = get_libero_env(
                task,
                cfg.model_family,
                resolution=cfg.env_img_res,
                change_spawn=cfg.change_spawn,
                train_spawn_distribution=cfg.spawn_train_distribution,
                env_seed=episode_env_seed,
            )
            log_message(f"Rebuilt env with seed={episode_env_seed} for randomized spawn.", log_file)
        
        log_message(f"\nTask: {task_description}", log_file)
        # if episode_idx < len_episode_full_list:
        #     # If the episode already exists, skip it
        #     log_message(f"Skipping episode {episode_idx} as it already exists in {rollout_dir}", log_file)
        #     continue
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
        if cfg.change_spawn:
            log_message("Setting initial state with changed spawn region...", log_file)
            initial_state, all_initial_state = None, None

        
        success, replay_traj = run_episode(
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
        replay_traj['original_task_name'] = task.name
        replay_traj['task_name'] = task.name

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_traj, 
            total_episodes, 
            success=success, 
            task_description=task_description, 
            log_file=log_file,
            dataset_name=cfg.task_suite_name,
            run=cfg.run_number, 
            change_spawn=cfg.change_spawn,
            train_spawn_distribution=cfg.spawn_train_distribution
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    try:
        env.close()
    except Exception:
        pass

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

    return total_episodes, total_successes


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

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
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

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
