"""
extract_embeddings_rollout.py

Extract multimodal embeddings from OpenVLA model during REAL inference rollouts.
Uses predict_action() to execute real actions and extracts embeddings at each step.

For each task and command level:
- Runs N rollout episodes with the model predicting real actions
- Extracts text-conditioned embeddings at each step
- Computes mean embedding per episode and overall mean across episodes
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Setup paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.openvla_utils import (
    get_processor,
    get_vla,
    get_action_head,
    get_proprio_projector,
    get_noisy_action_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.libero.libero_utils import (
    extract_command_from_bddl, 
    get_libero_path, 
    get_libero_env, 
    get_libero_image,
    get_libero_wrist_image,
    get_libero_dummy_action,
    quat2axisangle,
)
from experiments.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from libero.libero import benchmark
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Max steps per task suite
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 200,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction during rollout."""
    # Model parameters
    pretrained_checkpoint: str = ""
    model_family: str = "openvla"
    
    # Action head parameters
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    
    # Inference parameters
    center_crop: bool = True
    num_open_loop_steps: int = 8
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Task parameters
    task_suite_name: str = "libero_goal"
    unnorm_key: str = ""
    
    # Environment parameters
    env_img_res: int = 256
    num_steps_wait: int = 10


def load_model_and_components(cfg: EmbeddingConfig):
    """Load model and all inference components."""
    print(f"Loading model from: {cfg.pretrained_checkpoint}")
    
    # Load VLA model
    model = get_vla(cfg)
    
    # Set unnorm_key
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found!"
    cfg.unnorm_key = unnorm_key
    
    # Load processor
    processor = get_processor(cfg)
    
    # Load proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    
    # Load action head
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            print("✓ Action head loaded")
        except (AssertionError, FileNotFoundError):
            print("⚠️ Action head not found, assuming integrated in model")
    
    # Load noisy action projector for diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
    
    # Get resize size for policy input
    resize_size = get_image_resize_size(cfg)
    
    return model, processor, action_head, proprio_projector, noisy_action_projector, resize_size


def build_bddl_path(task, level: str) -> str:
    """Ritorna il path del BDDL per default o per la variazione syn_lX."""
    if level == "default":
        return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    base_name = task.bddl_file.replace(".bddl", "")
    bddl_filename = f"{base_name}_syn_{level}.bddl"
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, bddl_filename)


def build_prompt(task_label: str) -> str:
    """Prompt coerente con la policy OpenVLA."""
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def extract_embedding_from_forward(
    model,
    processor,
    prompt: str,
    primary_image: Image.Image,
    wrist_image: Optional[Image.Image] = None,
):
    """
    Extract text-conditioned embedding using model's forward pass.
    
    Uses the same input processing as predict_action() but extracts
    the text embeddings from the language model hidden states.
    
    Returns:
        embedding: (hidden_dim,) - pooled text embedding
    """
    with torch.no_grad():
        # Process primary image
        inputs = processor(prompt, primary_image).to(model.device, dtype=torch.bfloat16)
        
        # Add wrist image if provided (multi-image input)
        if wrist_image is not None:
            wrist_inputs = processor(prompt, wrist_image).to(model.device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )
        
        # === Forward through model components ===
        
        # 1. Vision backbone
        pixel_values = inputs["pixel_values"]
        patch_embeddings = model.vision_backbone(pixel_values)
        
        # 2. Projector
        projected_patches = model.projector(patch_embeddings)
        n_patches = projected_patches.shape[1]
        
        # 3. Text embeddings
        input_ids = inputs["input_ids"]
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        
        # 4. Build multimodal sequence: [BOS] + [patches] + [text_rest]
        bos_embed = text_embeds[:, :1, :]
        text_rest = text_embeds[:, 1:, :]
        multimodal_embeds = torch.cat([bos_embed, projected_patches, text_rest], dim=1)
        
        # 5. Extended attention mask
        attention_mask_text = inputs["attention_mask"]
        patch_mask = torch.ones(
            (attention_mask_text.shape[0], n_patches),
            dtype=attention_mask_text.dtype,
            device=attention_mask_text.device
        )
        attention_mask_multimodal = torch.cat(
            [attention_mask_text[:, :1], patch_mask, attention_mask_text[:, 1:]],
            dim=1
        )
        
        # 6. Forward through language model
        lm_outputs = model.language_model(
            inputs_embeds=multimodal_embeds,
            attention_mask=attention_mask_multimodal,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 7. Extract last layer hidden states
        last_hidden = lm_outputs.hidden_states[-1]
        
        # 8. Extract only text tokens (excluding patches)
        text_hidden = torch.cat([
            last_hidden[:, :1, :],              # BOS
            last_hidden[:, 1 + n_patches:, :],  # text tokens
        ], dim=1)
        
        # 9. Mean pooling over text tokens
        mask = attention_mask_text.unsqueeze(-1).to(text_hidden.dtype)
        masked = text_hidden * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return pooled.squeeze(0).detach().cpu().float().numpy()


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
    
    return observation, img, wrist_img


def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def extract_first_step_embedding(
    cfg: EmbeddingConfig,
    env,
    prompt: str,
    model,
    processor,
    resize_size,
    initial_state=None,
):
    """
    Extract embedding from the first observation only (no rollout).
    
    Returns:
        embedding: Single embedding from first step
    """
    # Reset environment
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    # Wait for objects to stabilize
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    
    # Prepare observation
    observation, img, wrist_img = prepare_observation(obs, resize_size)
    
    # Convert to PIL for embedding extraction
    img_pil = Image.fromarray(img).convert("RGB")
    wrist_pil = Image.fromarray(wrist_img).convert("RGB") if cfg.num_images_in_input > 1 else None
    
    # Extract embedding
    embedding = extract_embedding_from_forward(
        model, processor, prompt, img_pil, wrist_pil
    )
    
    return embedding


def run_single_episode(
    cfg: EmbeddingConfig,
    env,
    task_description: str,
    prompt: str,
    model,
    processor,
    resize_size,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    max_steps: int = 300,
):
    """
    Run a single episode with real actions and extract embeddings.
    
    Returns:
        embeddings: List of embeddings, one per step
        success: Whether the episode was successful
        num_steps: Number of steps executed
    """
    # Reset environment
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    # Action queue for open-loop execution
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    
    embeddings = []
    t = 0
    success = False
    
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Wait for objects to stabilize
            if t < cfg.num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            # Prepare observation
            observation, img, wrist_img = prepare_observation(obs, resize_size)
            
            # Convert to PIL for embedding extraction
            img_pil = Image.fromarray(img).convert("RGB")
            wrist_pil = Image.fromarray(wrist_img).convert("RGB") if cfg.num_images_in_input > 1 else None
            
            # Extract embedding for this step
            embedding = extract_embedding_from_forward(
                model, processor, prompt, img_pil, wrist_pil
            )
            embeddings.append(embedding)
            
            # Get action if queue is empty
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
            
            # Execute action
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            
            if done:
                success = True
                break
            
            t += 1
    
    except Exception as e:
        print(f"      Episode error: {e}")
    
    return embeddings, success, len(embeddings)


def extract_embeddings_rollout(
    checkpoint_path: str,
    task_suite_name: str = "libero_goal",
    command_levels=("default", "l1", "l2", "l3"),
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/embeddings",
    resolution: int = 256,
    seed: int = 0,
    num_rollouts_per_task: int = 10,
    first_step_only: bool = False,
):
    """
    Extract mean embeddings during real inference rollouts.
    
    For each task (10 tasks) and each command level:
    - Runs num_rollouts_per_task episodes (10 rollouts)
    - If first_step_only: extracts only the first observation embedding per rollout
    - Otherwise: extracts embeddings at each step of each episode
    - Computes mean embedding across all rollouts
    
    Total episodes: 10 tasks × 10 rollouts = 100 episodes per command level
    """
    
    # Create config
    cfg = EmbeddingConfig(
        pretrained_checkpoint=checkpoint_path,
        task_suite_name=task_suite_name,
        env_img_res=resolution,
    )
    
    # Load model and components
    model, processor, action_head, proprio_projector, noisy_action_projector, resize_size = \
        load_model_and_components(cfg)
    
    # Task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 300)
    
    mode_str = "FIRST STEP ONLY" if first_step_only else "FULL ROLLOUT"
    print(f"\n{'='*80}")
    print(f"EMBEDDING EXTRACTION - {mode_str}")
    print(f"{'='*80}")
    print(f"Task suite: {task_suite_name} ({num_tasks} tasks)")
    print(f"Command levels: {command_levels}")
    print(f"Rollouts per task: {num_rollouts_per_task}")
    if not first_step_only:
        print(f"Max steps per episode: {max_steps}")
    print(f"Total episodes per level: {num_tasks * num_rollouts_per_task}")
    print(f"{'='*80}\n")

    all_embeddings = {}

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = getattr(task, 'name', str(task))
        
        # Get initial states for this task
        initial_states = task_suite.get_task_init_states(task_id)

        print("=" * 80)
        print(f"Task {task_id + 1}/{num_tasks}: {task_name}")
        print("=" * 80)

        for level in command_levels:
            bddl_file = build_bddl_path(task, level)
            
            if not os.path.exists(bddl_file):
                print(f"  {level.upper():8s}: BDDL file not found")
                continue

            command = extract_command_from_bddl(bddl_file)
            if command is None:
                print(f"  {level.upper():8s}: Could not extract command")
                continue

            print(f"\n  {level.upper():8s}: {command}")
            prompt = build_prompt(command)

            # Collect embeddings from all rollouts
            rollout_embeddings = []       # Embedding per rollout (first step or mean of all steps)
            rollout_all_embeddings = []   # All step embeddings (only for full rollout mode)
            rollout_successes = []        # Success status per rollout (only for full rollout mode)
            successes = 0
            total_steps = 0
            
            for rollout_idx in range(num_rollouts_per_task):
                # Create environment
                try:
                    env, task_description, _ = get_libero_env(
                        task,
                        change_command=(level != "default"),
                        command_level=level if level != "default" else None,
                        resolution=resolution
                    )
                    env.seed(seed + rollout_idx)
                except Exception as e:
                    print(f"    Rollout {rollout_idx+1}: Failed to create env - {e}")
                    continue
                
                # Get initial state for this rollout
                init_state = initial_states[rollout_idx % len(initial_states)]
                
                try:
                    if first_step_only:
                        # Extract embedding from first observation only
                        embedding = extract_first_step_embedding(
                            cfg=cfg,
                            env=env,
                            prompt=prompt,
                            model=model,
                            processor=processor,
                            resize_size=resize_size,
                            initial_state=init_state,
                        )
                        rollout_embeddings.append(embedding)
                        print(f"    Rollout {rollout_idx+1:2d}/{num_rollouts_per_task}: ✓ (1 step)")
                    else:
                        # Run full episode
                        episode_embeddings, success, num_steps = run_single_episode(
                            cfg=cfg,
                            env=env,
                            task_description=command,
                            prompt=prompt,
                            model=model,
                            processor=processor,
                            resize_size=resize_size,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            initial_state=init_state,
                            max_steps=max_steps,
                        )
                        
                        # Compute mean embedding for this rollout
                        if episode_embeddings:
                            rollout_emb = np.stack(episode_embeddings, axis=0)
                            rollout_mean = np.mean(rollout_emb, axis=0)
                            rollout_embeddings.append(rollout_mean)
                            rollout_all_embeddings.append(rollout_emb)
                            rollout_successes.append(success)
                        
                        successes += int(success)
                        total_steps += num_steps
                        
                        status = "✓" if success else "✗"
                        print(f"    Rollout {rollout_idx+1:2d}/{num_rollouts_per_task}: {status} ({num_steps} steps)")
                    
                except Exception as e:
                    print(f"    Rollout {rollout_idx+1}: Error - {e}")
                finally:
                    try:
                        env.close()
                    except:
                        pass
            
            # Compute statistics: mean of rollout embeddings
            if rollout_embeddings:
                rollout_embeddings_arr = np.stack(rollout_embeddings, axis=0)  # (num_rollouts, hidden_dim)
                mean_embedding = np.mean(rollout_embeddings_arr, axis=0)  # Mean of embeddings
                
                key = f"task_{task_id:02d}_{level}"
                all_embeddings[key] = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "command_level": level,
                    "command_text": command,
                    "prompt": prompt,
                    "embedding": mean_embedding,                        # Mean of rollout embeddings
                    "embedding_per_rollout": rollout_embeddings_arr,    # Embedding per rollout
                    "num_rollouts": len(rollout_embeddings),
                    "first_step_only": first_step_only,
                }
                
                # Add full rollout specific data
                if not first_step_only and rollout_all_embeddings:
                    all_embeddings[key]["embedding_all_steps"] = np.concatenate(rollout_all_embeddings, axis=0)
                    all_embeddings[key]["rollout_successes"] = rollout_successes
                    all_embeddings[key]["num_successes"] = successes
                    all_embeddings[key]["total_steps"] = total_steps
                    all_embeddings[key]["success_rate"] = successes / max(len(rollout_embeddings), 1)
                
                if first_step_only:
                    print(f"    Summary: {len(rollout_embeddings)} embeddings, shape: {mean_embedding.shape}")
                else:
                    print(f"    Summary: {successes}/{num_rollouts_per_task} success, "
                          f"{total_steps} total steps, embedding shape: {mean_embedding.shape}")
            else:
                print(f"    No embeddings extracted for {level}")

        print()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "first_step" if first_step_only else "full"
    output_file = os.path.join(
        output_dir, 
        f"rollout_embeddings_{task_suite_name}_{'_'.join(command_levels)}_{mode_suffix}_r{num_rollouts_per_task}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump(all_embeddings, f)

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    
    if all_embeddings:
        first_key = next(iter(all_embeddings.keys()))
        print(f"Total entries: {len(all_embeddings)}")
        print(f"Mean embedding shape: {all_embeddings[first_key]['embedding'].shape}")
        print(f"Mode: {'First step only' if first_step_only else 'Full rollout'}")
        
        # Success rate summary per level (only for full rollout)
        if not first_step_only:
            for level in command_levels:
                level_data = [v for k, v in all_embeddings.items() if v['command_level'] == level]
                if level_data:
                    avg_sr = np.mean([d['success_rate'] for d in level_data])
                    print(f"  {level.upper()} avg success rate: {avg_sr:.2%}")
    
    print(f"\nOutput saved to: {output_file}")
    return all_embeddings, output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract embeddings during real inference rollouts from OpenVLA on LIBERO"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to OpenVLA checkpoint"
    )
    parser.add_argument(
        "--task_suite", 
        type=str, 
        default="libero_goal"
    )
    parser.add_argument(
        "--command_levels",
        type=str,
        nargs="+",
        default=["default", "l1", "l2", "l3"],
        help="Command levels to extract (e.g., --command_levels default or --command_levels l1 l2)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/mnt/beegfs/a.cardamone7/outputs/embeddings"
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_rollouts", 
        type=int, 
        default=10,
        help="Number of rollout episodes per task (default: 10)"
    )
    parser.add_argument(
        "--first_step_only",
        action="store_true",
        help="Extract embedding only from first observation (no full rollout)"
    )

    args = parser.parse_args()

    extract_embeddings_rollout(
        checkpoint_path=args.checkpoint,
        task_suite_name=args.task_suite,
        command_levels=tuple(args.command_levels),
        output_dir=args.output_dir,
        resolution=args.resolution,
        seed=args.seed,
        num_rollouts_per_task=args.num_rollouts,
        first_step_only=args.first_step_only,
    )
