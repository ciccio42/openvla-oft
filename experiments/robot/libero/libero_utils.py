"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)
import libero.libero.envs.bddl_utils as BDDLUtils
import json
import copy
import time



def change_bddl_file(bddl_file, spawn_train_distribution=True):
    
    # get the problem info from the bddl file    
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    parsed_bddl = BDDLUtils.robosuite_parse_problem(bddl_file)

    # open spawn_region.json file
    # spawn_region_file = os.path.join(get_libero_path("bddl_files"), "spawn_region.json")
    bddl_folder = os.path.dirname(bddl_file)
    spawn_region_file = os.path.join(bddl_folder, "spawn_region.json")
    with open(spawn_region_file, 'r') as f:
        spawn_region = json.load(f)
        
    # change the spawn region in the bddl file
    if spawn_train_distribution:
        # get the current spawn region for the target object
        current_spawn_region = parsed_bddl['regions']['floor_target_object_region']['ranges']
        
        # sample a new spawn region from the spawn_region.json file
        new_spawn_region = spawn_region['target'][np.random.randint(len(spawn_region['target']))]    
        # ensure the new spawn region is different from the current one
        while new_spawn_region == current_spawn_region:
            new_spawn_region = spawn_region['target'][np.random.randint(len(spawn_region['target']))]
            
        # update the bddl file with the new spawn region
        parsed_bddl['regions']['floor_target_object_region']['ranges'] = new_spawn_region
        
        # change if other objects have the same spawn region
        changed_object = None
        for key in parsed_bddl['regions'].keys():
            if 'other_object' in key:
                region = parsed_bddl['regions'][key]['ranges']
                if region == new_spawn_region:
                    # change with the previous spawn region
                    parsed_bddl['regions'][key]['ranges'] = current_spawn_region
                    changed_object = key.split('floor_')[1]                    
                    break
                
        # write the new bddl file
        # open the bddl file
        with open(bddl_file, 'r') as f:
            old_bddl_content = f.readlines()
            new_bddl_content = copy.deepcopy(old_bddl_content)
            
        for row_indx, row in enumerate(old_bddl_content):
            if '(target_object_region' in row:
                new_bddl_content[row_indx + 3] = f'              ({new_spawn_region[0][0]} {new_spawn_region[0][1]} {new_spawn_region[0][2]} {new_spawn_region[0][3]})\n'
            if changed_object in row:
                new_bddl_content[row_indx + 3] = f'              ({current_spawn_region[0][0]} {current_spawn_region[0][1]} {current_spawn_region[0][2]} {current_spawn_region[0][3]})\n'
        
        # write the new bddl file
        new_bbdl_file = bddl_file.replace('.bddl', '_spawn_distribution.bddl')
        with open(new_bbdl_file, 'w') as f:
            f.writelines(new_bddl_content)
            
        return new_bbdl_file
    else:
        # sample new spawn region from other objects
        # get the current spawn region for the target object
        current_spawn_region = parsed_bddl['regions']['floor_target_object_region']['ranges']
        
        # sample a new spawn region from the spawn_region.json file
        new_spawn_region = spawn_region['other_objects'][np.random.randint(len(spawn_region['other_objects']))]    
        # ensure the new spawn region is different from the current one
        while new_spawn_region == current_spawn_region:
            new_spawn_region = spawn_region['other_objects'][np.random.randint(len(spawn_region['other_objects']))]
            
        # update the bddl file with the new spawn region
        parsed_bddl['regions']['floor_target_object_region']['ranges'] = new_spawn_region
        
        # change if other objects have the same spawn region
        changed_object = None
        for key in parsed_bddl['regions'].keys():
            if 'other_object' in key:
                region = parsed_bddl['regions'][key]['ranges']
                if region == new_spawn_region:
                    # change with the previous spawn region
                    parsed_bddl['regions'][key]['ranges'] = current_spawn_region
                    changed_object = key.split('floor_')[1]                    
                    break
                
        # write the new bddl file
        # open the bddl file
        with open(bddl_file, 'r') as f:
            old_bddl_content = f.readlines()
            new_bddl_content = copy.deepcopy(old_bddl_content)
            
        for row_indx, row in enumerate(old_bddl_content):
            if '(target_object_region' in row:
                new_bddl_content[row_indx + 3] = f'              ({new_spawn_region[0][0]} {new_spawn_region[0][1]} {new_spawn_region[0][2]} {new_spawn_region[0][3]})\n'
            if changed_object in row:
                new_bddl_content[row_indx + 3] = f'              ({current_spawn_region[0][0]} {current_spawn_region[0][1]} {current_spawn_region[0][2]} {current_spawn_region[0][3]})\n'
        
        # write the new bddl file
        new_bbdl_file = bddl_file.replace('.bddl', f'_spawn_train_distribution_{spawn_train_distribution}.bddl')
        with open(new_bbdl_file, 'w') as f:
            f.writelines(new_bddl_content)
        time.sleep(5)  # Ensure the file is written before returning
        return new_bbdl_file
        

def get_libero_env(task, model_family, change_spawn=False, train_spawn_distribution=True, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    if change_spawn:
        task_bddl_file = change_bddl_file(task_bddl_file, train_spawn_distribution)
        
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_traj, idx, success, task_description, log_file=None, dataset_name="libero", run=0, change_spawn=False, train_spawn_distribution=True):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{dataset_name}/change_spawn_{change_spawn}_train_{train_spawn_distribution}/run_{run}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_traj['image']:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
        
    # save replay trajectory as a numpy file
    npy_path = mp4_path.replace('.mp4', '.npy')
    np.save(npy_path, rollout_traj)
    print(f"Saved replay trajectory at path {npy_path}")    
    
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
