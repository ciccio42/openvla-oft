#!/bin/bash

#SBATCH -A hpc_default
#SBATCH --exclude=tnode[01-17]
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
PKL_PATH=/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place_rm_12_13_14_15+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--ur5e_pick_place_rm_12_13_14_15_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--75000_chkpt/changed_objects/command_orange_red_in_scene/rollout_pick_place_1_False_obj_set_-1_change_command_True

srun python create_video.py \
    --path_to_pkl $PKL_PATH \
    --output_dir $PKL_PATH/video
# --debug
