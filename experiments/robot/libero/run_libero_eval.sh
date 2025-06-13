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

ID_NOTE=parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio
MODEL_PATH=/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--${ID_NOTE}--150000_chkpt

srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_libero_eval.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --num_images_in_input 2 \
    --use_proprio True \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune" \
    --run_id_note ${ID_NOTE} \
    --task_suite_name "libero_object"
# --debug True
