#!/bin/bash

#SBATCH -A hpc_default
#SBATCH --partition=gpuq
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1           # Only ONE task per node!
#SBATCH --gres=gpu:4                  # Request 4 GPUs per node
#SBATCH --cpus-per-task=32             # Adjust for data loading, etc.
#SBATCH --exclude=tnode[01-17]
#SBATCH --export=ALL

CKPT_FOLDER="${1:-openvla-7b}" #+ur5e_pick_place_abs_pose+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--85000_chkpt
RUN_ID_NOTE="${2:-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img}"
RESUME="${3:-false}"
RESUME_STEP="${4:-0}"
RUN_ROOT_DIR="${5:-/home/rsofnc000/checkpoint_save_folder/open_vla}"
DATASET_NAME="${6:-ur5e_pick_place_abs_pose}"

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostname $SLURM_NODELIST | head -n1):29500 \
    finetune.py \
    --vla_path /home/rsofnc000/checkpoint_save_folder/open_vla/${CKPT_FOLDER} \
    --data_root_dir /home/rsofnc000/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets \
    --dataset_name $DATASET_NAME \
    --run_root_dir $RUN_ROOT_DIR \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 224686 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune_abs_pose_gripper_img_proprio" \
    --run_id_note $RUN_ID_NOTE \
    --resume $RESUME \
    --resume_step $RESUME_STEP
