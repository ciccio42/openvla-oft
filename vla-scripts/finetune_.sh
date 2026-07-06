#!/bin/bash

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1           # Only ONE task per node!
#SBATCH --gres=gpu:4                 # Request 1 GPUs per node
#SBATCH --cpus-per-task=32             # Adjust for data loading, etc.
#SBATCH --exclude=tnode[01-17]
#SBATCH --exclude=gnode14
#SBATCH --export=ALL

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_MODE=online

CKPT_FOLDER="${1:-openvla-7b}" #+ur5e_pick_place_abs_pose+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--85000_chkpt
RUN_ID_NOTE="${2:-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img}"
RESUME="${3:-false}"
RESUME_STEP="${4:-0}"
RUN_ROOT_DIR="${5:-/home/rsofnc000/checkpoint_save_folder/open_vla}"
DATASET_NAME="${6:-ur5e_pick_place_abs_pose}"
USE_PROPRIO="${7:-True}"
DATASET_FOLDER="${8:-/home/rsofnc000/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets}"

echo "Using checkpoint folder: $CKPT_FOLDER"
echo "Run ID note: $RUN_ID_NOTE"
echo "Resume: $RESUME"
echo "Resume step: $RESUME_STEP"
echo "Run root directory: $RUN_ROOT_DIR"
echo "Dataset name: $DATASET_NAME"
echo "Use proprioception: $USE_PROPRIO"

# assign a unique port based on the Slurm job ID
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

srun torchrun --standalone --nnodes 1 --nproc-per-node 4 --master-port $MASTER_PORT finetune.py \
    --vla_path  ${RUN_ROOT_DIR}/${CKPT_FOLDER} \
    --data_root_dir  ${DATASET_FOLDER} \
    --dataset_name $DATASET_NAME \
    --run_root_dir $RUN_ROOT_DIR \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio ${USE_PROPRIO} \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 224686 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune_${DATASET_NAME}" \
    --run_id_note $RUN_ID_NOTE \
    --resume $RESUME \
    --resume_step $RESUME_STEP
