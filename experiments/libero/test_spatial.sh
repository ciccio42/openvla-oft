#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=libero_spatial
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:59:00
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/libero_spatial_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/libero_spatial_%j.err
    
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "Libero Spatial Quick Test"
echo "Start time: $(date)"
echo ""

# Setup environment
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}

# Activate conda
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Change to work directory
cd ${WORK_DIR}

# Run quick test on libero_spatial (same task suite as training)
python run_libero_eval.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --model_family openvla \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --num_open_loop_steps 8 \
    --center_crop True \
    --num_trials_per_task 10 \
    --env_img_res 256 \
    --seed 42 \
    --change_command False \
    --run_id_note debug_spatial \
    --local_log_dir /mnt/beegfs/a.cardamone7/outputs/logs

echo ""
echo "Finish time: $(date)"
