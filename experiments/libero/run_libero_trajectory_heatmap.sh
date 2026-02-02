#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=libero_heatmap
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:59:00
#SBATCH --array=0-2  # Esegue 3 job in parallelo (seed 0, 1, 2)
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/trajectory_heatmap_seed_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/trajectory_heatmap_seed_%a_%j.err

SEED=$SLURM_ARRAY_TASK_ID

# Model configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-libero-goal-seed${SEED}"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs/trajectory/heatmap_libero_goal_seed_${SEED}"

echo "=========================================="
echo "LIBERO Trajectory Heatmap Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Seed): $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

# Setup environment
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1

# Activate conda
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Change to work directory
cd ${WORK_DIR}

# Run heatmap generation
python run_libero_trajectory_heatmap.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --model_family openvla \
    --task_suite_name libero_goal \
    --single_task_id 2 \
    --unnorm_key libero_goal_no_noops \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --num_open_loop_steps 8 \
    --center_crop True \
    --num_trials_per_task 50 \
    --env_img_res 256 \
    --seed ${SEED} \
    --command_levels "default" \
    --output_dir ${OUTPUT_DIR} \
    --save_videos True

echo ""
echo "Finish time: $(date)"
echo "=========================================="
echo "Heatmap generation with seed $SEED completed!"
echo "Output saved to: $OUTPUT_DIR"
