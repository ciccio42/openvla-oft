#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=openvla_task_comp_l1_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2          # Array index = seed (0, 1, 2 for multi-seed)
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/eval_openvla_50000_task_comp_l1_seed_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/eval_openvla_50000_task_comp_l1_seed_%a_%j.err

# ==========================================
# OpenVLA-OFT - Task Composition L1 Evaluation
# ==========================================
# Tests task-level generalization: the model must apply known
# primitives (pick-place, open drawer, etc.) to new object/target
# combinations never seen during training.
# ==========================================

SEED=$SLURM_ARRAY_TASK_ID
ID_NOTE="openvla_task_comp_l1_seed_${SEED}"

# Model configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-libero-goal-seed${SEED}"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "OpenVLA-OFT Task Composition L1 Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Seed): $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo ""

# ==========================================
# Environment Setup
# ==========================================

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1

# ==========================================
# Activate Conda Environment
# ==========================================

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ==========================================
# Run Evaluation
# ==========================================

cd ${WORK_DIR}

python run_libero_eval_task_comp.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --model_family openvla \
    --task_suite_name libero_goal \
    --unnorm_key libero_goal_noops \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --num_open_loop_steps 8 \
    --center_crop True \
    --num_trials_per_task 50 \
    --env_img_res 256 \
    --seed ${SEED} \
    --run_id_note ${ID_NOTE} \
    --local_log_dir /mnt/beegfs/a.cardamone7/outputs/logs

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task Composition L1 evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "Seed: $SEED"
echo "Finish time: $(date)"
echo "=========================================="

exit $EXIT_CODE
