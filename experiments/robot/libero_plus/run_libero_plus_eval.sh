#!/bin/bash

#SBATCH -A did_robot_learning_359
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib; do
    if [ -d "$d" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$d
    fi
done

# Add the path to LIBERO folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus
export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/transformers-openvla-oft
export LIBERO_CONFIG_PATH="/mnt/beegfs/frosa/.libero_plus"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero
bddl_files: /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero/bddl_files
init_states: /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero/init_files
datasets: /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/datasets
assets: /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero/assets
EOF

RUN_ID=$1
ID_NOTE=${2:-libero_plus_eval}
MODEL_PATH=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/models/OpenVLA-OFT+
CONDA_ENV_PREFIX=/mnt/beegfs/frosa/.conda/envs/openvla-oft-libero-plus
PYTHON_BIN=${CONDA_ENV_PREFIX}/bin/python
TORCHRUN_BIN=${CONDA_ENV_PREFIX}/bin/torchrun


# if [ -z "$RUN_ID" ] || [ -z "$MODEL_PATH" ]; then
#     echo "Usage: $0 <RUN_ID> <MODEL_PATH> [ID_NOTE]"
#     exit 1
# fi

echo "Running evaluation for run ${RUN_ID} with ID note: ${ID_NOTE}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python not found at ${PYTHON_BIN}"
    exit 1
fi

if [ -x "${TORCHRUN_BIN}" ]; then
    LAUNCH_CMD="${TORCHRUN_BIN} --standalone --nnodes 1 --nproc-per-node 1"
else
    LAUNCH_CMD="${PYTHON_BIN} -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1"
fi

srun ${LAUNCH_CMD} run_libero_plus_eval.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --num_images_in_input 2 \
    --use_proprio True \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune" \
    --run_id_note ${ID_NOTE} \
    --task_suite_name "libero_object" \
    --run_number ${RUN_ID} \
    --change_spawn False \
    --spawn_train_distribution False \
    --debug False \
    --num_trials_per_task 1 \
    # --enrich_existing_rollouts_only True
 
