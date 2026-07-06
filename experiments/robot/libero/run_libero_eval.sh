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

# Add the path to LIBERO folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO
export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/openvla-oft/transformers-openvla-oft
export LIBERO_CONFIG_PATH="/mnt/beegfs/frosa/.libero"

RUN_ID=$1
ID_NOTE=parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio
MODEL_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/open_vla/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--${ID_NOTE}--135000_chkpt

echo "Running evaluation for run ${RUN_ID} with ID note: ${ID_NOTE}"
if [ "$RUN_ID" -ne 1 ]; then
    SAVE=False
fi
srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_libero_eval.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --num_images_in_input 2 \
    --use_proprio True \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune" \
    --run_id_note ${ID_NOTE} \
    --task_suite_name "libero_object" \
    --run_number ${RUN_ID} \
    --change_spawn True \
    --spawn_train_distribution False \
    --debug False
 