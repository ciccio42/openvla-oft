#!/bin/bash

#SBATCH --exclude=tnode[01-17]
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL

srun torchrun --standalone --nnodes 1 --nproc-per-node 4 finetune.py \
    --vla_path /home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--45000_chkpt \
    --data_root_dir /home/rsofnc000/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets \
    --dataset_name ur5e_pick_place \
    --run_root_dir /home/rsofnc000/checkpoint_save_folder/open_vla \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 224686 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "francescorosa97" \
    --wandb_project "Open_VLA_OFT_finetune" \
    --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img \
    --resume True \
    --resume_step 45000
