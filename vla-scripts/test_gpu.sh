#!/bin/bash

#SBATCH -A hpc_default
#SBATCH --partition=gpuq
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1           # Only ONE task per node!
#SBATCH --gres=gpu:1                  # Request 4 GPUs per node
#SBATCH --cpus-per-task=32             # Adjust for data loading, etc.
#SBATCH --exclude=tnode[01-17]
#SBATCH --export=ALL

export CUDA_HOME=/cm/shared/apps/cuda11.7/toolkit/11.7.1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

srun python test_gpu.py
