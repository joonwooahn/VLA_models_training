#!/bin/bash

#SBATCH --job-name=univla-finetune-rlwrld
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# srun --gpus=1 --nodes=1 --pty /bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate univla_train

torchrun inference_rlwrld.py
