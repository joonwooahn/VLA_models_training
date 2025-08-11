#!/bin/bash
#SBATCH --job-name=pi0-vla
#SBATCH --output=logs/%j-%x.log
#SBATCH --partition=rlwrld
#SBATCH --gpus=1

# srun --comment="pi0 training" --gpus=1 --nodes=1 --pty /bin/bash

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
echo "✅ Conda environment 'lerobot' activated."

python3 lerobot/scripts/train.py \
    --job_name $SLURM_JOB_NAME \
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=24 \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=RLWRLD/lift_cylinder_VLA/ \
    --wandb.enable=true \
    --wandb.disable_artifact=true

# python3 lerobot/scripts/train.py \
#     --job_name $SLURM_JOB_NAME \
#     --steps=30000 \
#     --save_freq=10000 \
#     --batch_size=16 \
#     --policy.path=lerobot/pi0fast_base \
#     --dataset.repo_id=RLWRLD/lift_cylinder_VLA/ \
#     --wandb.enable=true \
#     --wandb.disable_artifact=true
