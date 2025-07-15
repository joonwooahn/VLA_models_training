#!/bin/bash
#SBATCH --job-name=pi0-ft-allex-bimanual-cube
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
echo "✅ Conda environment 'lerobot' activated."

python3 lerobot/scripts/train.py \
    --job_name $SLURM_JOB_NAME \
    --steps=30000 \
    --batch_size=24 \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=RLWRLD/allex_cube \
    --wandb.enable=true \
    --wandb.disable_artifact=true

# --dataset.repo_id 는 데이터 존재하는 path, 보통 .cache/huggingface/lerobot/RLWRLD/allex_cube 아래 있음
# --policy.path 는 훈련시킬 모델명
