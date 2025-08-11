#!/bin/bash
#SBATCH --job-name=gr00-vla
#SBATCH --output=_logs/gr00t/slurm-%j-%x.log
#SBATCH --partition=rlwrld
#SBATCH --gpus=1

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_vla
echo "✅ Conda environment 'gr00t_vla' activated."

DATA_CONFIG=lift_cylinder_VLA
ACTION_DIM=7

# Gr00t 훈련 실행
python scripts/gr00t_finetune.py \
   --dataset-path "/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/${DATA_CONFIG}" \
   --num-gpus 1 \
   --output-dir "/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/gr00t/qwer/${DATA_CONFIG}" \
   --max-steps 30000 \
   --save-steps 10000 \
   --data-config "$DATA_CONFIG" \
   --video-backend torchvision_av \
   --action_dim "$ACTION_DIM"

echo "Gr00t training completed!"

# --max-steps 30000 \
# --save-steps 10000 \
