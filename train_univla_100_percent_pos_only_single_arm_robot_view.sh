#!/bin/bash
#SBATCH --job-name=univla-univla_100_percent_pos_only_single_arm_robot_view
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh
conda activate univla_train

# 훈련 실행
python /virtual_lab/rlwrld/david/UniVLA/vla_scripts/finetune_rlwrld_ablation.py \
    --condition univla_100_percent_pos_only_single_arm_robot_view \
    --data-root-dir /virtual_lab/rlwrld/david/UniVLA/converted_data \
    --output-dir /virtual_lab/rlwrld/david/UniVLA/outputs/univla_100_percent_pos_only_single_arm_robot_view \
    --max-steps 10000 \
    --batch-size 8 \
    --learning-rate 0.0001
