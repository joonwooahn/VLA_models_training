#!/bin/bash
#SBATCH --job-name=pi0-david
#SBATCH --output=logs/%j-%x.log
#SBATCH --partition=rlwrld
#SBATCH --gpus=1

# srun --comment="pi0 training" --gpus=1 --nodes=1 --pty /bin/bash

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_vla
echo "✅ Conda environment 'lerobot_vla' activated."

python3 lerobot/scripts/train.py \
    --job_name $SLURM_JOB_NAME \
    --steps=30000 \
    --save_freq=5000 \
    --batch_size=24 \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=RLWRLD/250718/pi0/allex_gesture_easy_all/P_D_M \
    --wandb.enable=true \
    --wandb.disable_artifact=true


# pos_only_right_arm_robotview → P_R_R
# pos_only_right_arm_multiview → P_R_M
# pos_vel_right_arm_robotview → PV_R_R
# pos_vel_right_arm_multiview → PV_R_M
# pos_vel_dual_arm_robotview → PV_D_R
# pos_vel_dual_arm_multiview → PV_D_M
# pos_vel_torq_right_arm_robotview → PVT_R_R
# pos_vel_torq_right_arm_multiview → PVT_R_M
# pos_vel_torq_dual_arm_robotview → PVT_D_R
# pos_vel_torq_dual_arm_multiview → PVT_D_M