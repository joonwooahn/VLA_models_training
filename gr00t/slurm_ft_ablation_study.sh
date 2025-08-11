#!/bin/bash
#SBATCH --job-name=gr00t-ablation
#SBATCH --output=_logs/gr00t/slurm-%j-%x.log
#SBATCH --partition=rlwrld
#SBATCH --gpus=1

# 명령줄 인수 처리
if [ $# -lt 7 ]; then
    echo "Usage: $0 <JOB_NAME> <DATA_DIR> <OUTPUT_DATA> <CONDA_ENV> <STATE_INDICES> <ACTION_INDICES> <CHECKPOINT_NAME> [STATE_MODE] [ACTION_MODE] [VIDEO_MODE] [VLA_MODEL] [ROBOT_TYPE] [SIM_OR_REAL]"
    exit 1
fi

JOB_NAME="$1"
DATA_DIR="$2"
OUTPUT_DATA="$3"
CONDA_ENV="$4"
STATE_INDICES="$5"
ACTION_INDICES="$6"
CHECKPOINT_NAME="$7"
STATE_MODE="${8:-pos_only}"
ACTION_MODE="${9:-right_arm}"
VIDEO_MODE="${10:-robotview}"
VLA_MODEL="${11:-gr00t}"
ROBOT_TYPE="${12:-allex}"
SIM_OR_REAL="${13:-real}"

# 로그 디렉토리 생성
mkdir -p "_logs/gr00t"

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_vla
echo "✅ Conda environment 'gr00t_vla' activated."

# 스크립트가 실행되는 디렉토리를 gr00t 디렉토리로 변경
cd /virtual_lab/rlwrld/david/VLA_models_training/gr00t

# Action dimension 설정 (ROBOT_TYPE과 ACTION_MODE에 따라)
if [ "$ROBOT_TYPE" = "franka" ]; then
    # Franka robot: arm_eef_pos (6) + finger_joints (6) = 12
    ACTION_DIM=12
elif [ "$ACTION_MODE" = "right_arm" ]; then
    # Allex robot right arm: right_arm_eef_pos (6) + right_finger_joints (15) = 21
    ACTION_DIM=21
elif [ "$ACTION_MODE" = "dual_arm" ]; then
    # Allex robot dual arm: both arms (12) + both fingers (30) = 42
    ACTION_DIM=42
else
    echo "Error: Unknown ACTION_MODE: $ACTION_MODE"
    exit 1
fi

# Data config 생성
if [ "$ROBOT_TYPE" = "franka" ]; then
    DATA_CONFIG="franka_${STATE_MODE}_${ACTION_MODE}_${VIDEO_MODE}"
elif [ "$ROBOT_TYPE" = "allex" ] && [ "$SIM_OR_REAL" = "real" ]; then
    # Allex real robot: include real information
    DATA_CONFIG="${ROBOT_TYPE}_${SIM_OR_REAL}_${STATE_MODE}_${ACTION_MODE}_${VIDEO_MODE}"
else
    # Allex sim robot or other cases: use original format
    DATA_CONFIG="${STATE_MODE}_${ACTION_MODE}_${VIDEO_MODE}"
fi

echo "ROBOT_TYPE: $ROBOT_TYPE"
echo "SIM_OR_REAL: $SIM_OR_REAL"
echo "DATA_CONFIG: $DATA_CONFIG"
echo "ACTION_DIM: $ACTION_DIM"
echo "MAX_STEPS: $MAX_STEPS"
echo "SAVE_STEPS: $SAVE_STEPS"

# Checkpoint 저장 경로 설정
OUTPUT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/gr00t/${CHECKPOINT_NAME}"

# Gr00t 훈련 실행
if [ "$ROBOT_TYPE" = "franka" ]; then
    # Franka robot: shorter training for faster iteration
    MAX_STEPS=15000
    SAVE_STEPS=3000
else
    # Allex robot: original settings
    MAX_STEPS=10000
    SAVE_STEPS=5000
fi

python scripts/gr00t_finetune.py \
   --dataset-path "$DATA_DIR" \
   --num-gpus 1 \
   --output-dir "$OUTPUT_DIR" \
   --max-steps "$MAX_STEPS" \
   --save-steps "$SAVE_STEPS" \
   --data-config "$DATA_CONFIG" \
   --video-backend torchvision_av \
   --action_dim "$ACTION_DIM"

echo "Gr00t training completed!"
