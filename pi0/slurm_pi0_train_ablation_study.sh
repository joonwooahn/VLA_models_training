#!/bin/bash

#SBATCH --job-name=pi0-ablation
#SBATCH --partition=rlwrld
#SBATCH --gpus=1
#SBATCH --nodes=1

# 명령줄 인수 처리
if [ $# -lt 7 ]; then
    echo "Usage: $0 <JOB_NAME> <DATA_DIR> <OUTPUT_DATA> <CONDA_ENV> <STATE_INDICES> <ACTION_INDICES> <CHECKPOINT_NAME> [STATE_MODE] [ACTION_MODE] [VIDEO_MODE] [VLA_MODEL]"
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
VLA_MODEL="${11:-pi0}"

# VLA_MODEL에 따른 기본 batch_size 설정
if [ "$VLA_MODEL" = "pi0fast" ]; then
    if [ "$VIDEO_MODE" = "multiview" ]; then
        BATCH_SIZE=8
    else
        BATCH_SIZE=16
    fi
elif [ "$VLA_MODEL" = "diffusion" ]; then
    BATCH_SIZE=512
elif [ "$VLA_MODEL" = "act" ]; then
    BATCH_SIZE=128
else
    BATCH_SIZE=24
fi

# VLA_MODEL에 따른 로그 디렉토리 설정
if [ "$VLA_MODEL" = "pi0fast" ]; then
    LOG_DIR="_logs/pi0fast"
elif [ "$VLA_MODEL" = "diffusion" ]; then
    LOG_DIR="_logs/diffusion"
elif [ "$VLA_MODEL" = "act" ]; then
    LOG_DIR="_logs/act"
else
    LOG_DIR="_logs/pi0"
fi

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# SLURM 로그 파일 경로 설정
SLURM_LOG_FILE="${LOG_DIR}/slurm-${SLURM_JOB_ID:-$$}-${SLURM_JOB_NAME:-${JOB_NAME}}.log"

# 표준 출력과 에러를 로그 파일로 리디렉션 (SLURM 환경에서만)
# if [ -n "$SLURM_JOB_ID" ]; then
#     exec > >(tee -a "$SLURM_LOG_FILE")
#     exec 2>&1
# fi

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lerobot_vla
conda activate lerobot  
echo "✅ Conda environment 'lerobot' activated."

# 스크립트가 실행되는 디렉토리를 pi0 디렉토리로 변경
cd /virtual_lab/rlwrld/david/VLA_models_training/pi0
echo "✅ Changed to pi0 directory: $(pwd)"

# 데이터셋 경로 생성 (create_pi0_dataset.py의 short_name 로직 사용)
get_short_name() {
    local state_mode="$1"
    local action_mode="$2"
    local video_mode="$3"
    
    # State mode 약어
    case "$state_mode" in
        "pos_only") state_abbr="P" ;;
        "pos_vel") state_abbr="PV" ;;
        "pos_vel_torq") state_abbr="PVT" ;;
        *) state_abbr="P" ;;
    esac
    
    # Action mode 약어
    case "$action_mode" in
        "right_arm") action_abbr="R" ;;
        "dual_arm") action_abbr="D" ;;
        *) action_abbr="R" ;;
    esac
    
    # Video mode 약어
    case "$video_mode" in
        "robotview") video_abbr="R" ;;
        "multiview") video_abbr="M" ;;
        *) video_abbr="R" ;;
    esac
    
    echo "${state_abbr}_${action_abbr}_${video_abbr}"
}

# 태스크 이름 추출 (DATA_DIR의 마지막 폴더명)
TASK_NAME=$(basename "$DATA_DIR")

# Short name 생성
SHORT_NAME=$(get_short_name "$STATE_MODE" "$ACTION_MODE" "$VIDEO_MODE")

# 데이터셋 repo_id 구성
DATASET_REPO_ID="RLWRLD/pi0/${TASK_NAME}/${SHORT_NAME}"

# Policy path 설정
if [ "$VLA_MODEL" = "pi0fast" ]; then
    POLICY_PATH="lerobot/pi0fast_base"
elif [ "$VLA_MODEL" = "diffusion" ]; then
    POLICY_PATH="lerobot/diffusion"
elif [ "$VLA_MODEL" = "act" ]; then
    POLICY_PATH="lerobot/act"
else
    POLICY_PATH="lerobot/pi0"
fi

echo "========================================="
echo "Pi0 Training Configuration"
echo "========================================="
echo "VLA Model: $VLA_MODEL"
echo "Job Name: $JOB_NAME"
echo "Task Name: $TASK_NAME"
echo "State Mode: $STATE_MODE"
echo "Action Mode: $ACTION_MODE"
echo "Video Mode: $VIDEO_MODE"
echo "Short Name: $SHORT_NAME"
echo "Policy Path: $POLICY_PATH"
echo "Dataset Repo ID: $DATASET_REPO_ID"
echo "Log Directory: $LOG_DIR"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "SLURM Log File: $SLURM_LOG_FILE"
fi
echo "State Indices: $STATE_INDICES"
echo "Action Indices: $ACTION_INDICES"
echo "Checkpoint Name: $CHECKPOINT_NAME"
echo "========================================="

# Pi0 훈련 실행 (단일 GPU 사용)
echo "Using single GPU for training"
echo "Batch size: $BATCH_SIZE"

# Checkpoint 저장 경로 설정 (univla와 동일한 구조)
if [ "$VLA_MODEL" = "pi0fast" ]; then
    OUTPUT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/pi0fast/${CHECKPOINT_NAME}"
elif [ "$VLA_MODEL" = "diffusion" ]; then
    OUTPUT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/diffusion/${CHECKPOINT_NAME}"
elif [ "$VLA_MODEL" = "act" ]; then
    OUTPUT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/act/${CHECKPOINT_NAME}"
else
    OUTPUT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/_checkpoints/pi0/${CHECKPOINT_NAME}"
fi

echo "Checkpoint output directory: $OUTPUT_DIR"

# 기존 출력 디렉토리가 있으면 삭제
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing existing output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# Diffusion/Act 모델인 경우 policy.type 사용
if [ "$VLA_MODEL" = "diffusion" ]; then
    python3 lerobot/scripts/train.py \
        --job_name "$JOB_NAME" \
        --steps=50000 \
        --save_freq=50000 \
        --batch_size="$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --policy.type=diffusion \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --wandb.enable=true \
        --wandb.disable_artifact=true
elif [ "$VLA_MODEL" = "act" ]; then
    python3 lerobot/scripts/train.py \
        --job_name "$JOB_NAME" \
        --steps=50000 \
        --save_freq=50000 \
        --batch_size="$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --policy.type=act \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --wandb.enable=true \
        --wandb.disable_artifact=true
else
    python3 lerobot/scripts/train.py \
        --job_name "$JOB_NAME" \
        --steps=1000 \
        --save_freq=1000 \
        --batch_size="$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --policy.path="$POLICY_PATH" \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --wandb.enable=true \
        --wandb.disable_artifact=true
fi

echo "Pi0 training completed!"

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