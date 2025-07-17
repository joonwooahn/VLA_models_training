#!/bin/bash

#SBATCH --output=_logs/univla/slurm-%j-%x.log
#SBATCH --gpus=1
#SBATCH --partition=rlwrld

# srun --gpus=1 --nodes=1 --pty /bin/bash

# 스크립트 디렉토리 설정 (절대 경로 사용)
SCRIPT_DIR="/virtual_lab/rlwrld/david/VLA_models_training/univla/vla_scripts"

# 스크립트 디렉토리로 이동
cd "$SCRIPT_DIR"

# 기본 설정
OUTPUT_DATA="$SCRIPT_DIR/converted_data_for_univla"
CONDA_ENV="univla_train"

# 명령행 인자 처리
if [ $# -eq 0 ]; then
    echo "사용법: $0 <JOB_NAME> <INPUT_DATA_PATH> [OUTPUT_DATA_PATH] [CONDA_ENV] [STATE_INDICES] [ACTION_INDICES] [CHECKPOINT_DIR_NAME]"
    echo ""
    echo "예시:"
    echo "  $0 my_job /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_rightarm_truncated"
    echo "  $0 ablation_study /path/to/input /path/to/output univla_train \"0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23,24,25,26,27,28,29,30\" \"0,1,2,3,4,5,12,13,14,15,16,17\" \"allex_gesture_easy_pos_only_right_arm\""
    echo ""
    echo "기본값:"
    echo "  OUTPUT_DATA: $OUTPUT_DATA"
    echo "  CONDA_ENV: $CONDA_ENV"
    echo "  CHECKOUT_NAME: 자동으로 INPUT_DATA_PATH의 마지막 폴더명으로 설정"
    echo "  STATE_INDICES: \"0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23,24,25,26,27,28,29,30\""
    echo "  ACTION_INDICES: \"0,1,2,3,4,5,12,13,14,15,16,17\""
    echo "  CHECKPOINT_DIR_NAME: ablation study script에서 자동 생성"
    exit 1
fi

JOB_NAME="$1"
INPUT_DATA="$2"
OUTPUT_DATA="${3:-$OUTPUT_DATA}"
CONDA_ENV="${4:-$CONDA_ENV}"
STATE_INDICES="${5:-"0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23,24,25,26,27,28,29,30"}"
ACTION_INDICES="${6:-"0,1,2,3,4,5,12,13,14,15,16,17"}"

# Checkpoint directory name (passed from ablation study script)
CHECKPOINT_DIR_NAME="${7:-}"

# INPUT_DATA_PATH의 마지막 폴더명을 기본 CHECKOUT_NAME으로 사용
# 하지만 CHECKPOINT_DIR_NAME이 제공되면 그것을 사용
if [ -n "$CHECKPOINT_DIR_NAME" ]; then
    CHECKOUT_NAME="$CHECKPOINT_DIR_NAME"
else
    CHECKOUT_NAME=$(basename "$INPUT_DATA")
fi

# 색상 출력을 위한 함수
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_step() {
    echo -e "\n\033[1;36m========================================\033[0m"
    echo -e "\033[1;36m$1\033[0m"
    echo -e "\033[1;36m========================================\033[0m"
}

# 설정 정보 출력
print_step "설정 정보"
print_info "입력 데이터: $INPUT_DATA"
print_info "출력 데이터: $OUTPUT_DATA"
print_info "Conda 환경: $CONDA_ENV"
print_info "실험 이름: $CHECKOUT_NAME"
if [ -n "$CHECKPOINT_DIR_NAME" ]; then
    print_info "체크포인트 디렉토리: $CHECKPOINT_DIR_NAME"
fi
print_info "State 인덱스: $STATE_INDICES"
print_info "Action 인덱스: $ACTION_INDICES"

# conda 환경 활성화
print_step "환경 설정"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
print_success "Conda 환경 활성화: $CONDA_ENV"

# 입력 데이터 경로 확인
print_step "입력 데이터 확인"
if [ ! -d "$INPUT_DATA" ]; then
    print_error "입력 데이터 경로가 존재하지 않습니다: $INPUT_DATA"
    exit 1
fi
print_success "입력 데이터 경로 확인: $INPUT_DATA"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DATA"
print_success "출력 디렉토리 생성: $OUTPUT_DATA"

# 변환된 데이터 존재 여부 확인
print_step "변환된 데이터 확인"
converted_data_exists=false
actual_data_path=""

# 입력 데이터의 마지막 폴더명으로 예상되는 변환된 데이터 경로
expected_converted_dir="$OUTPUT_DATA/$(basename "$INPUT_DATA")"
print_info "예상 변환된 데이터 경로: $expected_converted_dir"

# 엄격한 확인: 예상 경로에만 변환된 데이터가 있는지 확인
if [ -d "$expected_converted_dir" ]; then
    # 예상 경로에 변환된 데이터가 있는지 확인
    for item in "$expected_converted_dir"/*; do
        if [ -d "$item" ] && [ -f "$item/state.npy" ] && [ -f "$item/action.npy" ]; then
            converted_data_exists=true
            actual_data_path="$expected_converted_dir"
            break
        fi
    done
fi

# 다른 폴더의 데이터는 무시하고, 예상 경로에만 있는 경우에만 변환을 건너뜀

if [ "$converted_data_exists" = true ]; then
    print_warning "변환된 데이터가 이미 존재합니다: $actual_data_path"
    print_info "데이터 변환 단계를 건너뛰고 훈련으로 진행합니다."
    # 실제 데이터 경로로 업데이트
    OUTPUT_DATA="$actual_data_path"
else
    print_info "변환된 데이터가 없습니다. 데이터 변환을 시작합니다."
fi

# 1단계: 데이터 변환 (필요한 경우에만)
if [ "$converted_data_exists" = false ]; then
    print_step "1단계: LeRobot 데이터셋 변환"
    
    # 변환 스크립트 경로
    CONVERT_SCRIPT="$SCRIPT_DIR/convert_lerobot_dataset_for_univla.py"
    
    # 변환 스크립트가 존재하는지 확인
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        print_error "변환 스크립트를 찾을 수 없습니다: $CONVERT_SCRIPT"
        exit 1
    fi
    
    # 임시 백업 파일 생성
    cp "$CONVERT_SCRIPT" "${CONVERT_SCRIPT}.backup"
    
    # 경로 수정 (원본 파일에서 직접 수정)
    sed -i "s|base_input_dir = None|base_input_dir = \"$INPUT_DATA\"|g" "$CONVERT_SCRIPT"
    sed -i "s|output_dir = None|output_dir = \"$OUTPUT_DATA\"|g" "$CONVERT_SCRIPT"
    
    print_info "변환 스크립트 경로 수정 완료"
    print_info "  입력: $INPUT_DATA"
    print_info "  출력: $OUTPUT_DATA"
    
    # 변환 스크립트 실행
    print_info "데이터 변환 시작..."
    if python "$CONVERT_SCRIPT"; then
        print_success "데이터 변환 완료"
    else
        print_error "데이터 변환 실패"
        # 백업 파일 복원
        mv "${CONVERT_SCRIPT}.backup" "$CONVERT_SCRIPT"
        exit 1
    fi
    
    # 백업 파일 복원
    mv "${CONVERT_SCRIPT}.backup" "$CONVERT_SCRIPT"
else
    print_step "1단계: 데이터 변환 (건너뛰기)"
fi

# 변환된 데이터 최종 확인
print_step "변환된 데이터 최종 확인"
final_check=false
actual_data_path=""

if [ -d "$OUTPUT_DATA" ]; then
    # 먼저 직접 확인
    for item in "$OUTPUT_DATA"/*; do
        if [ -d "$item" ] && [ -f "$item/state.npy" ] && [ -f "$item/action.npy" ]; then
            final_check=true
            actual_data_path="$OUTPUT_DATA"
            break
        fi
    done
    
    # 직접 확인에서 못 찾았으면 하위 디렉토리 확인
    if [ "$final_check" = false ]; then
        for subdir in "$OUTPUT_DATA"/*; do
            if [ -d "$subdir" ]; then
                for item in "$subdir"/*; do
                    if [ -d "$item" ] && [ -f "$item/state.npy" ] && [ -f "$item/action.npy" ]; then
                        final_check=true
                        actual_data_path="$subdir"
                        break 2
                    fi
                done
            fi
        done
    fi
fi

if [ "$final_check" = false ]; then
    print_error "변환된 데이터를 찾을 수 없습니다: $OUTPUT_DATA"
    exit 1
fi

# 실제 데이터 경로로 업데이트
OUTPUT_DATA="$actual_data_path"
print_success "변환된 데이터 확인 완료: $OUTPUT_DATA"

# 2단계: 모델 훈련
print_step "2단계: UniVLA 모델 훈련"

# 훈련 스크립트 경로
FINETUNE_SCRIPT="$SCRIPT_DIR/finetune_rlwrld.py"

print_info "훈련 스크립트 실행 준비 완료"
print_info "  데이터 경로: $OUTPUT_DATA"
print_info "  실험 이름: $CHECKOUT_NAME"
if [ -n "$CHECKPOINT_DIR_NAME" ]; then
    print_info "  체크포인트 디렉토리: $CHECKPOINT_DIR_NAME"
fi
print_info "  State 인덱스: $STATE_INDICES"
print_info "  Action 인덱스: $ACTION_INDICES"

# 훈련 스크립트 실행 (명령행 인자로 전달)
print_info "모델 훈련 시작..."
if torchrun --standalone --nnodes 1 --nproc-per-node 1 "$FINETUNE_SCRIPT" \
    --data_root_dir "$OUTPUT_DATA" \
    --indices_for_state "$STATE_INDICES" \
    --indices_for_action "$ACTION_INDICES" \
    --checkout_name "$CHECKOUT_NAME"; then
    print_success "모델 훈련 완료"
else
    print_error "모델 훈련 실패"
    exit 1
fi

# 완료 메시지
print_step "작업 완료"
print_success "모든 작업이 성공적으로 완료되었습니다!"
print_info "변환된 데이터 위치: $OUTPUT_DATA"
print_info "훈련 결과는 runs/ 디렉토리에서 확인할 수 있습니다." 