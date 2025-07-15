#!/bin/bash

# =============================================================================
# VLA Models Ablation Study - 모든 조합 자동 실행 스크립트
# =============================================================================
# 모든 모델(GR00T, PI0, PI0_FAST, UniVLA)에 대해 모든 data/state/action 조합을 실행합니다.
# 
# 사용법:
#   chmod +x run_all_ablations.sh
#   ./run_all_ablations.sh                           # 모든 모델 실행 (기본값들)
#   ./run_all_ablations.sh univla                    # UniVLA만 실행
#   ./run_all_ablations.sh --max-steps 5000 gr00t    # max_steps=5000으로 GR00T 실행
#   ./run_all_ablations.sh --input-dir "/path/to/data" pi0  # 커스텀 데이터 경로
#   ./run_all_ablations.sh --help                    # 도움말 표시
# =============================================================================

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 기본값 설정
DEFAULT_MAX_STEPS=10000
DEFAULT_SAVE_INTERVAL=5000
DEFAULT_NUM_WORKERS=8
DEFAULT_DATASET_NAME="allex_gesture_easy_pos_vel_torq"
DEFAULT_INPUT_DIR="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/${DEFAULT_DATASET_NAME}"

# 모델별 설정을 가져오는 함수
get_model_config() {
    local model=$1
    local config_type=$2
    
    case $config_type in
        "state")
            if [ "${model}" == "gr00t" ]; then
                echo "pos_only"
            else
                echo "pos_only pos_vel pos_vel_torque"
            fi
            ;;
        "camera")
            if [ "${model}" == "univla" ]; then
                echo "robot_view"
            else
                echo "robot_view multi_view"
            fi
            ;;
        "combinations")
            if [ "${model}" == "gr00t" ]; then
                echo "4"  # 2(data) × 1(state) × 2(action) = 4
            elif [ "${model}" == "pi0" ] || [ "${model}" == "pi0_fast" ]; then
                echo "12"  # 2(data) × 3(state) × 2(action) = 12
            else
                echo "12"  # 2(data) × 3(state) × 2(action) = 12 (univla)
            fi
            ;;
    esac
}

# 사용법 출력 함수
show_usage() {
    echo -e "${CYAN}🚀 VLA Models Ablation Study Script${NC}"
    echo ""
    echo -e "${YELLOW}사용법:${NC}"
    echo "  $0 [옵션] [모델명...]                # 특정 모델만 실행"
    echo "  $0                                  # 모든 모델 실행 (기본값)"
    echo "  $0 --help                           # 이 도움말 표시"
    echo ""
    echo -e "${YELLOW}옵션:${NC}"
    echo "  --max-steps STEPS, -s STEPS         최대 훈련 스텝 수 (기본값: ${DEFAULT_MAX_STEPS})"
    echo "  --save-interval STEPS, -c STEPS     체크포인트 저장 주기 (기본값: ${DEFAULT_SAVE_INTERVAL})"
    echo "  --input-dir PATH, -i PATH           데이터셋 경로 (기본값: allex_gesture_easy_pos_vel_torq)"
    echo ""
    echo -e "${YELLOW}지원되는 모델:${NC}"
    echo "  - gr00t     : GR00T 모델"
    echo "  - pi0       : PI0 모델" 
    echo "  - pi0_fast  : PI0_FAST 모델"
    echo "  - univla    : UniVLA 모델"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo "  $0                                  # 36개 조합 모두 실행 (기본값들)"
    echo "  $0 univla                           # UniVLA 12개 조합만 실행"
    echo "  $0 -s 8000 -c 2500 pi0 gr00t       # max_steps=8000, save_interval=2500로 PI0+GR00T 실행"
    echo "  $0 --input-dir \"/path/to/data\" gr00t  # 커스텀 데이터 경로로 GR00T 실행"
    echo ""
    echo -e "${YELLOW}각 모델당 실행되는 조합:${NC}"
    echo "  - 데이터: 30%, 100%"
    echo "  - 상태: pos_only, pos_vel, pos_vel_torque (GR00T는 pos_only만)"
    echo "  - 액션: right_arm, dual_arm"
    echo "  - 카메라: robot_view (UniVLA는 robot_view만)"
    echo "  - 총 조합 수:"
    echo "    * GR00T: 2 × 1 × 2 = 4개"
    echo "    * PI0: 2 × 3 × 2 = 12개"
    echo "    * PI0_FAST: 2 × 3 × 2 = 12개"
    echo "    * UniVLA: 2 × 3 × 2 = 12개 (robot_view만)"
    echo ""
    echo -e "${YELLOW}배치 사이즈 자동 설정:${NC}"
    echo "  - GR00T: 4 (GPU 메모리 제약으로 작은 배치)"
    echo "  - PI0: 24 (최적화된 배치 사이즈)"
    echo "  - PI0_FAST: 16 (중간 성능 배치 사이즈)"
    echo "  - UniVLA: 16 (안정적인 배치 사이즈)"
    echo ""
    echo -e "${YELLOW}훈련 스텝 권장사항:${NC}"
    echo "  - max_steps=1000   : 빠른 테스트용"
    echo "  - max_steps=5000   : 중간 훈련"
    echo "  - max_steps=10000  : 기본 훈련 (기본값)"
    echo "  - max_steps=20000+ : 장시간 훈련"
}

# 인자 파싱
MAX_STEPS=$DEFAULT_MAX_STEPS
SAVE_INTERVAL=$DEFAULT_SAVE_INTERVAL
INPUT_DIR=$DEFAULT_INPUT_DIR
MODELS_ARG=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --max-steps|-s)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                MAX_STEPS=$2
                shift 2
            else
                echo -e "${RED}❌ 오류: --max-steps 옵션에는 숫자를 입력해야 합니다.${NC}"
                echo -e "${YELLOW}예시: --max-steps 5000${NC}"
                exit 1
            fi
            ;;
        --save-interval|-c)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                SAVE_INTERVAL=$2
                shift 2
            else
                echo -e "${RED}❌ 오류: --save-interval 옵션에는 숫자를 입력해야 합니다.${NC}"
                echo -e "${YELLOW}예시: --save-interval 5000${NC}"
                exit 1
            fi
            ;;
        --input-dir|-i)
            if [[ -n $2 ]]; then
                INPUT_DIR=$2
                shift 2
            else
                echo -e "${RED}❌ 오류: --input-dir 옵션에는 경로를 입력해야 합니다.${NC}"
                echo -e "${YELLOW}예시: --input-dir \"/path/to/dataset\"${NC}"
                exit 1
            fi
            ;;
        -*)
            echo -e "${RED}❌ 오류: 알 수 없는 옵션 '$1'${NC}"
            show_usage
            exit 1
            ;;
        *)
            MODELS_ARG+=("$1")
            shift
            ;;
    esac
done

# 사용 가능한 모델들
AVAILABLE_MODELS=("gr00t" "pi0" "pi0_fast" "univla")

# 실행할 모델들 결정
if [ ${#MODELS_ARG[@]} -eq 0 ]; then
    # 인자가 없으면 모든 모델 실행
    MODELS=("${AVAILABLE_MODELS[@]}")
    echo -e "${PURPLE}📢 모델 인자가 없어서 모든 모델을 실행합니다.${NC}"
else
    # 입력된 모델들 검증
    MODELS=()
    for model in "${MODELS_ARG[@]}"; do
        # 유효한 모델인지 확인
        if [[ " ${AVAILABLE_MODELS[*]} " =~ " $model " ]]; then
            MODELS+=("$model")
        else
            echo -e "${RED}❌ 오류: '$model'는 지원되지 않는 모델입니다.${NC}"
            echo -e "${YELLOW}지원되는 모델: ${AVAILABLE_MODELS[*]}${NC}"
            echo ""
            show_usage
            exit 1
        fi
    done
    
    # 중복 제거
    MODELS=($(printf "%s\n" "${MODELS[@]}" | sort -u))
    echo -e "${PURPLE}📢 선택된 모델: ${MODELS[*]}${NC}"
fi

# 옵션들 검증
if [ $MAX_STEPS -lt 1 ]; then
    echo -e "${RED}❌ 오류: 최대 스텝 수는 1 이상이어야 합니다.${NC}"
    exit 1
fi

if [ $SAVE_INTERVAL -lt 1 ]; then
    echo -e "${RED}❌ 오류: 체크포인트 저장 주기는 1 이상이어야 합니다.${NC}"
    exit 1
fi

# 배치 사이즈 정보 표시
echo -e "${BLUE}🎯 배치 사이즈 자동 설정 (모델별):${NC}"
for model in "${MODELS[@]}"; do
    case $model in
        "gr00t")
            echo -e "${BLUE}   - GR00T: 4${NC}"
            ;;
        "pi0")
            echo -e "${BLUE}   - PI0: 24${NC}"
            ;;
        "pi0_fast")
            echo -e "${BLUE}   - PI0_FAST: 16${NC}"
            ;;
        "univla")
            echo -e "${BLUE}   - UniVLA: 16${NC}"
            ;;
    esac
done

# 데이터셋 경로 표시
if [ "$INPUT_DIR" = "$DEFAULT_INPUT_DIR" ]; then
    echo -e "${BLUE}�� 기본 데이터셋 사용: allex_gesture_easy_pos_vel_torq${NC}"
else
    echo -e "${CYAN}📁 커스텀 데이터셋: ${INPUT_DIR}${NC}"
fi

# 파라미터 정의 (ablation_config.py의 AblationDefaults와 일치)
DATA_AMOUNTS=(30 100)  # 일반적인 ablation study 퍼센트
STATE_TYPES=("pos_only" "pos_vel" "pos_vel_torque")
ACTION_TYPES=("right_arm" "dual_arm")
CAMERA_TYPES=("robot_view" "multi_view")  # 모든 카메라 타입 지원

# 통계 변수
TOTAL_COMBINATIONS=0
SUCCESSFUL_JOBS=0
FAILED_JOBS=0
SUBMITTED_JOBS=()
FAILED_CONDITIONS=()

# 총 조합 수 계산
for model in "${MODELS[@]}"; do
    COMBINATIONS_PER_MODEL=$(get_model_config "$model" "combinations")
    TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + COMBINATIONS_PER_MODEL))
done

echo ""
echo -e "${CYAN}🚀 VLA Models Ablation Study - 선택된 모델 실행${NC}"
echo "================================================================"
echo -e "${YELLOW}📊 실행 계획:${NC}"
echo "   - 모델: ${MODELS[*]} (${#MODELS[@]}개)"
echo "   - 최대 스텝: ${MAX_STEPS}"
echo "   - 체크포인트 저장 주기: ${SAVE_INTERVAL} 스텝"
echo "   - 데이터셋 경로: ${INPUT_DIR}"
echo "   - 데이터셋 이름: ${DEFAULT_DATASET_NAME}"
echo "   - 데이터: ${DATA_AMOUNTS[*]}%"
echo "   - 상태: ${STATE_TYPES[*]}"
echo "   - 액션: ${ACTION_TYPES[*]}"
echo "   - 카메라: ${CAMERA_TYPES[*]}"

echo ""
echo -e "${PURPLE}📈 총 ${TOTAL_COMBINATIONS}개 조합이 실행됩니다:${NC}"

# 실행할 조합들 미리보기
COMBINATION_NUM=0
for model in "${MODELS[@]}"; do
    # 모델별 state 제한사항 적용
    STATE_TYPES_MODEL=$(get_model_config "$model" "state")
    
    # 모델별 카메라 제한사항 적용
    CAMERA_TYPES_MODEL=$(get_model_config "$model" "camera")
    
    COMBINATIONS_PER_MODEL=$(get_model_config "$model" "combinations")
    echo ""
    echo -e "${BLUE}📋 ${model^^} 모델 (${COMBINATIONS_PER_MODEL}개 조합):${NC}"
    for data in "${DATA_AMOUNTS[@]}"; do
        for state in $STATE_TYPES_MODEL; do
            for action in "${ACTION_TYPES[@]}"; do
                for camera in $CAMERA_TYPES_MODEL; do
                    ((COMBINATION_NUM++))
                    condition_name="${model}_${data}_percent_${state}_${action}_${camera}"
                    echo "   ${COMBINATION_NUM}. ${condition_name}"
                done
            done
        done
    done
done

echo ""
echo -e "${YELLOW}⏸️  실행을 계속하려면 Enter를 누르세요 (Ctrl+C로 취소 가능)...${NC}"
read -r

echo ""
echo -e "${GREEN}🎯 Ablation Study 실행 시작!${NC}"
echo "================================================================"

# 시작 시간 기록
START_TIME=$(date +%s)
CURRENT_NUM=0

# 모든 조합 실행
for model in "${MODELS[@]}"; do
    echo ""
    echo -e "${BLUE}🤖 ${model^^} 모델 실행 중...${NC}"
    echo "----------------------------------------------------------------"
    
    for data in "${DATA_AMOUNTS[@]}"; do
        # 모델별 state 제한사항 적용
        STATE_TYPES_MODEL=$(get_model_config "$model" "state")
        
        # 모델별 카메라 제한사항 적용
        CAMERA_TYPES_MODEL=$(get_model_config "$model" "camera")
        
        for state in $STATE_TYPES_MODEL; do
            for action in "${ACTION_TYPES[@]}"; do
                for camera in $CAMERA_TYPES_MODEL; do
                    ((CURRENT_NUM++))
                    condition_name="${model}_${data}_percent_${state}_${action}_${camera}"
                
                echo ""
                echo -e "${CYAN}�� [${CURRENT_NUM}/${TOTAL_COMBINATIONS}] 실행 중: ${condition_name}${NC}"
                
                # run_ablation.py 명령 구성
                cmd="python run_ablation.py --model ${model} --data ${data} --state ${state} --action ${action} --camera ${camera} --sbatch"
                
                # 추가 옵션들 적용
                cmd="$cmd --max-steps ${MAX_STEPS}"
                cmd="$cmd --input-dir \"${INPUT_DIR}\""
                
                # checkpoint 저장 주기 설정 (모든 모델 동일)
                cmd="$cmd --save-interval ${SAVE_INTERVAL}"
                
                echo -e "${YELLOW}   💻 Command: ${cmd}${NC}"
                
                # 명령 실행 및 결과 캡처
                if output=$(eval $cmd 2>&1); then
                    echo -e "${GREEN}   ✅ 성공적으로 제출됨${NC}"
                    SUBMITTED_JOBS+=("$condition_name")
                    ((SUCCESSFUL_JOBS++))
                    
                    # Job ID 추출 시도
                    if echo "$output" | grep -q "Submitted batch job\|Job ID:"; then
                        job_info=$(echo "$output" | grep -E "Submitted batch job|Job ID:" | head -1)
                        echo -e "${GREEN}   📝 ${job_info}${NC}"
                    fi
                else
                    echo -e "${RED}   ❌ 실패${NC}"
                    echo -e "${RED}   🔍 Error: ${output}${NC}"
                    FAILED_CONDITIONS+=("$condition_name")
                    ((FAILED_JOBS++))
                fi
                
                # 다음 실행 전 잠시 대기 (시스템 부하 방지)
                if [ $CURRENT_NUM -lt $TOTAL_COMBINATIONS ]; then
                    echo -e "${YELLOW}   ⏳ 3초 대기 중...${NC}"
                    sleep 3
                fi
                done
            done
        done
    done
done

# 종료 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================================"
echo -e "${GREEN}📊 VLA Models Ablation Study 실행 완료!${NC}"
echo -e "${YELLOW}⏰ 실행 시간: ${MINUTES}분 ${SECONDS}초${NC}"
echo -e "${YELLOW}🕐 완료 시간: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BLUE}🎯 사용된 설정:${NC}"
echo -e "${BLUE}   - 최대 스텝: ${MAX_STEPS}${NC}"
echo -e "${BLUE}   - 체크포인트 저장 주기: ${SAVE_INTERVAL} 스텝${NC}"
echo -e "${BLUE}   - 데이터셋 경로: ${INPUT_DIR}${NC}"
echo -e "${BLUE}   - 데이터셋 이름: ${DEFAULT_DATASET_NAME}${NC}"
echo ""

# 성공한 작업 표시
echo -e "${GREEN}✅ 성공적으로 제출된 작업: ${SUCCESSFUL_JOBS}개${NC}"
if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo -e "${GREEN}   - ${job}${NC}"
    done
fi

# 실패한 작업 표시
if [ $FAILED_JOBS -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ 실패한 작업: ${FAILED_JOBS}개${NC}"
    for job in "${FAILED_CONDITIONS[@]}"; do
        echo -e "${RED}   - ${job}${NC}"
    done
fi

echo ""
echo -e "${BLUE}🔍 유용한 명령어들:${NC}"
echo -e "${CYAN}   # 작업 상태 확인${NC}"
echo "   squeue -u \$(whoami)"
echo ""
echo -e "${CYAN}   # 로그 확인${NC}"
echo "   ls -la logs/*/"
echo "   tail -f logs/univla/job_XXX_*.log"
echo "   tail -f logs/pi0/job_XXX_*.log"
echo "   tail -f logs/gr00t/job_XXX_*.log"
echo ""
echo -e "${CYAN}   # 실행 중인 작업 취소 (필요시)${NC}"
echo "   scancel \$(squeue -u \$(whoami) -h -o %i)"
echo ""
echo -e "${GREEN}🎉 Happy Training! 🤖${NC}" 