# LeRobot 데이터셋 변환 및 UniVLA 훈련 통합 스크립트

이 shell 스크립트는 LeRobot 데이터셋을 UniVLA 형식으로 변환하고, 변환된 데이터로 UniVLA 모델을 훈련하는 과정을 자동화합니다.

## 기능

1. **데이터 변환**: LeRobot 데이터셋을 UniVLA 훈련에 필요한 형식으로 변환
2. **모델 훈련**: 변환된 데이터를 사용하여 UniVLA 모델 훈련
3. **경로 자동 설정**: 입력/출력 데이터 경로를 동적으로 설정
4. **스마트 실행**: 이미 변환된 데이터가 있으면 변환 과정을 건너뛰고 훈련으로 진행
5. **오류 처리**: 각 단계별 오류 검사 및 처리
6. **색상 출력**: 진행 상황을 색상으로 구분하여 표시

## 사용법

### 기본 사용법

```bash
# SLURM 작업으로 실행 (입력 데이터 경로만 지정)
sbatch run_conversion_and_training_david.sh /path/to/lerobot/dataset

# 또는 직접 실행
bash run_conversion_and_training_david.sh /path/to/lerobot/dataset
```

### 고급 사용법

```bash
# 모든 매개변수 지정
bash run_conversion_and_training_david.sh /path/to/input /path/to/output univla_train

# 출력 경로만 변경
bash run_conversion_and_training_david.sh /path/to/input /path/to/output

# conda 환경만 변경
bash run_conversion_and_training_david.sh /path/to/input "" gr00t
```

### 매개변수

| 위치 | 매개변수 | 필수 | 기본값 | 설명 |
|------|----------|------|--------|------|
| 1 | INPUT_DATA | ✅ | - | 변환할 LeRobot 데이터셋 경로 |
| 2 | OUTPUT_DATA | ❌ | `./converted_data_for_univla` (스크립트 디렉토리 기준) | 변환된 데이터 저장 경로 |
| 3 | CONDA_ENV | ❌ | `univla_train` | 사용할 conda 환경 이름 |

### 사용 예시

```bash
# 기본 사용
bash run_conversion_and_training_david.sh /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_rightarm_truncated

# 다른 출력 경로 지정
bash run_conversion_and_training_david.sh /path/to/input /path/to/converted_data

# 다른 conda 환경 사용
bash run_conversion_and_training_david.sh /path/to/input "" gr00t

# 모든 매개변수 지정
bash run_conversion_and_training_david.sh /path/to/input /path/to/output gr00t
```

## 스크립트 동작 과정

1. **환경 설정**: conda 환경 활성화
2. **입력 데이터 확인**: 변환할 데이터 경로 존재 확인
3. **출력 디렉토리 생성**: 변환된 데이터 저장 경로 생성
4. **변환된 데이터 확인**: 이미 변환된 데이터가 있는지 확인
5. **조건부 변환**: 
   - 데이터가 있으면 → 변환 과정 건너뛰기
   - 데이터가 없으면 → 변환 과정 실행
6. **최종 데이터 확인**: 변환된 데이터 존재 여부 재확인
7. **모델 훈련**: UniVLA 모델 훈련 실행
8. **완료**: 결과 출력

## 출력 예시

```
========================================
환경 설정
========================================
[SUCCESS] Conda 환경 활성화: univla_train

========================================
입력 데이터 확인
========================================
[SUCCESS] 입력 데이터 경로 확인: /path/to/input

========================================
변환된 데이터 확인
========================================
[WARNING] 변환된 데이터가 이미 존재합니다: /path/to/output
[INFO] 데이터 변환 단계를 건너뛰고 훈련으로 진행합니다.

========================================
1단계: 데이터 변환 (건너뛰기)
========================================

========================================
2단계: UniVLA 모델 훈련
========================================
[INFO] 훈련 스크립트 경로 수정 완료
[INFO] 모델 훈련 시작...
```

## 주요 특징

### 🚀 **스마트 실행**
- 이미 변환된 데이터가 있으면 변환 과정을 자동으로 건너뜀
- 시간 절약 및 중복 작업 방지

### 🛡️ **안전한 파일 수정**
- 스크립트 실행 전 원본 파일 백업
- 오류 발생 시 자동으로 원본 파일 복원
- 실행 완료 후 백업 파일 제거

### 🎨 **시각적 피드백**
- 색상으로 구분된 상태 메시지
- 진행 단계별 명확한 구분
- 성공/경고/오류 상태 표시

### ⚡ **SLURM 호환**
- SLURM 작업 스케줄러와 완벽 호환
- GPU 할당 및 로그 관리
- 배치 작업으로 실행 가능

## 주의사항

1. **권한**: 출력 디렉토리에 쓰기 권한이 있어야 합니다.
2. **환경**: 적절한 conda 환경이 설정되어 있어야 합니다.
3. **GPU**: 훈련에는 GPU가 필요합니다.
4. **메모리**: 충분한 메모리가 필요합니다.
5. **파일 백업**: 스크립트 실행 중 원본 파일이 임시로 수정됩니다.

## 오류 해결

### PermissionError 발생 시

출력 경로를 사용자가 쓰기 권한을 가진 경로로 변경하세요:

```bash
OUTPUT_DATA="/home/username/converted_data"
```

### Conda 환경 오류 시

올바른 conda 환경 이름을 지정하세요:

```bash
CONDA_ENV="correct_env_name"
```

### 데이터 변환 실패 시

입력 데이터 경로가 올바른지 확인하세요:

```bash
ls -la /path/to/lerobot/dataset
```

## 파일 구조

```
VLA_models_training/univla/vla_scripts/
├── run_conversion_and_training_david.sh    # 메인 통합 스크립트
├── convert_lerobot_dataset_for_univla.py   # 데이터 변환 스크립트
├── finetune_rlwrld.py                      # 모델 훈련 스크립트
└── README_integrated_script.md             # 이 파일
``` 