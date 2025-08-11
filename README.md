# VLA_models_training

## Ablation Runner 소개 (`run_ablation_study.py`)
VLA 계열 모델들에 대해 상태(state)/액션(action)/비디오(video) 조합을 자동으로 생성하고, SLURM 잡으로 일괄 학습을 제출하는 러너입니다. 모델별로 필요한 인덱스를 자동 계산하고, 체크포인트/로그 경로도 통일된 규칙으로 관리합니다.

- 지원 모델: `gr00t`, `pi0`, `pi0fast`, `univla`, `diffusion`, `act`
- 비디오 모드 지원: `gr00t`, `pi0`, `pi0fast`, `diffusion`, `act`는 `robotview | multiview`를 사용. `univla`는 내부적으로 `robotview` 고정
- 상태 모드: `pos_only | pos_vel`
- 액션 모드: `right_arm | dual_arm` (단, 로봇/데이터셋 타입에 따라 제한됨)
- 자동 감지:
  - 로봇: 데이터 경로에 `franka`가 포함되면 `franka`, 그렇지 않으면 `allex`
  - 시뮬/리얼: 경로에 `sim`이 있으면 `sim`, `real`이 있으면 `real`, 둘 다 없으면 기본 `sim` (allex sim 데이터는 관례상 경로에 `sim`이 빠짐)

### 산출물 경로 규칙
- 체크포인트: `_checkpoints/<모델>/<태스크>/<state>_<action>_<video(또는 robotview)>/`
- 로그: `_logs/<모델>/<태스크>/slurm-<JOBID>-<JOBNAME>.log`
  - `JOBNAME`에는 조합 정보가 들어갑니다. 로그 디렉토리 구조는 모델/태스크 기준으로 고정됩니다.

## 빠른 시작

### 1) 모든 모델 일괄 실행 (모든 조합)
```bash
python run_ablation_study.py \
  --run_mode_all \
  --data_dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube
```
- `pi0`/`pi0fast`/`diffusion`/`act` 계열은 필요한 경우 데이터셋 변환을 먼저 수행한 뒤 학습을 제출합니다.

### 2) 특정 모델만 모든 조합 실행
- `univla` 예시:
```bash
python run_ablation_study.py \
  --run_mode_all \
  --vla_model univla \
  --data_dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube
```
- `pi0` 예시(비디오 모드 자동 포함):
```bash
python run_ablation_study.py \
  --run_mode_all \
  --vla_model pi0 \
  --data_dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube
```

### 3) 단일 조합 실행
```bash
python run_ablation_study.py \
  --vla_model gr00t \
  --data_dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube \
  --state_mode pos_vel \
  --action_mode right_arm \
  --video_mode multiview
```
- `univla`는 `--video_mode`가 필요 없습니다.

### 4) (선택) pi0 계열 데이터셋 변환만 먼저 수행
```bash
python run_ablation_study.py \
  --pi0_dataset_convert \
  --data_dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube
```

## 명령행 인자 요약
- **--vla_model**: 사용할 모델 이름. `gr00t | pi0 | pi0fast | univla | diffusion | act`
- **--data_dir**: 원본 데이터 경로. 로봇/시뮬 여부 자동 감지에 사용
- **--state_mode**: `pos_only | pos_vel`
- **--action_mode**: `right_arm | dual_arm` (로봇/데이터셋에 따라 제한)
- **--video_mode**: `robotview | multiview` (비디오 모드 지원 모델에만 해당)
- **--run_mode_all**: 지정된 모델(또는 전체 모델)에 대해 가능한 모든 조합을 SLURM으로 제출
- **--pi0_dataset_convert**: `pi0`/`pi0fast`/`diffusion`/`act` 변환을 모든 조합으로 선행 실행

## 모니터링과 결과 확인
- 제출 후 잡 개수/상태 확인: `squeue -u $USER`
- 로그: `_logs/<모델>/<태스크>/slurm-<JOBID>-<JOBNAME>.log`
- 체크포인트: `_checkpoints/<모델>/<태스크>/<조합명>/`

## 주의 사항
- `allex` 실데이터(`real`)는 제약: `action_mode=right_arm`, `video_mode=robotview`만 허용
- `franka`는 단일 팔 로봇: `action_mode=right_arm`만 허용
- GPU 사용 수: `univla=2`, 그 외 단일 GPU. SLURM 파티션/자원 정책에 맞추어 제출됩니다.

### 작동 방식(내부 로직 개요)
1) 입력 경로(`--data_dir`)로부터 로봇/시뮬-리얼을 자동 감지합니다.
   - 경로에 `franka` 포함 시 `franka`, 아니면 `allex`
   - 경로에 `sim` 있으면 `sim`, `real` 있으면 `real`, 둘 다 없으면 기본 `sim`으로 처리(allex sim 데이터는 관례상 `sim`이 경로에 없음)
2) 로봇/데이터셋 타입에 따라 사용 가능한 `action_mode`/`video_mode` 목록을 제약합니다.
3) `modality_*.json`을 로딩해 상태/액션 인덱스를 계산합니다.
4) 모델별 실행 스크립트를 선택합니다.
   - `gr00t`: `gr00t/slurm_ft_ablation_study.sh`
   - `pi0|pi0fast|diffusion|act`: `pi0/slurm_pi0_train_ablation_study.sh`
   - `univla`: `univla/vla_scripts/run_conversion_and_training_david.sh`
5) SLURM 잡을 제출합니다. 잡 이름에는 모델/조합 정보가 포함되고, 로그는 `_logs/<모델>/<태스크>/`에 저장됩니다.
6) 체크포인트는 `_checkpoints/<모델>/<태스크>/<state>_<action>_<video(또는 robotview)>/` 경로에 생성됩니다.

### 인자 상세
- `--vla_model`: 특정 모델만 실행할 때 지정. 생략하면 `--run_mode_all`일 때 전체 모델을 순차 실행합니다.
- `--data_dir`: 원본 데이터 경로. 경로명으로 로봇/시뮬-리얼을 판단하며, 조합별 태스크명은 마지막 디렉토리명으로 정합니다.
- `--state_mode`: 단일 조합 실행 모드에서 필수(`--run_mode_all`이 아니고, `--pi0_dataset_convert`도 아닐 때).
- `--action_mode`: 위와 동일한 조건에서 필수. 로봇 제약을 위반하면 즉시 에러를 반환합니다.
- `--video_mode`: 비디오 모드가 있는 모델에만 적용(`gr00t|pi0|pi0fast|diffusion|act`). `univla`는 내부적으로 `robotview` 고정.
- `--run_mode_all`: 지정된 모델(또는 전체 모델)에 대해 가능한 모든 조합을 자동 제출.
- `--pi0_dataset_convert`: `pi0` 계열용 데이터셋 변환만 선행 일괄 실행.

### 모델별 동작/제약
- `univla`
  - 멀티 GPU(2개) 사용 가정으로 제출됩니다.
  - 최초 1개 조합에서 데이터 변환을 트리거하고, 변환 완료(또는 타임아웃) 후 나머지 조합을 제출합니다.
  - `video_mode` 인자를 받지 않으며 내부적으로 `robotview`로 고정된 조합명을 사용합니다.
- `pi0|pi0fast|diffusion|act`
  - 필요 시 모든 조합에 대해 변환 스크립트를 먼저 실행한 뒤 학습을 제출합니다.
  - 변환 산출물은 `RLWRLD/pi0/<태스크>/<약어조합명>/meta/info.json` 등의 존재로 완료 여부를 판단합니다.
- `gr00t`
  - 비디오 모드를 사용하며 조합별로 1 GPU로 제출됩니다.

### 디렉토리/파일 구조 예시
```
VLA_models_training/
  run_ablation_study.py
  _logs/
    pi0/
      lift_cube/
        slurm-1234567-pi0_pos_vel_right_arm_multiview.log
  _checkpoints/
    pi0/
      lift_cube/
        PV_R_M/   # 예: pos_vel + right_arm + multiview (약어 예시는 pi0 변환 폴더명 컨벤션)
    univla/
      lift_cube/
        pos_vel_right_arm_robotview/
```

### 자주 만나는 오류와 해결
- Data directory does not exist: `--data_dir` 경로가 실제로 존재하는지 확인하세요.
- Unsupported action_mode/video_mode: 로봇/데이터 타입 제약을 위반했습니다. `allex real`은 `right_arm + robotview`만 허용, `franka`는 `right_arm`만 허용.
- Training script not found: 모델별 스크립트가 누락되었을 수 있습니다. 레포 내 해당 경로가 존재하는지 확인하세요.
- sbatch 관련 오류: SLURM 환경에서만 동작합니다. 로컬에서 테스트 시에는 SLURM 노드에서 실행하거나 스크립트를 직접 실행하도록 수정이 필요합니다.

### 운영 팁
- 잡 모니터링: `squeue -u $USER`
- 로그 tail: `tail -f _logs/<모델>/<태스크>/slurm-<JOBID>-<JOBNAME>.log`
- 체크포인트 정리 예시: `run_ablation_study.py` 하단의 주석 예시를 참고해 불필요한 step 폴더를 정리하세요.

---

## 개별 모델 훈련 가이드 (세부)
아래 내용은 각 모델을 단독으로 설정/훈련/추론하는 방법입니다. Ablation Runner로 일괄 학습을 돌리지 않고, 수동으로 진행해야 할 때 참고하세요.

## 추론을 위한 omni-pilot 환경 설정
### uv 설치
```sh
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc
uv --version
```

### omni-pilot 받기
```sh
git clone git@github.com:RLWRLD/omni-pilot.git
```

* omni-pilot 내부 수정
omni-pilot/packages/pi0/pyproject.toml 에서 아래 내용 수정해야 ssh key를 사용하는 방식으로 git 받아짐
```
#"lerobot[pi0] @ git+https://github.com/RLWRLD/lerobot_research@pi0-allex",
"lerobot[pi0] @ git+ssh://git@github.com/RLWRLD/lerobot_research.git@pi0-allex",
```

* omni-pilot 설치
```sh
cd omni-pilot/packages/gr00t
uv venv
source .venv/bin/activate
cd ../..
uv pip install setuptools
uv pip install torch
uv pip install psutil
GIT_LFS_SKIP_SMUDGE=1 make sync
```

-------------------------------------
-------------------------------------
## [gr00t]
### gr00t-1-1. 훈련 환경 설정
* slurm에서 수행
```sh
cd gr00t
conda create -n gr00t_vla python=3.10
conda activate gr00t_vla
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install --upgrade jax jaxlib ml_dtypes
pip install tokenizers
```
> 다음 에러는 무시해도 됨:
"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorboard 2.19.0 requires packaging, which is not installed.
tensorboard 2.19.0 requires six>1.9, which is not installed."
"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gdown 5.2.0 requires beautifulsoup4, which is not installed.
lerobot 0.1.0 requires pyserial>=3.5, which is not installed.
datasets 3.6.0 requires pyarrow>=15.0.0, but you have pyarrow 14.0.1 which is incompatible.
jax 0.6.2 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible.
jaxlib 0.6.2 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible.
lerobot 0.1.0 requires av>=14.2.0, but you have av 12.3.0 which is incompatible.
lerobot 0.1.0 requires gymnasium==0.29.1, but you have gymnasium 1.0.0 which is incompatible.
lerobot 0.1.0 requires torchvision>=0.21.0, but you have torchvision 0.20.1 which is incompatible.
rerun-sdk 0.23.1 requires pyarrow>=14.0.2, but you have pyarrow 14.0.1 which is incompatible.
tensorstore 0.1.75 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible."

### gr00t-1-2. 훈련
   #### 1-2-1. dataset 경로
   * '/demo_data' 아래 데이터 옮기기
   * /demo_data/allex_cube/meta/modality.json 있는지 확인!
   
   #### 1-2-2. 실행 스크립트 작성
   * VLA_models_training/gr00t/slurm_ft_allex_bimanual_cube.sh
   ```sh
   #!/bin/bash
   #SBATCH --job-name=gr00t-n1.5-ft-allex-bimanual-cube
   #SBATCH --output=tmp/slurm-%j-%x.log
   #SBATCH --partition=batch
   #SBATCH --gpus=1
   
   # Conda 초기화 및 환경 활성화
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate gr00t_vla
   echo "✅ Conda environment 'gr00t_vla' activated."
   
   mkdir -p tmp 2>/dev/null
   mkdir -p checkpoints 2>/dev/null
   
   python scripts/gr00t_finetune.py \
      --dataset-path ./demo_data/allex_cube \
      --num-gpus 1 \
      --output-dir checkpoints/allex-bimanual-cube2  \
      --max-steps 8000 \
      --data-config allex_bimanual \
      --video-backend torchvision_av \
      --action_dim 42 \
      > tmp/slurm-$SLURM_JOB_ID-policy.log 2>&1
   
   # --dataset-path 는 데이터 존재하는 path, 보통 demo_data 아래 있음
   # --output-dir 는 생성될 checkpoint 폴더명
   # --data-config 는 demo_data아래 있는 폴더명
   ```
   
   #### 1-2-3. 훈련 실행
   ```sh
   chmod +x slurm_ft_allex_bimanual_cube.sh	# 실행 권한 부여 (한번만)
   sbatch --comment "gr00t training" slurm_ft_allex_bimanual_cube.sh	# slurm을 통한 훈련 실행
   squeue --me	# job 돌아가는지 확인
   tail -f tmp/slurm-$SLURM_JOB_ID-policy.log	# 로그 확인
   ```
   * checkpoints/ 아래 allex-bimanual-cube2 폴더 생성되는지 확인

-------------------------------------

### gr00t-1-3. 추론 환경 설정
* gpu2 (david@172.30.1.102)에서 수행
* gr00t 설치 -> 위 방법 참조
* omni-pilot 설치 -> 위 방법 참조

### gr00t-1-4. 추론 실행
   #### 1-4-1. 학습된 checkpoints 복사 받기
   ```sh
   scp -r david@61.109.237.73:/virtual_lab/rlwrld/david/VLA_models_training/gr00t/checkpoints VLA_models_training/gr00t/artifact
   ```
   
   #### 1-4-2. Isaac-GR00T에서 Inference 서버 실행 (터미널 1)
   ```sh
   conda activate gr00t_vla
   python scripts/inference_service.py --server --model_path artifact/checkpoints/allex-bimanual-cube2 --embodiment_tag new_embodiment --data_config allex_cube --port 7777
   ```
   
   #### 1-4-3. omni-pilot에서 robosuite 시뮬레이터 실행 (터미널 2)
   ```sh
   cd omni-pilot
   source packages/gr00t/.venv/bin/activate
   bin/python packages/gr00t/gr00t_control.py --data-config allex_cube --env-name LiftOurs --task-instruction "Lift the cube from the left stand and place it on the right stand." --episode-horizon 1000 --save-log true --save-video true --no-render --port 7777
   ```

-------------------------------------
-------------------------------------

## [pi0]
### pi0-1-1. 훈련 환경 설정
   #### 1-1-1. pi0 conda 환경 설정
   ```sh
   cd pi0
   conda create -y -n lerobot_vla python=3.10 \
pytorch torchvision torchaudio pytorch-cuda=12.1 \
ffmpeg \
-c pytorch -c nvidia -c conda-forge -y
   conda activate lerobot_vla
   # conda install ffmpeg -c conda-forge
   pip install -e . 
   pip install tensorboard absl-py jax dm-tree
   pip install -e ".[pi0,pi0fast,test]"
   ```
   > libcusparseLt.so.0 관련 에러 나면,
   > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   > conda install ffmpeg -c conda-forge

   #### 1-1-1. huggingface에 있는 weight 사용 권한 받기
   https://huggingface.co/google/paligemma-3b-pt-224 들어가서 Authorize 관련 버튼 누르고 권한 신청
   > 아래 에러 대처: "OSError: You are trying to access a gated repo.
   Make sure to have access to it at https://huggingface.co/google/paligemma-3b-pt-224.
   403 Client Error. (Request ID: Root=1-6863a994-15d0531859f926764c4c3aec;91a0587c-5839-4ef7-a90c-561cd0549b2e)
   Cannot access gated repo for url https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/config.json.
   Access to model google/paligemma-3b-pt-224 is restricted and you are not in the authorized list. Visit https://huggingface.co/google/paligemma-3b-pt-224 to ask for access."
   
   #### 1-1-2. 데이터셋 로드 관련 대처
   * huggingface cli 설치
   ```sh
   pip install huggingface_hub
   ```
   * huggingface cli 로그인
   ```sh
   huggingface-cli login
   ```
   * https://huggingface.co/settings/tokens 에서 token 발급 후 입력
   
   #### 1-1-3. state, action 데이터 변환 스크립트
   https://rlwrld.slack.com/files/U08SQQ41RFC/F08UTDE0M5W/lift_dataset_convert.ipynb?origin_team=T077WTVBF8W&origin_channel=D08SQNR9JS2

### pi0-1-2. 훈련
   #### 1-2-1. dataset 경로
   ```sh
   conda activate lerobot_vla
   python3
   > from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
   LeRobotDataset('RLWRLD/put_cube')
   exit()
   ```
   ```sh
   cd .cache/huggingface/lerobot/RLWRLD/put_cube
   ```
   * .cache/huggingface/lerobot/RLWRLD/put_cube 대신에 .cache/huggingface/lerobot/RLWRLD/allex_cube로 폴더 만들기
   * 위 경로에 데이터 넣기
   * 훈련 실행할 때, --dataset.repo_id=RLWRLD/allex_cube 설정
   
   #### 1-2-2. 실행 스크립트 작성
   * VLA_models_training/pi0/slurm_ft_pi0_bimanual_cube.sh
   ```sh
   #!/bin/bash
   #SBATCH --job-name=pi0-ft-allex-bimanual-cube
   #SBATCH --output=tmp/slurm-%j-%x.log
   #SBATCH --partition=batch
   #SBATCH --gpus=1
   
   # Conda 초기화 및 환경 활성화
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate lerobot_vla
   echo "✅ Conda environment 'lerobot_vla' activated."
   
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
   ```
   
   #### 1-2-3. 훈련 실행
   ```sh
   chmod +x slurm_ft_pi0_bimanual_cube.sh	# 실행 권한 부여 (한번만)
   sbatch --comment "pi0 training" slurm_ft_pi0_bimanual_cube.sh	# slurm을 통한 훈련 실행
   squeue --me	# job 돌아가는지 확인
   tail -f tmp/slurm-$SLURM_JOB_ID.log	# 로그 확인
   ```
   * pi0/outputs/train/ 아래 2025-07-04/11-25-08_pi0-ft-allex-bimanual-cub 폴더 생성되는지 확인

-------------------------------------

### pi0-1-3. 추론 환경 설정
* slurm에서 수행
* omni-pilot 설치 -> 위 방법 참조
  
### pi0-1-4. 추론 실행
```sh
source packages/pi0/.venv/bin/activate
bin/python packages/pi0/pi0_control.py --data-config allex_cube --env-name LiftOurs --policy-path /virtual_lab/rlwrld/david/VLA_models_training/pi0/outputs/train/2025-07-04/11-25-08_pi0-ft-allex-bimanual-cube/checkpoints/last/pretrained_model --task-instruction "Lift the cube from the left stand and place it on the right stand." --episode-horizon 1000 --save-log true --save-video true --no-render --fps 20
```

-------------------------------------
-------------------------------------

## [univla]
### univla-1-1. 훈련 환경 설정
   #### 1-1-1. univla conda 환경 설정
   ```sh
   cd univla
   conda create -n univla_vla python=3.10 -y
   conda activate univla_vla
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install -e .
   pip install packaging ninja
   pip install "flash-attn==2.5.5" --no-build-isolation
   pip install pytz
   pip install pyarrow
   pip install braceexpand
   pip install webdataset
   pip install --upgrade jax jaxlib ml_dtypes
   pip install tokenizers==0.19.1
   pip install wandb dm-tree pandas
   ```
   
   #### 1-1-2. 기 훈련된 latent action model & vision large model 받기
   ```sh
   conda activate univla_vla
   cd vla_scripts
   git lfs install
   git clone https://huggingface.co/qwbu/univla-7b
   git clone https://huggingface.co/qwbu/univla-latent-action-model
   ```
   
   #### 1-1-3. 데이터셋 변환 관련
   * convert_lerobot_dataset_for_univla.py에서 어떤 view를 구성하여 데이터 변환할지 정함
   ```sh
   cd vla_script
   python3 convert_lerobot_dataset_for_univla.py
   ```
   * 코드 내부에서 변환된 데이터셋 저장 경로 설정

### univla-1-2. 훈련
   #### 1-2-1. dataset 경로
   dataset 경로에 있음
   * 위 경로에 데이터 넣기
   
   #### 1-2-2. 실행 스크립트 작성
   * slurm_ft_univla_bimanual_cube.sh
   ```sh
   #!/bin/bash
   
   #SBATCH --job-name=univla-ft-allex-bimanual-cube
   #SBATCH --output=tmp/slurm-%j-%x.log
   #SBATCH --partition=batch
   #SBATCH --gpus=1
   
   # srun --gpus=1 --nodes=1 --pty /bin/bash
   
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate univla_vla
   echo "✅ Conda environment 'univla_vla' activated."
   
   torchrun --standalone --nnodes 1 --nproc-per-node 1 vla_scripts/finetune_rlwrld.py \
       --data_root_dir "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_single_view_converted_state_action" \
       --batch_size 16 \
       --max_steps 20000 \
       --run_id_note "allex_state_action_filter_side_view" \
   
      # --data_root_dir 는 데이터 존재하는 path, 보통 dataset 아래 있음
      # --run_id_note 는 checkout 이름 라벨 (optional)
   ```
   
   #### 1-2-3. 훈련 실행
   ```sh
   chmod +x slurm_ft_univla_bimanual_cube.sh	# 실행 권한 부여 (한번만)
   sbatch --comment "univla training" slurm_ft_univla_bimanual_cube.sh	# slurm을 통한 훈련 실행
   squeue --me	# job 돌아가는지 확인
   tail -f tmp/slurm-$SLURM_JOB_ID.log	# 로그 확인
   ```
   * vla_scripts/ 아래 runs 폴더 생성되는지 확인

-------------------------------------

### univla-1-3. 추론 환경 설정
* slurm에서 수행
* omni-pilot 설치 -> 위 방법 참조
  
### univla-1-4. 추론 실행
```sh
source packages/univla/.venv/bin/activate
bin/python packages/univla/univla_control.py --data-config allex_cube --env-name LiftOurs --robot-name Allex --policy-path /virtual_lab/rlwrld/david/UniVLA/vla_scripts/runs/univla-7b+real_world+b32+lr-0.00035+lora-r32+dropout-0.0--allex_state_action_filter_single_view=w-LowLevelDecoder-ws-12 --checkpoint-step 20000 --task-instruction "Lift the cube from the left stand and place it on the right stand." --episode-horizon 1000 --save-log true --save-video true --no-render --fps 20
```
-------------------------------------
-------------------------------------



