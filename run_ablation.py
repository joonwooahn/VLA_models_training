#!/usr/bin/env python
"""
Unified Ablation Study Runner for VLA Models
Complete factorial design: 3 models × 2 data × 2 state × 2 action × 2 camera = 48 experiments
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from ablation_config import (
    ABLATION_CONDITIONS, 
    get_condition_by_name, 
    list_all_conditions,
    get_conditions_by_model,
    print_conditions_summary,
    AblationCondition,
    ModelType,
    ActionType,
    CameraType,
    DataAmount,
    StateType
)


def create_gr00t_command(condition):
    """Create command for GR00T training"""
    data_config = condition.get_data_config()
    
    # GR00T 훈련 명령 (conda 환경 활성화 포함)
    data_config_name = "allex_cube" if condition.action_type == ActionType.SINGLE_ARM else "allex_bimanual_cube"
    num_episodes_arg = f"--num-episodes {data_config.get_num_episodes()}" if data_config.get_num_episodes() else ""
    
    cmd = [
        "bash", "-c",
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate gr00t && python gr00t/scripts/gr00t_finetune.py --dataset-path ./demo_data/allex_cube --num-gpus 1 --output-dir checkpoints/{condition.get_output_dir()} --max-steps {condition.max_steps} --batch-size {condition.batch_size} --video-backend torchvision_av --data-config {data_config_name} --action_dim {condition.get_action_dim()} {num_episodes_arg}"
    ]
    
    return cmd


def create_pi0_command(condition):
    """Create command for PI0 training"""
    data_config = condition.get_data_config()
    
    # PI0 훈련 명령 (conda 환경 활성화 포함)
    episodes_arg = f"dataset.episodes=[0,{data_config.get_num_episodes()-1}]" if data_config.get_num_episodes() else ""
    
    cmd = [
        "bash", "-c",
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && python pi0/lerobot/scripts/train.py dataset.repo_id={data_config.dataset_name} policy=pi0 output_dir=outputs/{condition.get_output_dir()} steps={condition.max_steps} batch_size={condition.batch_size} seed=1000 {episodes_arg}"
    ]
    
    return cmd


def create_univla_command(condition, use_sbatch=False):
    """Create command for UniVLA training"""
    data_config = condition.get_data_config()
    
    # 데이터 변환 명령 (conda 환경 활성화 포함)
    convert_cmd = [
        "bash", "-c", 
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py --condition {condition.name} --input-dir /home/david/.cache/huggingface/lerobot/RLWRLD/allex_cube --output-dir ./converted_data"
    ]
    
    if use_sbatch:
        # SBATCH 스크립트 생성
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=univla-{condition.name}
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh
conda activate univla_train

# 훈련 실행
python univla/vla_scripts/finetune_rlwrld_ablation.py \\
    --condition {condition.name} \\
    --data-root-dir ./converted_data/{condition.name} \\
    --output-dir ./outputs/{condition.get_output_dir()} \\
    --max-steps {condition.max_steps} \\
    --batch-size {condition.batch_size} \\
    --learning-rate {condition.learning_rate}
"""
        return {"convert": convert_cmd, "sbatch_script": sbatch_script}
    else:
        # 직접 실행 명령 (워커노드에서 실행용)
        train_cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python univla/vla_scripts/finetune_rlwrld_ablation.py --condition {condition.name} --data-root-dir ./converted_data/{condition.name} --output-dir ./outputs/{condition.get_output_dir()} --max-steps {condition.max_steps} --batch-size {condition.batch_size} --learning-rate {condition.learning_rate}"
        ]
        return {"convert": convert_cmd, "train": train_cmd}


def run_condition(condition, dry_run=False, use_sbatch=False):
    """Run a single ablation condition"""
    
    print(f"\n{'='*80}")
    print(f"Running condition: {condition.name}")
    print(f"Model: {condition.model_type.value}")
    print(f"Data: {'20 percent' if condition.data_amount == DataAmount.PERCENT_20 else '100 percent'}")
    print(f"State: {'pos_only' if condition.state_type == StateType.POSITION_ONLY else 'full_state'}")
    print(f"Action: {'single_arm' if condition.action_type == ActionType.SINGLE_ARM else 'bimanual'}")
    print(f"Camera: {'robot_view' if condition.camera_type == CameraType.ROBOT_VIEW else 'multi_view'}")
    print(f"Action dim: {condition.get_action_dim()}")
    print(f"Output: {condition.get_output_dir()}")
    print(f"{'='*80}")
    
    try:
        # Create command based on model type
        if condition.model_type == ModelType.GR00T:
            cmd = create_gr00t_command(condition)
            print(f"Command: {' '.join(cmd)}")
            
            if not dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)
                _log_result(condition.name, cmd, result)
            else:
                print("(Dry run - not executing)")
                
        elif condition.model_type == ModelType.PI0:
            cmd = create_pi0_command(condition)
            print(f"Command: {' '.join(cmd)}")
            
            if not dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)
                _log_result(condition.name, cmd, result)
            else:
                print("(Dry run - not executing)")
                
        elif condition.model_type == ModelType.UNIVLA:
            commands = create_univla_command(condition, use_sbatch)
            
            # 데이터 변환 단계
            convert_cmd = commands["convert"]
            print(f"Convert Command: {' '.join(convert_cmd)}")
            
            if not dry_run:
                print("Step 1: Converting data...")
                convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
                _log_result(f"{condition.name}_convert", convert_cmd, convert_result)
                
                if convert_result.returncode != 0:
                    print(f"❌ Data conversion failed for: {condition.name}")
                    return
                
                print("✅ Data conversion completed")
            else:
                print("(Dry run - not executing convert)")
            
            # 훈련 단계
            if use_sbatch:
                # SBATCH 스크립트 생성 및 실행
                sbatch_script = commands["sbatch_script"]
                script_filename = f"train_{condition.name}.sh"
                
                print(f"Creating SBATCH script: {script_filename}")
                print("SBATCH script content:")
                print(sbatch_script)
                
                if not dry_run:
                    print("Step 2: Creating and submitting SBATCH job...")
                    
                    # tmp 디렉토리 생성
                    os.makedirs("tmp", exist_ok=True)
                    
                    # 스크립트 파일 작성
                    with open(script_filename, 'w') as f:
                        f.write(sbatch_script)
                    
                    # 스크립트 실행 권한 부여
                    os.chmod(script_filename, 0o755)
                    
                    # sbatch로 작업 제출
                    sbatch_cmd = ["sbatch", script_filename]
                    print(f"Submitting job: {' '.join(sbatch_cmd)}")
                    
                    train_result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
                    _log_result(f"{condition.name}_train", sbatch_cmd, train_result)
                    
                    if train_result.returncode == 0:
                        print(f"✅ Successfully submitted job: {condition.name}")
                        print(f"Job output: {train_result.stdout.strip()}")
                    else:
                        print(f"❌ Job submission failed: {condition.name}")
                        print(f"Error: {train_result.stderr}")
                else:
                    print("(Dry run - not executing sbatch)")
            else:
                # 직접 실행 (워커노드에서 실행용)
                train_cmd = commands["train"]
                print(f"Train Command: {' '.join(train_cmd)}")
                print("💡 Note: Make sure you are running on a worker node with GPU access!")
                print("💡 Use: srun --comment 'univla training' --gpus=1 --nodes=1 --pty /bin/bash")
                
                if not dry_run:
                    print("Step 2: Training model...")
                    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
                    _log_result(f"{condition.name}_train", train_cmd, train_result)
                    
                    if train_result.returncode == 0:
                        print(f"✅ Successfully completed: {condition.name}")
                    else:
                        print(f"❌ Training failed: {condition.name}")
                else:
                    print("(Dry run - not executing train)")
                
        else:
            raise ValueError(f"Unknown model type: {condition.model_type}")
            
    except Exception as e:
        print(f"❌ Error running condition {condition.name}: {str(e)}")


def _log_result(name: str, cmd: list, result):
    """Log subprocess result"""
    if result.returncode == 0:
        print(f"✅ Successfully completed: {name}")
    else:
        print(f"❌ Failed: {name}")
        print(f"Error: {result.stderr}")
        
    # Log the results
    log_file = f"ablation_results_{name}.log"
    with open(log_file, "w") as f:
        f.write(f"Name: {name}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Stdout:\n{result.stdout}\n")
        f.write(f"Stderr:\n{result.stderr}\n")


def run_ablation(model, data_percent, state_type, action_type, camera_type, 
                 output_dir="./outputs", max_steps=10000):
    """기본 ablation 실행 함수"""
    print(f"🚀 Running ablation for {model} model")
    print(f"📊 Configuration: {data_percent}% data, {state_type} state, {action_type} action, {camera_type} camera")
    
    # 조건 이름 생성
    condition_name = f"{model}_{data_percent}_percent_{state_type}_{action_type}_{camera_type}"
    
    # 데이터 변환
    print("📂 Converting data...")
    convert_log_file = f"ablation_results_{condition_name}_convert.log"
    convert_cmd = [
        "python", "univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py",
        "--condition", condition_name,
        "--output-dir", "./converted_data",
        "--data-percent", str(data_percent)
    ]
    
    with open(convert_log_file, 'w') as f:
        print(f"💾 Logging conversion to {convert_log_file}")
        result = subprocess.run(convert_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode != 0:
        print(f"❌ Data conversion failed! Check {convert_log_file}")
        return False
    
    print("✅ Data conversion completed")
    
    # 훈련 실행
    print("🎯 Starting training...")
    train_log_file = f"ablation_results_{condition_name}_train.log"
    train_cmd = [
        "python", "univla/vla_scripts/finetune_rlwrld_ablation.py",
        "--condition", condition_name,
        "--data-root-dir", "./converted_data", 
        "--output-dir", output_dir,
        "--max-steps", str(max_steps),
        "--batch-size", "16",  # A100 최적화된 배치 크기 (2 -> 16)
        "--learning-rate", "1e-4"
    ]
    
    with open(train_log_file, 'w') as f:
        print(f"💾 Logging training to {train_log_file}")
        result = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print(f"❌ Training failed! Check {train_log_file}")
        return False


def run_ablation_rtx4090_optimized(model, data_percent, state_type, action_type, camera_type, 
                                   output_dir="./outputs", max_steps=10000, batch_size=1):
    """RTX 4090 메모리 최적화 버전"""
    print(f"🚀 Running RTX 4090 optimized ablation for {model} model")
    print(f"📊 Configuration: {data_percent}% data, {state_type} state, {action_type} action, {camera_type} camera")
    print(f"💾 Memory optimization: batch_size={batch_size}, gradient_checkpointing=True")
    
    # 조건 이름 생성
    condition_name = f"{model}_{data_percent}_percent_{state_type}_{action_type}_{camera_type}"
    
    # 데이터 변환
    print("📂 Converting data...")
    convert_log_file = f"ablation_results_{condition_name}_convert.log"
    convert_cmd = [
        "python", "univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py",
        "--condition", condition_name,
        "--output-dir", "./converted_data",
        "--data-percent", str(data_percent)
    ]
    
    with open(convert_log_file, 'w') as f:
        print(f"💾 Logging conversion to {convert_log_file}")
        result = subprocess.run(convert_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode != 0:
        print(f"❌ Data conversion failed! Check {convert_log_file}")
        return False
    
    print("✅ Data conversion completed")
    
    # 훈련 실행
    print("🎯 Starting training...")
    train_log_file = f"ablation_results_{condition_name}_train.log"
    train_cmd = [
        "python", "univla/vla_scripts/finetune_rlwrld_ablation.py",
        "--condition", condition_name,
        "--data-root-dir", "./converted_data", 
        "--output-dir", output_dir,
        "--max-steps", str(max_steps),
        "--batch-size", str(batch_size),  # 메모리 최적화
        "--learning-rate", "1e-4"
    ]
    
    with open(train_log_file, 'w') as f:
        print(f"💾 Logging training to {train_log_file}")
        result = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print(f"❌ Training failed! Check {train_log_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run VLA model ablation study')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['univla', 'pi0', 'gr00t'],
                       help='Model to train')
    parser.add_argument('--data', type=int, required=True,
                       help='Percentage of data to use (1-100)')
    parser.add_argument('--state', type=str, required=True,
                       choices=['pos_only', 'full_state'],
                       help='State representation type')
    parser.add_argument('--action', type=str, required=True,
                       choices=['single_arm', 'bimanual'],
                       help='Action space type')
    parser.add_argument('--camera', type=str, required=True,
                       choices=['robot_view', 'third_person', 'multi_view'],
                       help='Camera configuration')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--rtx4090', action='store_true',
                       help='Use RTX 4090 memory optimization (batch_size=1)')
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        if args.rtx4090:
            print("🎯 RTX 4090 메모리 최적화 모드 활성화")
            success = run_ablation_rtx4090_optimized(
                model=args.model,
                data_percent=args.data, 
                state_type=args.state,
                action_type=args.action,
                camera_type=args.camera,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
                batch_size=1  # RTX 4090 최적화
            )
        else:
            success = run_ablation(
                model=args.model,
                data_percent=args.data, 
                state_type=args.state,
                action_type=args.action,
                camera_type=args.camera,
                output_dir=args.output_dir,
                max_steps=args.max_steps
            )
            
        if success:
            print("🎉 Ablation study completed successfully!")
        else:
            print("❌ Ablation study failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
