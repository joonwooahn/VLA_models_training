#!/usr/bin/env python
"""
Unified Ablation Study Runner for VLA Models
Complete factorial design: 3 models × 2 data × 2 state × 2 action × 2 camera = 48 experiments
"""

import os
import sys
import subprocess
import argparse
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
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python /virtual_lab/rlwrld/david/UniVLA/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py --condition {condition.name} --input-dir /virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_cube --output-dir /virtual_lab/rlwrld/david/UniVLA/converted_data"
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
python /virtual_lab/rlwrld/david/UniVLA/vla_scripts/finetune_rlwrld_ablation.py \\
    --condition {condition.name} \\
    --data-root-dir /virtual_lab/rlwrld/david/UniVLA/converted_data \\
    --output-dir /virtual_lab/rlwrld/david/UniVLA/outputs/{condition.get_output_dir()} \\
    --max-steps {condition.max_steps} \\
    --batch-size {condition.batch_size} \\
    --learning-rate {condition.learning_rate}
"""
        return {"convert": convert_cmd, "sbatch_script": sbatch_script}
    else:
        # 직접 실행 명령 (워커노드에서 실행용)
        train_cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python /virtual_lab/rlwrld/david/UniVLA/vla_scripts/finetune_rlwrld_ablation.py --condition {condition.name} --data-root-dir /virtual_lab/rlwrld/david/UniVLA/converted_data --output-dir /virtual_lab/rlwrld/david/UniVLA/outputs/{condition.get_output_dir()} --max-steps {condition.max_steps} --batch-size {condition.batch_size} --learning-rate {condition.learning_rate}"
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




def main():
    parser = argparse.ArgumentParser(description="Run VLA ablation study")
    
    # 기존 방식 (condition name으로 지정)
    parser.add_argument("--condition", type=str, help="Specific condition to run")
    
    # 새로운 방식 (개별 파라미터로 지정)
    parser.add_argument("--model", type=str, choices=["gr00t", "pi0", "univla"], 
                       help="Model type")
    parser.add_argument("--data", type=str, choices=["20", "100"], 
                       help="Data amount (20 percent or 100 percent)")
    parser.add_argument("--state", type=str, choices=["pos_only", "full_state"], 
                       help="State composition (pos_only or full_state)")
    parser.add_argument("--action", type=str, choices=["single_arm", "bimanual"], 
                       help="Action type (single_arm or bimanual)")
    parser.add_argument("--camera", type=str, choices=["robot_view", "multi_view"], 
                       help="Camera configuration (robot_view or multi_view)")
    
    # 유틸리티 옵션들
    parser.add_argument("--list", action="store_true", help="List all available conditions")
    parser.add_argument("--summary", action="store_true", help="Show conditions summary")
    parser.add_argument("--all", action="store_true", help="Run all conditions")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--use-sbatch", action="store_true", help="Use SBATCH for UniVLA training (default: direct execution)")
    
    args = parser.parse_args()
    
    if args.summary:
        print_conditions_summary()
        return
    
    if args.list:
        print("All available ablation conditions:")
        for i, name in enumerate(list_all_conditions(), 1):
            condition = get_condition_by_name(name)
            if condition:
                data_pct = "20 percent" if condition.data_amount == DataAmount.PERCENT_20 else "100 percent"
                state_desc = "pos_only" if condition.state_type == StateType.POSITION_ONLY else "full_state"
                action_desc = "single_arm" if condition.action_type == ActionType.SINGLE_ARM else "bimanual"
                camera_desc = "robot_view" if condition.camera_type == CameraType.ROBOT_VIEW else "multi_view"
                
                print(f"{i:2d}. {name}")
                print(f"    Model: {condition.model_type.value}, Data: {data_pct}, State: {state_desc}")
                print(f"    Action: {action_desc}, Camera: {camera_desc}")
                print()
        return
    
    # 개별 파라미터로 조건 생성
    if args.model and args.data and args.state and args.action and args.camera:
        # 파라미터를 enum으로 변환
        model_type = ModelType(args.model)
        data_amount = DataAmount.PERCENT_20 if args.data == "20" else DataAmount.PERCENT_100
        state_type = StateType.POSITION_ONLY if args.state == "pos_only" else StateType.FULL_STATE
        action_type = ActionType.SINGLE_ARM if args.action == "single_arm" else ActionType.BIMANUAL
        camera_type = CameraType.ROBOT_VIEW if args.camera == "robot_view" else CameraType.MULTI_VIEW
        
        # 조건 생성
        condition = AblationCondition(
            model_type=model_type,
            data_amount=data_amount,
            state_type=state_type,
            action_type=action_type,
            camera_type=camera_type
        )
        
        print(f"Generated condition: {condition.name}")
        print(f"Model: {args.model}, Data: {args.data} percent, State: {args.state}")
        print(f"Action: {args.action}, Camera: {args.camera}")
        
        run_condition(condition, dry_run=args.dry_run, use_sbatch=args.use_sbatch)
        return
    
    # 기존 방식 (condition name으로 지정)
    if args.condition:
        condition = get_condition_by_name(args.condition)
        if not condition:
            print(f"Error: Condition '{args.condition}' not found")
            print(f"Available conditions: {len(list_all_conditions())} total")
            print("Use --list to see all conditions")
            return
        
        run_condition(condition, dry_run=args.dry_run, use_sbatch=args.use_sbatch)
        
    elif args.model and not (args.data and args.state and args.action and args.camera):
        # 특정 모델의 모든 조건 실행
        model_type = ModelType(args.model)
        model_conditions = get_conditions_by_model(model_type)
        
        print(f"Running all {len(model_conditions)} conditions for {args.model.upper()}...")
        
        for i, condition in enumerate(model_conditions, 1):
            print(f"\n[{i}/{len(model_conditions)}] Starting: {condition.name}")
            run_condition(condition, dry_run=args.dry_run, use_sbatch=args.use_sbatch)
            
        print(f"\n✅ Completed all {len(model_conditions)} conditions for {args.model.upper()}!")
        
    elif args.all:
        print(f"Running all {len(ABLATION_CONDITIONS)} conditions...")
        print("This will run:")
        print("- 3 models (gr00t, pi0, univla)")
        print("- 2 data amounts (20 percent, 100 percent)")
        print("- 2 state types (pos_only, full_state)")
        print("- 2 action types (single_arm, bimanual)")
        print("- 2 camera types (robot_view, multi_view)")
        print(f"- Total: 3 × 2 × 2 × 2 × 2 = {len(ABLATION_CONDITIONS)} experiments")
        
        if not args.dry_run:
            confirm = input("\nThis will take a very long time. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return
        
        for i, condition in enumerate(ABLATION_CONDITIONS, 1):
            print(f"\n[{i}/{len(ABLATION_CONDITIONS)}] Starting: {condition.name}")
            run_condition(condition, dry_run=args.dry_run, use_sbatch=args.use_sbatch)
            
        print(f"\n✅ Completed all {len(ABLATION_CONDITIONS)} conditions!")
        
    else:
        parser.print_help()
        print(f"\nTotal available conditions: {len(ABLATION_CONDITIONS)}")
        print("Use --summary to see organized breakdown")
        print("\nExamples:")
        print("  # Run specific condition by parameters")
        print("  python run_ablation.py --model univla --data 100 --state pos_only --action single_arm --camera robot_view --dry-run")
        print("  # Run specific condition by name")
        print("  python run_ablation.py --condition univla_100_percent_pos_only_single_arm_robot_view --dry-run")
        print("  # Run all conditions for a model")
        print("  python run_ablation.py --model univla --dry-run")
        print("  # List all conditions")
        print("  python run_ablation.py --list")


if __name__ == "__main__":
    main()
