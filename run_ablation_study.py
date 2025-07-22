#!/usr/bin/env python3
"""
Ablation Study Runner for VLA Models Training

This script runs ablation studies for different state and action configurations
based on the modality.json file.

Usage:
    python run_ablation_study.py --vla_model univla --data_dir /path/to/data --state_mode pos_only --action_mode right_arm
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Constants to eliminate duplication
SUPPORTED_VLA_MODELS = ["gr00t", "pi0", "pi0fast", "univla"]
VIDEO_SUPPORTED_MODELS = ["pi0", "pi0fast", "gr00t"]  # Models that support video modes
SINGLE_GPU_MODELS = ["gr00t", "pi0", "pi0fast"]  # Models that use single GPU
MULTI_GPU_MODELS = ["univla"]  # Models that use multiple GPUs

STATE_MODES = ["pos_only", "pos_vel"]   # ["pos_only", "pos_vel", "pos_vel_torq"]
ACTION_MODES = ["right_arm", "dual_arm"]
VIDEO_MODES = ["robotview", "multiview"]

def is_video_supported_model(vla_model):
    """Check if the model supports video modes"""
    return vla_model in VIDEO_SUPPORTED_MODELS

def is_single_gpu_model(vla_model):
    """Check if the model uses single GPU"""
    return vla_model in SINGLE_GPU_MODELS

def get_gpu_count(vla_model):
    """Get GPU count for the model"""
    return 1 if is_single_gpu_model(vla_model) else 4

def calculate_indices(modality_config, state_mode, action_mode):
    """Calculate state and action indices and convert to strings"""
    state_indices = get_state_indices(modality_config, state_mode, action_mode)
    action_indices = get_action_indices(modality_config, action_mode)
    
    state_indices_str = ",".join(map(str, state_indices))
    action_indices_str = ",".join(map(str, action_indices))
    
    return state_indices, action_indices, state_indices_str, action_indices_str

def load_modality_config():
    """Load modality configuration from modality.json"""
    modality_path = Path(__file__).parent / "modality.json"
    with open(modality_path, 'r') as f:
        return json.load(f)

def get_state_indices(modality_config, state_mode, action_mode):
    """Get state indices based on state_mode and action_mode, excluding head_joints for training."""
    state_indices = []
    
    # Base position joints (exclude head_joints)
    base_joints = [
        # "torso_joints", "head_joints", "right_arm_joints", 
        "torso_joints", "right_arm_joints", 
        "left_arm_joints", "right_hand_joints", "left_hand_joints"
    ]
    
    # If action_mode is right_arm, exclude left-related joints
    if action_mode == "right_arm":
        base_joints = [joint for joint in base_joints if "left" not in joint]
    
    for joint in base_joints:
        if joint in modality_config["state"]:
            start = modality_config["state"][joint]["start"]
            end = modality_config["state"][joint]["end"]
            state_indices.extend(range(start, end))
    
    # Add velocity joints if pos_vel or pos_vel_torq (exclude head_joints_vel)
    # if state_mode in ["pos_vel", "pos_vel_torq"]:
    if state_mode in ["pos_vel"]:
        vel_joints = [
            # "torso_joints_vel", "head_joints_vel", "right_arm_joints_vel",
            "torso_joints_vel", "right_arm_joints_vel",
            "left_arm_joints_vel", "right_hand_joints_vel", "left_hand_joints_vel"
        ]
        # If action_mode is right_arm, exclude left-related joints
        if action_mode == "right_arm":
            vel_joints = [joint for joint in vel_joints if "left" not in joint]
        for joint in vel_joints:
            if joint in modality_config["state"]:
                start = modality_config["state"][joint]["start"]
                end = modality_config["state"][joint]["end"]
                state_indices.extend(range(start, end))
    
    # # Add torque joints if pos_vel_torq (exclude head_joints_torque)
    # if state_mode == "pos_vel_torq":
    #     torque_joints = [
    #         # "torso_joints_torque", "head_joints_torque", "right_arm_joints_torque",
    #         "torso_joints_torque", "right_arm_joints_torque",
    #         "left_arm_joints_torque", "right_hand_joints_torque", "left_hand_joints_torque"
    #     ]
    #     # If action_mode is right_arm, exclude left-related joints
    #     if action_mode == "right_arm":
    #         torque_joints = [joint for joint in torque_joints if "left" not in joint]
    #     for joint in torque_joints:
    #         if joint in modality_config["state"]:
    #             start = modality_config["state"][joint]["start"]
    #             end = modality_config["state"][joint]["end"]
    #             state_indices.extend(range(start, end))
    
    return sorted(state_indices)

def get_action_indices(modality_config, action_mode):
    """Get action indices based on action_mode"""
    action_indices = []
    
    if action_mode == "right_arm":
        # Only right arm and right finger joints
        right_joints = ["right_arm_eef_pos", "right_finger_joints"]
        for joint in right_joints:
            if joint in modality_config["action"]:
                start = modality_config["action"][joint]["start"]
                end = modality_config["action"][joint]["end"]
                action_indices.extend(range(start, end))
    
    elif action_mode == "dual_arm":
        # Both arms and finger joints
        all_joints = ["right_arm_eef_pos", "left_arm_eef_pos", "right_finger_joints", "left_finger_joints"]
        for joint in all_joints:
            if joint in modality_config["action"]:
                start = modality_config["action"][joint]["start"]
                end = modality_config["action"][joint]["end"]
                action_indices.extend(range(start, end))
    
    return sorted(action_indices)

def run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=False, video_mode=None):
    # GPU 수 자동 결정
    gpus = get_gpu_count(vla_model)
    
    # Extract task_name from data_dir (last part of the path)
    task_name = Path(data_dir).name
    
    # Create checkpoint name based on task and ablation conditions
    checkpoint_name = f"{task_name}/{state_mode}_{action_mode}_{video_mode}"
    
    """Run the training script with calculated indices"""
    
    # Load modality configuration
    modality_config = load_modality_config()
    
    # Calculate indices using helper function
    state_indices, action_indices, state_indices_str, action_indices_str = calculate_indices(modality_config, state_mode, action_mode)
    
    print(f"=== Ablation Study Configuration ===")
    print(f"VLA Model: {vla_model}")
    print(f"Data Directory: {data_dir}")
    print(f"Task Name: {task_name}")
    print(f"State Mode: {state_mode}")
    print(f"Action Mode: {action_mode}")
    print(f"Checkpoint Name: {checkpoint_name}")
    print(f"State Indices: {state_indices_str}")
    print(f"Action Indices: {action_indices_str}")
    print(f"State Dimension: {len(state_indices)}")
    print(f"Action Dimension: {len(action_indices)}")
    print("=" * 40)
    
    # 스크립트 경로 설정 (모델별 규칙)
    script_paths = {
        "gr00t": Path(__file__).parent / "gr00t" / "slurm_ft_ablation_study.sh",
        "pi0": Path(__file__).parent / "pi0" / "slurm_pi0_train_ablation_study.sh",
        "pi0fast": Path(__file__).parent / "pi0" / "slurm_pi0_train_ablation_study.sh",
        "univla": Path(__file__).parent / "univla" / "vla_scripts" / "run_conversion_and_training_david.sh"
    }
    
    if vla_model not in script_paths:
        raise ValueError(f"Unsupported VLA model: {vla_model}")
    
    script_path = script_paths[vla_model]
    
    # Check if script exists
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # Make script executable
    script_path.chmod(0o755)
    
    # job-name에서 모델명 제거하고 task_name 포함
    job_name = f"{state_mode}_{action_mode}_{video_mode}"
    
    # 모델별 추가 인수 전달 (공통)
    if is_video_supported_model(vla_model):
        # video_mode를 지원하는 모델들
        additional_args = [state_mode, action_mode, video_mode, vla_model]
    else:
        # univla는 video_mode를 지원하지 않으므로 추가 인수 없음
        additional_args = []
    
    if use_sbatch:
        # The script already has SLURM headers, so run it directly with sbatch
        # 기본 뼈대: 모든 모델이 공통으로 사용하는 부분
        log_dir = f"_logs/{vla_model}/{task_name}"
        
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "sbatch",
            "--job-name", job_name,
            "--comment", f"ablation study: {state_mode} + {action_mode} + {video_mode}",
            f"--gpus={gpus}",
            "--output", f"{log_dir}/slurm-%j-%x.log",
            "--error", f"{log_dir}/slurm-%j-%x.log",
            str(script_path),
            job_name,  # job-name for the script
            data_dir,
            "",  # OUTPUT_DATA (use default)
            "",  # CONDA_ENV (use default)
            state_indices_str,
            action_indices_str,
            checkpoint_name  # checkpoint name
        ]
        
        # 추가 인수 전달
        cmd.extend(additional_args)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"SLURM job submitted: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"SLURM job submission failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
    else:
        # Direct execution - set working directory to script directory
        script_dir = script_path.parent
        cmd = [
            str(script_path),
            checkpoint_name,  # job-name for the script
            data_dir,
            "",  # OUTPUT_DATA (use default)
            "",  # CONDA_ENV (use default)
            state_indices_str,
            action_indices_str,
            checkpoint_name  # checkpoint name
        ]
        
        # 추가 인수 전달
        cmd.extend(additional_args)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, cwd=script_dir)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code {e.returncode}")
            return False

def run_pi0_dataset_conversion(data_dir):
    """Run pi0 dataset conversion for all combinations of state_mode, action_mode, and video_mode"""
    
    total_combinations = len(STATE_MODES) * len(ACTION_MODES) * len(VIDEO_MODES)
    current_combination = 0
    
    print(f"Converting pi0 datasets for {total_combinations} combinations")
    
    # Extract task name from data_dir
    task_name = Path(data_dir).name
    
    # Load modality configuration once
    modality_config = load_modality_config()
    
    results = []
    
    for state_mode in STATE_MODES:
        for action_mode in ACTION_MODES:
            for video_mode in VIDEO_MODES:
                current_combination += 1
                print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode} + {video_mode}: ", end="")
                
                try:
                    # Calculate indices using helper function
                    state_indices, action_indices, state_indices_str, action_indices_str = calculate_indices(modality_config, state_mode, action_mode)
                    
                    # Run pi0 dataset conversion
                    script_path = Path(__file__).parent / "pi0" / "create_pi0_dataset.py"
                    
                    cmd = [
                        "python", str(script_path),
                        "--source_dir", data_dir,
                        "--task_name", task_name,
                        "--state_mode", state_mode,
                        "--action_mode", action_mode,
                        "--video_mode", video_mode,
                        "--state_indices", state_indices_str,
                        "--action_indices", action_indices_str
                    ]
                    
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': True})
                    print("✅ success")
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ failed: {e}")
                    print(f"Error output: {e.stderr}")
                    results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': False, 'error': str(e)})
                except Exception as e:
                    print(f"❌ error: {e}")
                    results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': False, 'error': str(e)})
    
    successful_conversions = sum(1 for r in results if r['success'])
    print(f"\nSummary: {successful_conversions}/{total_combinations} conversions completed successfully")
    return successful_conversions > 0

def run_all_combinations(vla_model, data_dir, use_sbatch=False):
    """Run all combinations of state_mode and action_mode"""
    
    # video_modes는 pi0, pi0fast, gr00t에만 적용
    if is_video_supported_model(vla_model):
        video_modes = VIDEO_MODES
        total_combinations = len(STATE_MODES) * len(ACTION_MODES) * len(video_modes)
        print(f"Running {total_combinations} combinations ({'SLURM' if use_sbatch else 'Sequential'} mode) with video modes")
    else:
        video_modes = ["robotview"]  # univla는 robotview만 사용
        total_combinations = len(STATE_MODES) * len(ACTION_MODES)
        print(f"Running {total_combinations} combinations ({'SLURM' if use_sbatch else 'Sequential'} mode)")
    
    current_combination = 0
    results = []
    job_ids = []
    
    # Generate unique timestamps for each combination to avoid conflicts
    import time
    base_timestamp = int(time.time())
    
    for state_mode in STATE_MODES:
        for action_mode in ACTION_MODES:
            for video_mode in video_modes:
                current_combination += 1
                if is_video_supported_model(vla_model):
                    print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode} + {video_mode}: ", end="")
                else:
                    print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode}: ", end="")
                
                # Create unique timestamp for each combination
                unique_timestamp = base_timestamp + current_combination
                
                try:
                    if use_sbatch:
                        success = run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=True, video_mode=video_mode)
                        if success:
                            if is_video_supported_model(vla_model):
                                job_ids.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode})
                            else:
                                job_ids.append({'state_mode': state_mode, 'action_mode': action_mode})
                            print("✅ submitted")
                        else:
                            print("❌ failed")
                    else:
                        success = run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=False, video_mode=video_mode)
                        if is_video_supported_model(vla_model):
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': success})
                        else:
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'success': success})
                        print("✅ success" if success else "❌ failed")
                        
                except Exception as e:
                    print(f"❌ error: {e}")
                    if not use_sbatch:
                        if is_video_supported_model(vla_model):
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': False, 'error': str(e)})
                        else:
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'success': False, 'error': str(e)})
    
    print(f"\nSummary: {len(job_ids if use_sbatch else results)} jobs {'submitted' if use_sbatch else 'completed'}")
    if use_sbatch:
        print("Monitor with: squeue -u $USER")
    return len(job_ids if use_sbatch else results) > 0

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for VLA models")
    parser.add_argument("--vla_model", type=str, required=False,
                       default="univla", choices=SUPPORTED_VLA_MODELS,
                       help="VLA model to use for training (pi0 will convert data first, then train)")
    parser.add_argument("--data_dir", type=str, required=False,
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/250718/allex_gesture_easy_all",
                       help="Path to input data directory")
    parser.add_argument("--state_mode", type=str, required=False,
                    #    choices=["pos_only", "pos_vel", "pos_vel_torq"],
                    #    help="State mode: pos_only, pos_vel, or pos_vel_torq")
                       choices=STATE_MODES,
                       help="State mode: pos_only, or pos_vel")
    parser.add_argument("--action_mode", type=str, required=False,
                       choices=ACTION_MODES,
                       help="Action mode: right_arm or dual_arm")
    parser.add_argument("--video_mode", type=str, required=False, default="multiview",
                       choices=VIDEO_MODES,
                       help="Video mode: robotview or multiview (only for pi0/pi0fast)")
    parser.add_argument("--run_mode_all", action="store_true",
                       help="Run all combinations of state_mode and action_mode")
    parser.add_argument("--sbatch", action="store_true",
                       help="Use SLURM sbatch for parallel execution")
    parser.add_argument("--pi0_dataset_convert", action="store_true",
                       help="Convert dataset for pi0 model for all state_mode and action_mode combinations")

    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Check if pi0_dataset_convert is enabled
    if args.pi0_dataset_convert:
        success = run_pi0_dataset_conversion(args.data_dir)
        if not success:
            print("Some dataset conversions failed. Check the summary above.")
            sys.exit(1)
        return
    
    # Check if run_mode_all is enabled
    if args.run_mode_all:
        if args.vla_model is None:
            # Run all models if no specific model is specified
            all_models = SUPPORTED_VLA_MODELS
            print(f"Running --run_mode_all for all models: {all_models}")
            
            overall_success = True
            for model in all_models:
                print(f"\n{'='*60}")
                print(f"Running ablation study for {model.upper()}")
                print(f"{'='*60}")
                
                success = run_all_combinations(model, args.data_dir, use_sbatch=args.sbatch)
                if not success:
                    print(f"Some combinations failed for {model}. Check the summary above.")
                    overall_success = False
                
                # Add a small delay between models to avoid overwhelming the system
                import time
                time.sleep(2)
            
            if not overall_success:
                print("\nSome models failed. Check the summaries above.")
                sys.exit(1)
            else:
                print(f"\n✅ All models completed successfully!")
        else:
            # Run for specific model
            success = run_all_combinations(args.vla_model, args.data_dir, use_sbatch=args.sbatch)
            if not success:
                print("Some combinations failed. Check the summary above.")
                sys.exit(1)
    else:
        # Single mode - validate required arguments
        if args.vla_model is None:
            print("Error: --vla_model is required when not using --pi0_dataset_convert")
            sys.exit(1)
        if args.state_mode is None or args.action_mode is None:
            print("Error: --state_mode and --action_mode are required when not using --run_mode_all")
            print("Use --run_mode_all to run all combinations automatically")
            sys.exit(1)
        
        # Validate video_mode for pi0/pi0fast
        if args.vla_model in VIDEO_SUPPORTED_MODELS and args.video_mode is None:
            print("Error: --video_mode is required for pi0/pi0fast models")
            sys.exit(1)
        
        # Run single training
        if args.sbatch:
            success = run_training(args.vla_model, args.data_dir, args.state_mode, args.action_mode, use_sbatch=True, video_mode=args.video_mode)
            if success:
                print("SLURM job submitted successfully!")
            else:
                print("SLURM job submission failed.")
                sys.exit(1)
        else:
            success = run_training(args.vla_model, args.data_dir, args.state_mode, args.action_mode, use_sbatch=False, video_mode=args.video_mode)
            if not success:
                sys.exit(1)

if __name__ == "__main__":
    main() 

### 최초로 task 마다 한번 수행 해야함. 
# python run_ablation_study.py --pi0_dataset_convert
# python run_ablation_study.py --vla_model univla --state_mode pos_vel_torq --action_mode dual_arm 
