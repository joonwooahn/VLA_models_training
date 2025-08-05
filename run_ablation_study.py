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
SUPPORTED_VLA_MODELS = ["gr00t", "pi0", "pi0fast", "univla", "diffusion", "act"]
VIDEO_SUPPORTED_MODELS = ["gr00t", "pi0", "pi0fast", "diffusion", "act"]  # Models that support video modes
SINGLE_GPU_MODELS = ["gr00t", "pi0", "pi0fast", "diffusion", "act"]  # Models that use single GPU
MULTI_GPU_MODELS = ["univla"]  # Models that use multiple GPUs

STATE_MODES = ["pos_only", "pos_vel"]   # ["pos_only", "pos_vel", "pos_vel_torq"]
ACTION_MODES = ["right_arm", "dual_arm"]
ACTION_MODES_FRANKA = ["right_arm"]  # franka only supports right_arm
VIDEO_MODES = ["robotview", "multiview"]

def detect_robot_type(data_dir):
    """Detect robot type from data directory path"""
    if "franka" in str(data_dir).lower():
        return "franka"
    else:
        return "allex"

def get_action_modes_for_robot(robot_type):
    """Get supported action modes for the robot type"""
    if robot_type == "franka":
        return ACTION_MODES_FRANKA
    else:
        return ACTION_MODES

def is_video_supported_model(vla_model):
    """Check if the model supports video modes"""
    return vla_model in VIDEO_SUPPORTED_MODELS

def is_single_gpu_model(vla_model):
    """Check if the model uses single GPU"""
    return vla_model in SINGLE_GPU_MODELS

def get_gpu_count(vla_model):
    """Get GPU count for the model"""
    return 1 if is_single_gpu_model(vla_model) else 2

def calculate_indices(modality_config, state_mode, action_mode, robot_type="allex"):
    """Calculate state and action indices and convert to strings"""
    state_indices = get_state_indices(modality_config, state_mode, action_mode, robot_type)
    action_indices = get_action_indices(modality_config, action_mode, robot_type)
    
    state_indices_str = ",".join(map(str, state_indices))
    action_indices_str = ",".join(map(str, action_indices))
    
    return state_indices, action_indices, state_indices_str, action_indices_str

def load_modality_config(robot_type="allex"):
    """Load modality configuration from modality.json or modality_franka.json"""
    if robot_type == "franka":
        modality_path = Path(__file__).parent / "modality_franka.json"
    else:
        modality_path = Path(__file__).parent / "modality.json"
    
    with open(modality_path, 'r') as f:
        return json.load(f)

def get_state_indices(modality_config, state_mode, action_mode, robot_type="allex"):
    """Get state indices based on state_mode and action_mode, excluding head_joints for training."""
    state_indices = []
    
    if robot_type == "franka":
        # Franka robot joints
        base_joints = ["arm_joints", "hand_joints"]
        
        for joint in base_joints:
            if joint in modality_config["state"]:
                start = modality_config["state"][joint]["start"]
                end = modality_config["state"][joint]["end"]
                state_indices.extend(range(start, end))
        
        # Add velocity joints if pos_vel
        if state_mode in ["pos_vel"]:
            vel_joints = ["arm_joint_velocities"]
            for joint in vel_joints:
                if joint in modality_config["state"]:
                    start = modality_config["state"][joint]["start"]
                    end = modality_config["state"][joint]["end"]
                    state_indices.extend(range(start, end))
    else:
        # Allex robot joints (original logic)
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

def get_action_indices(modality_config, action_mode, robot_type="allex"):
    """Get action indices based on action_mode"""
    action_indices = []
    
    if robot_type == "franka":
        # Franka only supports right_arm mode (single arm robot)
        franka_joints = ["arm_eef_pos", "finger_joints"]
        for joint in franka_joints:
            if joint in modality_config["action"]:
                start = modality_config["action"][joint]["start"]
                end = modality_config["action"][joint]["end"]
                action_indices.extend(range(start, end))
    else:
        # Allex robot joints (original logic)
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

    # Detect robot type from data_dir
    robot_type = detect_robot_type(data_dir)

    # Extract task_name from data_dir (last part of the path)
    task_name = Path(data_dir).name
    
    # Create checkpoint name based on task and ablation conditions
    if is_video_supported_model(vla_model):
        checkpoint_name = f"{task_name}/{state_mode}_{action_mode}_{video_mode}"
    else:
        # univla는 video_mode가 없으므로 robotview로 고정
        checkpoint_name = f"{task_name}/{state_mode}_{action_mode}_robotview"
    
    """Run the training script with calculated indices"""
    
    # Load modality configuration
    modality_config = load_modality_config(robot_type)
    
    # Calculate indices using helper function
    state_indices, action_indices, state_indices_str, action_indices_str = calculate_indices(modality_config, state_mode, action_mode, robot_type)
    
    print(f"=== Ablation Study Configuration ===")
    print(f"Robot Type: {robot_type}")
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
        "diffusion": Path(__file__).parent / "pi0" / "slurm_pi0_train_ablation_study.sh",
        "act": Path(__file__).parent / "pi0" / "slurm_pi0_train_ablation_study.sh",
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
    if is_video_supported_model(vla_model):
        job_name = f"{vla_model}_{state_mode}_{action_mode}_{video_mode}"
    else:
        # univla는 video_mode가 없으므로 robotview로 고정
        job_name = f"{vla_model}_{state_mode}_{action_mode}_robotview"
    
    # 모델별 추가 인수 전달 (공통)
    if is_video_supported_model(vla_model):
        # video_mode를 지원하는 모델들
        additional_args = [state_mode, action_mode, video_mode, vla_model, robot_type]
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
            "--comment", f"ablation study: {state_mode} + {action_mode} + {video_mode if is_video_supported_model(vla_model) else 'robotview'}",
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

def check_pi0_conversion_needed(data_dir):
    """Check if pi0 dataset conversion is needed for any combination"""
    # Detect robot type from data_dir
    robot_type = detect_robot_type(data_dir)
    action_modes = get_action_modes_for_robot(robot_type)
    
    # Extract task name from data_dir
    task_name = Path(data_dir).name
    
    # Check if any combination needs conversion
    for state_mode in STATE_MODES:
        for action_mode in action_modes:
            for video_mode in VIDEO_MODES:
                # Check if converted dataset exists for this combination
                converted_dir = Path(__file__).parent / "pi0" / "converted_datasets" / f"{task_name}_{state_mode}_{action_mode}_{video_mode}"
                if not converted_dir.exists():
                    return True  # At least one combination needs conversion
                
                # Check if the converted dataset has required files
                info_file = converted_dir / "info.json"
                if not info_file.exists():
                    return True  # Conversion incomplete
    
    return False  # All combinations are already converted

def run_pi0_dataset_conversion(data_dir):
    """Run pi0/diffusion/act dataset conversion for all combinations of state_mode, action_mode, and video_mode"""
    
    # Detect robot type from data_dir
    robot_type = detect_robot_type(data_dir)
    action_modes = get_action_modes_for_robot(robot_type)
    
    total_combinations = len(STATE_MODES) * len(action_modes) * len(VIDEO_MODES)
    current_combination = 0
    
    print(f"Converting pi0/diffusion/act datasets for {robot_type} robot with {total_combinations} combinations")
    
    # Extract task name from data_dir
    task_name = Path(data_dir).name
    
    # Load modality configuration once
    modality_config = load_modality_config(robot_type)
    
    results = []
    
    for state_mode in STATE_MODES:
        for action_mode in action_modes:
            for video_mode in VIDEO_MODES:
                current_combination += 1
                print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode} + {video_mode}: ", end="")
                
                try:
                    # Calculate indices using helper function
                    state_indices, action_indices, state_indices_str, action_indices_str = calculate_indices(modality_config, state_mode, action_mode, robot_type)
                    
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

def run_all_combinations(vla_model, data_dir):
    """Run all combinations of state_mode and action_mode"""
    
    # Detect robot type from data_dir
    robot_type = detect_robot_type(data_dir)
    action_modes = get_action_modes_for_robot(robot_type)
    
    # Load modality configuration once for indices calculation
    modality_config = load_modality_config(robot_type)
    
    # Print indices summary for all combinations
    print(f"\n{'='*60}")
    print(f"INDICES SUMMARY for {vla_model.upper()} ({robot_type.upper()} ROBOT)")
    print(f"{'='*60}")
    
    for state_mode in STATE_MODES:
        for action_mode in action_modes:
            state_indices, action_indices, state_indices_str, action_indices_str = calculate_indices(modality_config, state_mode, action_mode, robot_type)
            print(f"{state_mode}_{action_mode}:")
            print(f"  State indices: {state_indices_str}")
            print(f"  Action indices: {action_indices_str}")
            print(f"  State dim: {len(state_indices)}, Action dim: {len(action_indices)}")
    
    print(f"{'='*60}\n")
    # exit(1)
    
    # video_modes는 pi0, pi0fast, gr00t, diffusion, act에만 적용
    if is_video_supported_model(vla_model):
        video_modes = VIDEO_MODES
        total_combinations = len(STATE_MODES) * len(action_modes) * len(video_modes)
        print(f"Running {total_combinations} combinations (SLURM mode) with video modes for {robot_type} robot")
    else:
        video_modes = ["robotview"]  # univla는 robotview만 사용
        total_combinations = len(STATE_MODES) * len(action_modes)
        print(f"Running {total_combinations} combinations (SLURM mode) for {robot_type} robot")
    
    current_combination = 0
    results = []
    job_ids = []
    
    # Generate unique timestamps for each combination to avoid conflicts
    import time
    base_timestamp = int(time.time())
    
    # Special handling for univla to avoid dataset conversion race conditions
    if vla_model == "univla":
        print("🔄 Special handling for univla: Checking if dataset conversion is needed...")
        
        # Check if dataset is already converted
        task_name = Path(data_dir).name
        converted_data_dir = Path(__file__).parent / "univla" / "vla_scripts" / "converted_data_for_univla" / task_name
        
        # Count original episodes from LeRobot dataset structure
        original_episodes = 0
        if Path(data_dir).exists():
            try:
                # LeRobot datasets have structure: data/chunk-000/episode_*.parquet
                data_chunk_dir = Path(data_dir) / "data" / "chunk-000"
                if data_chunk_dir.exists():
                    for item in data_chunk_dir.iterdir():
                        if item.is_file() and item.name.startswith('episode_') and item.name.endswith('.parquet'):
                            original_episodes += 1
                else:
                    # Fallback: check if it's already a converted format with episode directories
                    for item in Path(data_dir).iterdir():
                        if item.is_dir() and item.name.startswith('episode_'):
                            original_episodes += 1
                print(f"📊 Original dataset has {original_episodes} episodes")
            except Exception as e:
                print(f"⚠️ Could not count original episodes: {e}")
                original_episodes = 0
        
        # Check if conversion is already complete
        conversion_complete = False
        if converted_data_dir.exists() and original_episodes > 0:
            converted_episodes = 0
            for item in converted_data_dir.iterdir():
                if (item.is_dir() and 
                    item.name.startswith('episode_') and
                    (item / "state.npy").exists() and 
                    (item / "action.npy").exists()):
                    converted_episodes += 1
            
            if converted_episodes >= original_episodes:
                print(f"✅ Dataset already converted! {converted_episodes}/{original_episodes} episodes found.")
                conversion_complete = True
            else:
                print(f"📊 Partial conversion found: {converted_episodes}/{original_episodes} episodes")
        
        # If conversion is not complete, run one job to do the conversion
        if not conversion_complete:
            print("🔄 Running dataset conversion...")
            first_state_mode = STATE_MODES[0]
            first_action_mode = action_modes[0]
            first_video_mode = video_modes[0]
            
            print(f"[Conversion] {first_state_mode} + {first_action_mode} (with dataset conversion): ", end="")
            
            try:
                # Run first combination with dataset conversion
                success = run_training(vla_model, data_dir, first_state_mode, first_action_mode, use_sbatch=True, video_mode=first_video_mode)
                if success:
                    job_ids.append({'state_mode': first_state_mode, 'action_mode': first_action_mode})
                    print("✅ submitted")
                    
                    # Wait for dataset conversion to complete
                    print("⏳ Waiting for dataset conversion to complete...")
                    conversion_timeout = 1800  # 30 minutes
                    check_interval = 30
                    elapsed_time = 0
                    
                    while elapsed_time < conversion_timeout:
                        if converted_data_dir.exists():
                            converted_episodes = 0
                            for item in converted_data_dir.iterdir():
                                if (item.is_dir() and 
                                    item.name.startswith('episode_') and
                                    (item / "state.npy").exists() and 
                                    (item / "action.npy").exists()):
                                    converted_episodes += 1
                            
                            print(f"📊 Converted episodes: {converted_episodes}/{original_episodes}")
                            
                            if original_episodes > 0 and converted_episodes >= original_episodes:
                                print("✅ Dataset conversion completed!")
                                break
                        
                        time.sleep(check_interval)
                        elapsed_time += check_interval
                    
                    if elapsed_time >= conversion_timeout:
                        print("⚠️ Dataset conversion timeout reached. Proceeding anyway...")
                    
                    # Skip this combination in the main loop since we already submitted it
                    current_combination = 1
                else:
                    print("❌ failed")
                    return False
                    
            except Exception as e:
                print(f"❌ error: {e}")
                return False
        else:
            print("✅ Dataset conversion not needed, proceeding with training...")
            current_combination = 0
    
    # Track which combinations have been processed for univla
    processed_combinations = set()
    if vla_model == "univla" and current_combination > 0:
        # Mark the first combination as already processed
        first_state_mode = STATE_MODES[0]
        first_action_mode = action_modes[0]
        processed_combinations.add((first_state_mode, first_action_mode))
    
    # Run remaining combinations (or all combinations for non-univla models)
    for state_mode in STATE_MODES:
        for action_mode in action_modes:
            for video_mode in video_modes:
                # Skip combinations that have already been processed for univla
                if vla_model == "univla" and (state_mode, action_mode) in processed_combinations:
                    current_combination += 1
                    continue
                    
                current_combination += 1
                if is_video_supported_model(vla_model):
                    print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode} + {video_mode}: ", end="")
                else:
                    print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode}: ", end="")
            
                # Create unique timestamp for each combination
                unique_timestamp = base_timestamp + current_combination
                
                try:
                    # Always use SLURM mode
                    success = run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=True, video_mode=video_mode)
                    if success:
                        if is_video_supported_model(vla_model):
                            job_ids.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode})
                        else:
                            job_ids.append({'state_mode': state_mode, 'action_mode': action_mode})
                        print("✅ submitted")
                    else:
                        print("❌ failed")
                            
                except Exception as e:
                    print(f"❌ error: {e}")
                    if not use_sbatch:
                        if is_video_supported_model(vla_model):
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'video_mode': video_mode, 'success': False, 'error': str(e)})
                        else:
                            results.append({'state_mode': state_mode, 'action_mode': action_mode, 'success': False, 'error': str(e)})
    
    print(f"\nSummary: {len(job_ids)} jobs submitted")
    import os
    user = os.getenv('USER', 'david')
    print(f"Monitor with: squeue -u {user}")
    return len(job_ids) > 0

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for VLA models")
    parser.add_argument("--vla_model", type=str, required=False,
                       choices=SUPPORTED_VLA_MODELS,
                       help="VLA model to use for training (pi0/diffusion/act will convert data first, then train)")
    parser.add_argument("--data_dir", type=str, required=False,
                    ### allex robot
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_all",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_lift_cylinder",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cylinder",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cylinder_reduced_hand",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cylinder_reduced_hand_10",
                    ##### new dataset
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/gesture",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cube",
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/lift_cylinder",
                    ### franka dataset
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/franka_lift_cylinder",
                    ###
                       help="Path to input data directory")
    parser.add_argument("--state_mode", type=str, required=False,
                    #    choices=["pos_only", "pos_vel", "pos_vel_torq"],
                       choices=STATE_MODES,
                       help="State mode: pos_only or pos_vel")
                    #    help="State mode: pos_only, pos_vel or pos_vel_torq")
    parser.add_argument("--action_mode", type=str, required=False,
                       choices=ACTION_MODES,
                       help="Action mode: right_arm or dual_arm")
    parser.add_argument("--video_mode", type=str, required=False, default="multiview",
                       choices=VIDEO_MODES,
                       help="Video mode: robotview or multiview (only for gr00t/pi0/pi0fast/diffusion/act)")
    parser.add_argument("--run_mode_all", action="store_true",
                       help="Run all combinations of state_mode and action_mode")
    # Removed --sbatch argument - now always uses SLURM
    parser.add_argument("--pi0_dataset_convert", action="store_true",
                       help="Convert dataset for pi0/diffusion/act models for all state_mode and action_mode combinations")

    
    args = parser.parse_args()
    
    # Validate data directory (only if not using --run_mode_all without specific model)
    if args.data_dir is None:
        if args.run_mode_all and args.vla_model is None:
            print("Error: --data_dir is required when using --run_mode_all")
            sys.exit(1)
        elif not args.pi0_dataset_convert:
            print("Error: --data_dir is required")
            sys.exit(1)
    elif not Path(args.data_dir).exists():
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
            # Check if pi0 dataset conversion is needed (only when running all models)
            if check_pi0_conversion_needed(args.data_dir):
                print("🔄 Step 1: Running pi0 dataset conversion for all combinations...")
                conversion_success = run_pi0_dataset_conversion(args.data_dir)
                if not conversion_success:
                    print("Some dataset conversions failed. Check the summary above.")
                    sys.exit(1)
                print("✅ Dataset conversion completed successfully!\n")
            else:
                print("✅ Pi0 dataset conversion not needed - all combinations already converted!\n")
            
            # Then, run training for all models
            print("🔄 Step 2: Starting training for all models...")
        
        if args.vla_model is None:
            # Run all models if no specific model is specified
            all_models = SUPPORTED_VLA_MODELS
            print(f"Running training for all models: {all_models}")
            
            overall_success = True
            for model in all_models:
                print(f"\n{'='*60}")
                print(f"Running ablation study for {model.upper()}")
                print(f"{'='*60}")
                
                success = run_all_combinations(model, args.data_dir)
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
            print(f"🔄 Running training for specific model: {args.vla_model}")
            
            # Only run pi0 dataset conversion if the model needs it
            if args.vla_model in ["pi0", "pi0fast", "diffusion", "act"]:
                if check_pi0_conversion_needed(args.data_dir):
                    print("🔄 Running pi0 dataset conversion for pi0-family models...")
                    conversion_success = run_pi0_dataset_conversion(args.data_dir)
                    if not conversion_success:
                        print("Some dataset conversions failed. Check the summary above.")
                        sys.exit(1)
                    print("✅ Dataset conversion completed successfully!\n")
                else:
                    print("✅ Pi0 dataset conversion not needed - all combinations already converted!\n")
            
            success = run_all_combinations(args.vla_model, args.data_dir)
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
        
        # Validate action_mode for franka robot
        robot_type = detect_robot_type(args.data_dir)
        supported_action_modes = get_action_modes_for_robot(robot_type)
        if args.action_mode not in supported_action_modes:
            print(f"Error: --action_mode '{args.action_mode}' is not supported for {robot_type} robot")
            print(f"Supported action modes for {robot_type}: {supported_action_modes}")
            sys.exit(1)
        
        # Validate video_mode for pi0/pi0fast/diffusion/act
        if args.vla_model in VIDEO_SUPPORTED_MODELS and args.video_mode is None:
            print("Error: --video_mode is required for pi0/pi0fast/diffusion/act models")
            sys.exit(1)
        
        # Run single training (always use SLURM)
        success = run_training(args.vla_model, args.data_dir, args.state_mode, args.action_mode, use_sbatch=True, video_mode=args.video_mode)
        if success:
            print("SLURM job submitted successfully!")
        else:
            print("SLURM job submission failed.")
            sys.exit(1)

if __name__ == "__main__":
    main() 



### 체크포인트 삭제 note
# cd /virtual_lab/rlwrld/david/VLA_models_training/_checkpoints
# find gr00t/franka_lift_cylinder* -type d \( -name "checkpoint-2000" -o -name "checkpoint-4000" \) -exec rm -rf {} +
# find pi0/allex_gesture_easy_all* -type d \( -name "005000" -o -name "010000" -o -name "015000" -o -name "020000" -o -name "025000" \) -exec rm -rf {} +
# find diffusion/allex_gesture_easy_all* -type d \( -name "005000" -o -name "010000" -o -name "015000" -o -name "020000" -o -name "025000" \) -exec rm -rf {} +
# find act/allex_gesture_easy_all* -type d \( -name "005000" -o -name "010000" -o -name "015000" -o -name "020000" -o -name "025000" \) -exec rm -rf {} +
# find univla/allex_gesture_easy_all* -type d \( -name "10000" -o -name "20000" \) -exec rm -rf {} +
# find pi0/franka_lift_cylinder* -type d \( -name "020000" \) -exec rm -rf {} +
# find act/franka_lift_cylinder* -type d \( -name "020000" -o -name "040000" \) -exec rm -rf {} +