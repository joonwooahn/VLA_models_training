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


def load_modality_config():
    """Load modality configuration from modality.json"""
    modality_path = Path(__file__).parent / "modality.json"
    with open(modality_path, 'r') as f:
        return json.load(f)


def get_state_indices(modality_config, state_mode, action_mode):
    """Get state indices based on state_mode and action_mode"""
    state_indices = []
    
    # Base position joints
    base_joints = [
        "torso_joints", "head_joints", "right_arm_joints", 
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
    
    # Add velocity joints if pos_vel or pos_vel_torq
    if state_mode in ["pos_vel", "pos_vel_torq"]:
        vel_joints = [
            "torso_joints_vel", "head_joints_vel", "right_arm_joints_vel",
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
    
    # Add torque joints if pos_vel_torq
    if state_mode == "pos_vel_torq":
        torque_joints = [
            "torso_joints_torque", "head_joints_torque", "right_arm_joints_torque",
            "left_arm_joints_torque", "right_hand_joints_torque", "left_hand_joints_torque"
        ]
        
        # If action_mode is right_arm, exclude left-related joints
        if action_mode == "right_arm":
            torque_joints = [joint for joint in torque_joints if "left" not in joint]
        
        for joint in torque_joints:
            if joint in modality_config["state"]:
                start = modality_config["state"][joint]["start"]
                end = modality_config["state"][joint]["end"]
                state_indices.extend(range(start, end))
    
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


def run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=False):
    # Extract task_name from data_dir (last part of the path)
    task_name = Path(data_dir).name
    
    # Create checkpoint directory name based on task and ablation conditions
    checkpoint_dir_name = f"{task_name}_{state_mode}_{action_mode}"
    
    """Run the training script with calculated indices"""
    
    # Load modality configuration
    modality_config = load_modality_config()
    
    # Calculate indices
    state_indices = get_state_indices(modality_config, state_mode, action_mode)
    action_indices = get_action_indices(modality_config, action_mode)
    
    # Convert to comma-separated strings
    state_indices_str = ",".join(map(str, state_indices))
    action_indices_str = ",".join(map(str, action_indices))
    
    print(f"=== Ablation Study Configuration ===")
    print(f"VLA Model: {vla_model}")
    print(f"Data Directory: {data_dir}")
    print(f"Task Name: {task_name}")
    print(f"State Mode: {state_mode}")
    print(f"Action Mode: {action_mode}")
    print(f"Checkpoint Directory: {checkpoint_dir_name}")
    print(f"State Indices: {state_indices_str}")
    print(f"Action Indices: {action_indices_str}")
    print(f"State Dimension: {len(state_indices)}")
    print(f"Action Dimension: {len(action_indices)}")
    print("=" * 40)
    
    # Determine script path based on vla_model
    if vla_model == "univla":
        script_path = Path(__file__).parent / "univla" / "vla_scripts" / "run_conversion_and_training_david.sh"
    else:
        raise ValueError(f"Unsupported VLA model: {vla_model}")
    
    # Check if script exists
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # Make script executable
    script_path.chmod(0o755)
    
    # Build command
    if use_sbatch:
        # Use sbatch with the existing script (which already has SLURM headers)
        cmd = [
            "sbatch",
            "--job-name", f"{task_name}_{state_mode}_{action_mode}",
            "--comment", f"ablation study: {state_mode} + {action_mode}",
            str(script_path),
            f"{task_name}_{state_mode}_{action_mode}",  # job-name for the script
            data_dir,
            "",  # OUTPUT_DATA (use default)
            "",  # CONDA_ENV (use default)
            state_indices_str,
            action_indices_str,
            checkpoint_dir_name  # checkpoint directory name
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"SLURM job submission failed: {e}")
            return None
    else:
        # Direct execution
        cmd = [
            str(script_path),
            f"{task_name}_{state_mode}_{action_mode}",  # job-name for the script
            data_dir,
            "",  # OUTPUT_DATA (use default)
            "",  # CONDA_ENV (use default)
            state_indices_str,
            action_indices_str,
            checkpoint_dir_name  # checkpoint directory name
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code {e.returncode}")
            return False

def run_all_combinations(vla_model, data_dir, use_sbatch=False):
    """Run all combinations of state_mode and action_mode"""
    state_modes = ["pos_only", "pos_vel", "pos_vel_torq"]
    action_modes = ["right_arm", "dual_arm"]
    
    total_combinations = len(state_modes) * len(action_modes)
    current_combination = 0
    
    print(f"Running {total_combinations} combinations ({'SLURM' if use_sbatch else 'Sequential'} mode)")
    
    results = []
    job_ids = []
    
    for state_mode in state_modes:
        for action_mode in action_modes:
            current_combination += 1
            print(f"[{current_combination}/{total_combinations}] {state_mode} + {action_mode}: ", end="")
            
            try:
                if use_sbatch:
                    success = run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=True)
                    if success:
                        job_ids.append({'state_mode': state_mode, 'action_mode': action_mode})
                        print("✅ submitted")
                    else:
                        print("❌ failed")
                else:
                    success = run_training(vla_model, data_dir, state_mode, action_mode, use_sbatch=False)
                    results.append({'state_mode': state_mode, 'action_mode': action_mode, 'success': success})
                    print("✅ success" if success else "❌ failed")
                        
            except Exception as e:
                print(f"❌ error: {e}")
                if not use_sbatch:
                    results.append({'state_mode': state_mode, 'action_mode': action_mode, 'success': False, 'error': str(e)})
    
    print(f"\nSummary: {len(job_ids if use_sbatch else results)} jobs {'submitted' if use_sbatch else 'completed'}")
    if use_sbatch:
        print("Monitor with: squeue -u $USER")
    return len(job_ids if use_sbatch else results) > 0





def main():
    parser = argparse.ArgumentParser(description="Run ablation study for VLA models")
    parser.add_argument("--vla_model", type=str, required=True,
                       default="univla", choices=["gr00t", "pi0", "pi0fast", "univla"],  # Add more models as needed
                       help="VLA model to use for training")
    parser.add_argument("--data_dir", type=str, required=False,
                    #    default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq",
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/250716/allex_gesture_easy_all",
                       help="Path to input data directory")
    parser.add_argument("--state_mode", type=str, required=False,
                       choices=["pos_only", "pos_vel", "pos_vel_torq"],
                       help="State mode: pos_only, pos_vel, or pos_vel_torq")
    parser.add_argument("--action_mode", type=str, required=False,
                       choices=["right_arm", "dual_arm"],
                       help="Action mode: right_arm or dual_arm")
    parser.add_argument("--run_mode_all", action="store_true",
                       help="Run all combinations of state_mode and action_mode")
    parser.add_argument("--sbatch", action="store_true",
                       help="Use SLURM sbatch for parallel execution")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Check if run_mode_all is enabled
    if args.run_mode_all:
        success = run_all_combinations(args.vla_model, args.data_dir, use_sbatch=args.sbatch)
        if not success:
            print("Some combinations failed. Check the summary above.")
            sys.exit(1)
    else:
        # Single mode - validate required arguments
        if args.state_mode is None or args.action_mode is None:
            print("Error: --state_mode and --action_mode are required when not using --run_mode_all")
            print("Use --run_mode_all to run all combinations automatically")
            sys.exit(1)
        
        # Run single training
        if args.sbatch:
            success = run_training(args.vla_model, args.data_dir, args.state_mode, args.action_mode, use_sbatch=True)
            if success:
                print("SLURM job submitted successfully!")
            else:
                print("SLURM job submission failed.")
                sys.exit(1)
        else:
            success = run_training(args.vla_model, args.data_dir, args.state_mode, args.action_mode, use_sbatch=False)
            if not success:
                sys.exit(1)


if __name__ == "__main__":
    main() 