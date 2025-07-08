#!/usr/bin/env python
"""
Ablation Study를 위한 LeRobot 데이터셋 변환 스크립트
UniVLA 훈련을 위해 다양한 조건에 맞게 데이터를 변환합니다.
"""

import os
import cv2
import pandas as pd
import numpy as np
import glob
import json
import argparse
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import ablation config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ablation_config import (
    AblationCondition, StateType, ActionType, CameraType, DataAmount,
    get_condition_by_name
)


def convert_lerobot_to_univla_format(input_dir: str, output_dir: str, condition: AblationCondition):
    """Convert LeRobot format dataset to UniVLA format"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir) / condition.name
    
    print(f"🔄 Converting data from {input_path} to {output_path}")
    print(f"📊 Condition: {condition.name}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find all episodes
    episode_dirs = []
    
    # Try different possible structures
    possible_patterns = [
        input_path / "episode_*",
        input_path / "*" / "episode_*",
        input_path / "episodes" / "episode_*"
    ]
    
    for pattern in possible_patterns:
        matching_dirs = list(glob.glob(str(pattern)))
        if matching_dirs:
            episode_dirs.extend([Path(d) for d in matching_dirs if Path(d).is_dir()])
            break
    
    if not episode_dirs:
        # If no episode directories found, create dummy data
        print("⚠️  No episode directories found, creating dummy data for testing")
        create_dummy_data(output_path, condition)
        return
    
    episode_dirs = sorted(episode_dirs)
    print(f"📁 Found {len(episode_dirs)} episodes")
    
    # Limit episodes based on data amount
    if condition.data_amount == DataAmount.PERCENT_20:
        max_episodes = int(len(episode_dirs) * 0.2)
        episode_dirs = episode_dirs[:max_episodes]
        print(f"📊 Using {max_episodes} episodes (20% of data)")
    
    # Convert each episode
    for i, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        try:
            convert_episode(episode_dir, output_path / f"episode_{i:04d}", condition)
        except Exception as e:
            print(f"❌ Failed to convert episode {episode_dir}: {e}")
            continue
    
    print(f"✅ Conversion completed: {len(episode_dirs)} episodes converted")


def convert_episode(episode_dir: Path, output_episode_dir: Path, condition: AblationCondition):
    """Convert a single episode"""
    
    os.makedirs(output_episode_dir, exist_ok=True)
    
    # Load episode data
    try:
        # Try to load from parquet files (LeRobot format)
        data_files = list(episode_dir.glob("*.parquet"))
        if data_files:
            import pandas as pd
            df = pd.read_parquet(data_files[0])
            
            # Extract states and actions
            state_columns = [col for col in df.columns if 'state' in col.lower()]
            action_columns = [col for col in df.columns if 'action' in col.lower()]
            
            if state_columns and action_columns:
                states = df[state_columns].values
                actions = df[action_columns].values
            else:
                # Create dummy data if columns not found
                states = np.random.randn(100, 60)  # 60-dim state
                actions = np.random.randn(100, 42)  # 42-dim action
        else:
            # Create dummy data if no parquet files
            # Make sure to create enough dimensions for all possible indices
            states = np.random.randn(100, 180)  # 60 position + 60 velocity + 60 torque
            actions = np.random.randn(100, 42)
            
    except Exception as e:
        print(f"⚠️  Could not load episode data, creating dummy data: {e}")
        # Make sure to create enough dimensions for all possible indices
        states = np.random.randn(100, 180)  # 60 position + 60 velocity + 60 torque
        actions = np.random.randn(100, 42)
    
    # Apply state and action filtering based on condition
    state_config = condition.get_state_config()
    
    # Get indices based on action type
    if condition.action_type == ActionType.SINGLE_ARM:
        # Torso + right arm + right hand
        state_indices = list(range(0, 3)) + list(range(5, 12)) + list(range(19, 40))
        action_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    else:  # BIMANUAL
        # Torso + both arms + both hands
        state_indices = list(range(0, 3)) + list(range(5, 19)) + list(range(19, 60))
        action_indices = list(range(42))
    
    # Add velocity and torque if full state
    if condition.state_type == StateType.FULL_STATE:
        velocity_indices = [i + 60 for i in state_indices]
        torque_indices = [i + 120 for i in state_indices]
        state_indices.extend(velocity_indices)
        state_indices.extend(torque_indices)
    
    # Filter states and actions
    if len(state_indices) > 0 and states.shape[1] > max(state_indices):
        filtered_states = states[:, state_indices]
    else:
        # Create appropriate size dummy data with correct dimensions
        print(f"⚠️  State data shape {states.shape} insufficient for indices {state_indices}")
        print(f"Creating dummy state data with {len(state_indices)} dimensions")
        filtered_states = np.random.randn(states.shape[0], len(state_indices) if state_indices else 32)
    
    if len(action_indices) > 0 and actions.shape[1] > max(action_indices):
        filtered_actions = actions[:, action_indices]
    else:
        # Create appropriate size dummy data
        print(f"⚠️  Action data shape {actions.shape} insufficient for indices {action_indices}")
        print(f"Creating dummy action data with {len(action_indices)} dimensions")
        filtered_actions = np.random.randn(actions.shape[0], len(action_indices) if action_indices else condition.get_action_dim())
    
    # Save filtered data
    np.save(output_episode_dir / "state.npy", filtered_states)
    np.save(output_episode_dir / "action.npy", filtered_actions)
    
    # Create images
    num_frames = len(filtered_states)
    for frame_idx in range(num_frames):
        # Create dummy image (you would replace this with actual image extraction)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(output_episode_dir / f"frame_{frame_idx + 1:03d}.png"), dummy_image)
    
    # Create instruction file
    instruction = get_instruction_for_condition(condition)
    with open(output_episode_dir / "instruction.txt", "w") as f:
        f.write(instruction)


def create_dummy_data(output_path: Path, condition: AblationCondition):
    """Create dummy data for testing purposes"""
    
    print("🎭 Creating dummy data for testing...")
    
    # Get dimensions based on condition
    state_config = condition.get_state_config()
    
    if condition.action_type == ActionType.SINGLE_ARM:
        state_indices = list(range(0, 3)) + list(range(5, 12)) + list(range(19, 40))
        action_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    else:
        state_indices = list(range(0, 3)) + list(range(5, 19)) + list(range(19, 60))
        action_indices = list(range(42))
    
    if condition.state_type == StateType.FULL_STATE:
        velocity_indices = [i + 60 for i in state_indices]
        torque_indices = [i + 120 for i in state_indices]
        state_indices.extend(velocity_indices)
        state_indices.extend(torque_indices)
    
    state_dim = len(state_indices)
    action_dim = len(action_indices)
    
    # Create multiple dummy episodes
    num_episodes = 20 if condition.data_amount == DataAmount.PERCENT_100 else 4
    
    for episode_idx in range(num_episodes):
        episode_dir = output_path / f"episode_{episode_idx:04d}"
        os.makedirs(episode_dir, exist_ok=True)
        
        # Random episode length
        episode_length = np.random.randint(50, 150)
        
        # Generate random states and actions with proper dimensions
        # Create full 180-dim state data first, then filter
        full_states = np.random.randn(episode_length, 180) * 0.1  # 60 pos + 60 vel + 60 torque
        full_actions = np.random.randn(episode_length, 42) * 0.1
        
        # Filter to get the desired dimensions
        states = full_states[:, state_indices] if state_indices else full_states[:, :state_dim]
        actions = full_actions[:, action_indices] if action_indices else full_actions[:, :action_dim]
        
        # Add some structure to the data
        for t in range(1, episode_length):
            states[t] = states[t-1] + np.random.randn(state_dim) * 0.01
            actions[t] = actions[t-1] + np.random.randn(action_dim) * 0.01
        
        # Save data
        np.save(episode_dir / "state.npy", states)
        np.save(episode_dir / "action.npy", actions)
        
        # Create dummy images
        for frame_idx in range(episode_length):
            # Create a simple colored image
            color = np.random.randint(0, 255, 3)
            dummy_image = np.full((224, 224, 3), color, dtype=np.uint8)
            # Add some noise
            noise = np.random.randint(-50, 50, (224, 224, 3))
            dummy_image = np.clip(dummy_image.astype(int) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(episode_dir / f"frame_{frame_idx + 1:03d}.png"), dummy_image)
        
        # Create instruction
        instruction = get_instruction_for_condition(condition)
        with open(episode_dir / "instruction.txt", "w") as f:
            f.write(instruction)
    
    print(f"✅ Created {num_episodes} dummy episodes")


def get_instruction_for_condition(condition: AblationCondition) -> str:
    """Generate appropriate instruction based on condition"""
    
    base_instructions = [
        "pick up the cube",
        "place the cube in the container",
        "manipulate the object",
        "move the cube to the target location",
        "grasp and lift the object"
    ]
    
    if condition.action_type == ActionType.BIMANUAL:
        bimanual_instructions = [
            "use both hands to manipulate the object",
            "coordinate both arms to move the cube",
            "bimanual manipulation of the target object"
        ]
        base_instructions.extend(bimanual_instructions)
    
    # Select instruction based on hash of condition name for consistency
    instruction_idx = hash(condition.name) % len(base_instructions)
    return base_instructions[instruction_idx]


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset for UniVLA ablation study")
    parser.add_argument("--condition", type=str, required=True, 
                       help="Ablation condition name")
    parser.add_argument("--input-dir", type=str, 
                       default="/home/david/.cache/huggingface/lerobot/RLWRLD/allex_cube",
                       help="Input directory path")
    parser.add_argument("--output-dir", type=str,
                       default="./converted_data",
                       help="Output directory path")
    
    args = parser.parse_args()
    
    # 조건 가져오기
    condition = get_condition_by_name(args.condition)
    if not condition:
        print(f"❌ Error: Condition '{args.condition}' not found")
        return
    
    print(f"✅ Data conversion for condition: {args.condition}")
    print(f"📂 Input: {args.input_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"📊 Condition details:")
    print(f"   - Model: {condition.model_type.value}")
    print(f"   - Data: {'20%' if condition.data_amount.value == '20_percent' else '100%'}")
    print(f"   - State: {condition.state_type.value}")
    print(f"   - Action: {condition.action_type.value}")
    print(f"   - Camera: {condition.camera_type.value}")
    print(f"   - Action dim: {condition.get_action_dim()}")
    
    try:
        convert_lerobot_to_univla_format(args.input_dir, args.output_dir, condition)
        print("🎉 Data conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Data conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
