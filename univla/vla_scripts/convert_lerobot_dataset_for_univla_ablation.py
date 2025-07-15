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
import random
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import ablation config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ablation_config import (
    AblationCondition, StateType, ActionType, CameraType,
    get_condition_by_name
)


def convert_lerobot_to_univla_format(input_dir: str, output_dir: str, condition: AblationCondition):
    """Convert LeRobot format dataset to UniVLA format"""
    
    input_path = Path(input_dir)
    base_output_path = Path(output_dir)
    
    # Create condition-specific output directory
    output_path = base_output_path / condition.name
    
    # Check if conversion already exists and is complete
    if output_path.exists():
        # Check if conversion appears complete by looking for episodes
        existing_episodes = list(output_path.glob("episode_*"))
        if existing_episodes:
            print(f"🔍 Found existing converted data at: {output_path}")
            print(f"📂 {len(existing_episodes)} episodes already converted")
            print(f"✅ Skipping conversion (data already exists)")
            return
    
    print(f"🔄 Converting LeRobot dataset for UniVLA training...")
    print(f"   - Input: {input_path}")
    print(f"   - Output: {output_path}")
    print(f"   - Model: {condition.model_type.value}")
    print(f"   - Data: {condition.data_amount}%")
    print(f"   - State: {condition.state_type.value}")
    print(f"   - Action: {condition.action_type.value}")
    print(f"   - Camera: {condition.camera_type.value}")
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    # Get episode parquet files from data/chunk-000 directory
    data_dir = input_path / "data" / "chunk-000"
    episode_files = sorted([f for f in data_dir.glob("episode_*.parquet")])
    
    if condition.data_amount < 100:
        max_episodes = int(len(episode_files) * condition.data_amount / 100)
        
        # Set random seed for reproducible sampling
        random.seed(42)
        np.random.seed(42)
        
        # Random sampling instead of sequential sampling
        episode_files = random.sample(episode_files, max_episodes)
        episode_files = sorted(episode_files)  # Sort selected episodes for consistent naming
        print(f"📊 Using {max_episodes} episodes ({condition.data_amount}% of data - randomly sampled)")
    else:
        print(f"📊 Using all {len(episode_files)} episodes (100% of data)")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert each episode
    for episode_file in tqdm(episode_files, desc="Converting episodes"):
        episode_num = episode_file.stem.replace('episode_', '')
        output_episode_dir = output_path / f"episode_{episode_num}"
        convert_episode(episode_file, output_episode_dir, condition)
    
    print(f"✅ Conversion complete: {len(episode_files)} episodes converted")


def convert_episode(episode_file: Path, output_episode_dir: Path, condition: AblationCondition):
    """Convert a single episode from LeRobot to UniVLA format"""
    
    output_episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LeRobot episode data
    print(f"Loading episode: {episode_file.name}")
    
    # Load episode data from parquet
    df = pd.read_parquet(episode_file)
    
    # Get episode number for video file
    episode_num = episode_file.stem  # e.g., "episode_000000"
    
    # Get video file path based on camera type
    video_base_dir = episode_file.parent.parent.parent / "videos" / "chunk-000"
    if condition.camera_type == CameraType.ROBOT_VIEW:
        video_file = video_base_dir / "observation.images.robot0_robotview" / f"{episode_num}.mp4"
    else:  # For other camera types, use robot view as default
        video_file = video_base_dir / "observation.images.robot0_robotview" / f"{episode_num}.mp4"
    
    if not video_file.exists():
        print(f"⚠️  Video file not found: {video_file}")
        return
    
    # Get indices based on state type and action type
    # Based on actual Allex robot structure from info.json:
    # Position (0-59): torso(0-3) + head(4-5) + right_arm(6-12) + left_arm(13-19) + right_hand(20-39) + left_hand(40-59)
    # Velocity (60-119): same structure with offset +60
    # Torque (120-179): same structure with offset +120
    
    if condition.state_type == StateType.POSITION_ONLY:
        # Position only: use indices 0-59
        if condition.action_type == ActionType.RIGHT_ARM:
            # torso(0-3) + head(4-5) + right_arm(6-12) + right_hand(20-39) = 33개
            state_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            # right end-effector(0-5) + right fingers(12-26) = 21개
            action_indices = list(range(0, 6)) + list(range(12, 27))
        else:  # DUAL_ARM
            # All position indices: torso + head + both arms + both hands = 60개
            state_indices = list(range(0, 60))
            # All action indices: both arms + both hands = 42개
            action_indices = list(range(42))
    elif condition.state_type == StateType.POSITION_VELOCITY:
        # Position + Velocity: use indices 0-119
        if condition.action_type == ActionType.RIGHT_ARM:
            # Position: torso(0-3) + head(4-5) + right_arm(6-12) + right_hand(20-39)
            pos_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            # Velocity: same structure with offset +60
            vel_indices = [i + 60 for i in pos_indices]
            state_indices = pos_indices + vel_indices
            # right end-effector(0-5) + right fingers(12-26) = 21개
            action_indices = list(range(0, 6)) + list(range(12, 27))
        else:  # DUAL_ARM
            # All position + velocity indices = 120개
            state_indices = list(range(0, 120))
            # All action indices = 42개
            action_indices = list(range(42))
    else:  # POSITION_VELOCITY_TORQUE
        # Position + Velocity + Torque: use indices 0-179
        if condition.action_type == ActionType.RIGHT_ARM:
            # Position: torso(0-3) + head(4-5) + right_arm(6-12) + right_hand(20-39)
            pos_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            # Velocity: same structure with offset +60
            vel_indices = [i + 60 for i in pos_indices]
            # Torque: same structure with offset +120
            torque_indices = [i + 120 for i in pos_indices]
            state_indices = pos_indices + vel_indices + torque_indices
            # right end-effector(0-5) + right fingers(12-26) = 21개
            action_indices = list(range(0, 6)) + list(range(12, 27))
        else:  # DUAL_ARM
            # All position + velocity + torque indices = 180개
            state_indices = list(range(0, 180))
            # All action indices = 42개
            action_indices = list(range(42))
    
    state_dim = len(state_indices)
    action_dim = len(action_indices)
    
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    # Extract and process data
    timesteps = len(df)
    
    # Load video and extract frames
    cap = cv2.VideoCapture(str(video_file))
    
    # Initialize arrays
    states = np.zeros((timesteps, state_dim))
    actions = np.zeros((timesteps, action_dim))
    images = []
    
    # Extract frames from video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != timesteps:
        print(f"⚠️  Frame count mismatch: video has {frame_count} frames, data has {timesteps} timesteps")
    
    # Process each timestep
    for i in range(timesteps):
        row = df.iloc[i]
        
        # Extract state from observation
        obs_state = np.array(row['observation.state'])
        states[i] = obs_state[state_indices]
        
        # Extract action
        action = np.array(row['action'])
        actions[i] = action[action_indices]
        
        # Extract frame from video
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB (OpenCV uses BGR, but we want RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)
        else:
            # If no more frames, use the last available frame
            if images:
                images.append(images[-1])
            else:
                # Create a dummy frame if no frames available
                images.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    
    # Save converted data
    np.save(output_episode_dir / "state.npy", states)
    np.save(output_episode_dir / "action.npy", actions)
    
    # Save images as PNG files
    image_dir = output_episode_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    for i, image in enumerate(images):
        cv2.imwrite(str(image_dir / f"{i:06d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print(f"  ✓ Converted episode {episode_file.name}: {timesteps} timesteps")


def create_dummy_data(output_path: Path, condition: AblationCondition):
    """Create dummy data for testing purposes"""
    
    print("🎭 Creating dummy data for testing...")
    
    # Get dimensions based on condition
    state_config = condition.get_state_config()
    
    if condition.state_type == StateType.POSITION_ONLY:
        # Position only: use indices 0-59 (currently available)
        if condition.action_type == ActionType.RIGHT_ARM:
            state_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            action_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        else:
            state_indices = list(range(0, 60))  # All position indices
            action_indices = list(range(42))
    elif condition.state_type == StateType.POSITION_VELOCITY:
        # Position + Velocity: use indices 0-119 (when available)
        if condition.action_type == ActionType.RIGHT_ARM:
            pos_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            vel_indices = list(range(60, 66)) + list(range(66, 73)) + list(range(80, 100))
            state_indices = pos_indices + vel_indices
            action_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        else:
            state_indices = list(range(0, 60)) + list(range(60, 120))  # 120개
            action_indices = list(range(42))
    else:  # POSITION_VELOCITY_TORQUE
        # Position + Velocity + Torque: use indices 0-179 (when available)
        if condition.action_type == ActionType.RIGHT_ARM:
            pos_indices = list(range(0, 6)) + list(range(6, 13)) + list(range(20, 40))
            vel_indices = list(range(60, 66)) + list(range(66, 73)) + list(range(80, 100))
            torque_indices = list(range(120, 126)) + list(range(126, 133)) + list(range(140, 160))
            state_indices = pos_indices + vel_indices + torque_indices
            action_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        else:
            state_indices = list(range(0, 180))  # 180개
            action_indices = list(range(42))
    
    state_dim = len(state_indices)
    action_dim = len(action_indices)
    
    # Create multiple dummy episodes based on data amount
    base_episodes = 20  # Base number for 100%
    num_episodes = max(1, int(base_episodes * condition.data_amount / 100))
    
    for episode_idx in range(num_episodes):
        episode_dir = output_path / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy data for this episode
        timesteps = 50  # Fixed number of timesteps per episode
        
        # Random states and actions
        np.random.seed(42 + episode_idx)  # Different seed per episode
        states = np.random.randn(timesteps, state_dim) * 0.1
        actions = np.random.randn(timesteps, action_dim) * 0.1
        
        # Save data
        np.save(episode_dir / "state.npy", states)
        np.save(episode_dir / "action.npy", actions)
        
        # Create dummy images
        image_dir = episode_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for i in range(timesteps):
            cv2.imwrite(str(image_dir / f"{i:06d}.png"), dummy_image)
    
    print(f"🎭 Created {num_episodes} dummy episodes with {timesteps} timesteps each")


def generate_instructions(condition: AblationCondition) -> List[str]:
    """Generate instruction prompts based on condition"""
    
    base_instructions = [
        "Use the robot to pick up the object",
        "Move the object to the target location", 
        "Manipulate the target object carefully",
        "Execute the manipulation task precisely",
        "Complete the robotic manipulation sequence"
    ]
    
    # Add dual-arm specific instructions
    if condition.action_type == ActionType.DUAL_ARM:
        dual_arm_instructions = [
            "Coordinate both arms to manipulate the object",
            "Use both left and right arms for the task",
            "dual-arm manipulation of the target object"
        ]
        base_instructions.extend(dual_arm_instructions)
    
    return base_instructions


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset for UniVLA ablation study")
    parser.add_argument("--condition", type=str, required=True, 
                       help="Ablation condition name")
    parser.add_argument("--input-dir", type=str, 
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq",
                       help="Input directory path")
    parser.add_argument("--output-dir", type=str,
                       default="./univla/converted_data",
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
    print(f"   - Data: {condition.data_amount}%")
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
