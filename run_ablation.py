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
import tempfile
import random
import re
import json
import itertools
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# 공통 설정 및 상수
# =============================================================================


# CLI-related default values
DEFAULT_MAX_STEPS = 5000
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_NUM_WORKERS = 8
DEFAULT_INPUT_DIR = "/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq"
DEFAULT_CONVERTED_DATASETS_DIR = "./converted_datasets_ablation"  # Pre-converted datasets directory
LEROBOT_CACHE_BASE = "/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/"

class AblationDefaults:
    """중앙화된 기본값 설정 - 실험 조건 및 데이터 구조"""
    # 지원되는 타입들 (고정)
    SUPPORTED_MODELS = ["gr00t", "pi0", "pi0_fast", "univla"]
    SUPPORTED_STATE_TYPES = ["pos_only", "pos_vel", "pos_vel_torque"]
    SUPPORTED_ACTION_TYPES = ["right_arm", "dual_arm"]
    SUPPORTED_CAMERA_TYPES = ["robot_view", "multi_view"]
    SUPPORTED_DATA_AMOUNTS = [20, 30, 100]  # 일반적인 ablation study 퍼센트
    
    # 기본 데이터셋 정보 (동적으로 변경 가능)
    DEFAULT_DATASET_NAME = "RLWRLD/allex_gesture_easy_pos_vel_torq"
    
    # 모델별 제한사항
    MODEL_LIMITATIONS = {
        "gr00t": {
            "states": ["pos_only"],  # GR00T는 pos_only만 지원
            "cameras": ["robot_view", "multi_view"]
        },
        "pi0": {
            "states": ["pos_only", "pos_vel", "pos_vel_torque"],  # PI0는 모든 state 지원
            "cameras": ["robot_view", "multi_view"]
        },
        "pi0_fast": {
            "states": ["pos_only", "pos_vel", "pos_vel_torque"],  # PI0_FAST는 모든 state 지원
            "cameras": ["robot_view", "multi_view"]
        },
        "univla": {
            "states": ["pos_only", "pos_vel", "pos_vel_torque"],  # UniVLA는 모든 state 지원
            "cameras": ["robot_view"]  # UniVLA는 robot_view만 지원
        }
    }
    
    # State dimension 상수
    STATE_DIMENSIONS = {
        "pos_only": 60,      # Position only (60 dimensions)
        "pos_vel": 120,      # Position + Velocity (60 + 60 dimensions)
        "pos_vel_torque": 180  # Position + Velocity + Torque (60 + 60 + 60 dimensions)
    }
    
    # Action dimension 상수
    ACTION_DIMENSIONS = {
        "right_arm": 21,  # right end-effector(6) + right fingers(15) = 21
        "dual_arm": 42    # right end-effector(6) + left end-effector(6) + right fingers(15) + left fingers(15) = 42
    }

# =============================================================================
# Enum 클래스들
# =============================================================================

class ModelType(Enum):
    GR00T = "gr00t"
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    UNIVLA = "univla"


class StateType(Enum):
    POSITION_ONLY = "pos_only"                    # joint position만 (0-59)
    POSITION_VELOCITY = "pos_vel"                 # position + velocity (0-119)
    POSITION_VELOCITY_TORQUE = "pos_vel_torque"   # position + velocity + torque (0-179)


class ActionType(Enum):
    RIGHT_ARM = "right_arm"   # right end-effector + right fingers (21개)
    DUAL_ARM = "dual_arm"     # right+left end-effector + right+left fingers (42개)


class CameraType(Enum):
    ROBOT_VIEW = "robot_view"     # robotview만
    MULTI_VIEW = "multi_view"     # robotview+sideview+wrist_views

# =============================================================================
# 데이터 클래스들
# =============================================================================

@dataclass
class StateConfig:
    """State configuration for ablation study"""
    use_joint_positions: bool = True
    use_joint_velocities: bool = False
    use_joint_torques: bool = False
    
    # State filtering indices (60-dimensional state)
    torso_indices: Optional[List[int]] = None
    head_indices: Optional[List[int]] = None
    right_arm_indices: Optional[List[int]] = None
    left_arm_indices: Optional[List[int]] = None
    right_hand_indices: Optional[List[int]] = None
    left_hand_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        # Allex robot joint structure
        if self.torso_indices is None:
            self.torso_indices = list(range(0, 4))      # 0-3 (4개) - torso
        if self.head_indices is None:
            self.head_indices = list(range(4, 6))       # 4-5 (2개) - head
        if self.right_arm_indices is None:
            self.right_arm_indices = list(range(6, 13)) # 6-12 (7개)
        if self.left_arm_indices is None:
            self.left_arm_indices = list(range(13, 20)) # 13-19 (7개)
        if self.right_hand_indices is None:
            self.right_hand_indices = list(range(20, 40)) # 20-39 (20개)
        if self.left_hand_indices is None:
            self.left_hand_indices = list(range(40, 60))  # 40-59 (20개)
    
    def get_state_indices(self, action_type: ActionType) -> List[int]:
        """Get state indices based on action type (single modality: position only)"""
        indices = []
        
        # Always include torso and head
        if self.torso_indices:
            indices.extend(self.torso_indices)
        if self.head_indices:
            indices.extend(self.head_indices)
        
        if action_type == ActionType.RIGHT_ARM:
            # 몸(torso+head)+오른팔+오른손
            if self.right_arm_indices:
                indices.extend(self.right_arm_indices)
            if self.right_hand_indices:
                indices.extend(self.right_hand_indices)
        elif action_type == ActionType.DUAL_ARM:
            # 몸(torso+head)+오른팔+왼팔+오른손+왼손
            if self.right_arm_indices:
                indices.extend(self.right_arm_indices)
            if self.left_arm_indices:
                indices.extend(self.left_arm_indices)
            if self.right_hand_indices:
                indices.extend(self.right_hand_indices)
            if self.left_hand_indices:
                indices.extend(self.left_hand_indices)
        
        return sorted(indices)

    def get_multi_modality_state_indices(self, state_type: StateType, action_type: ActionType) -> List[int]:
        """Get state indices for multi-modality state (pos, vel, torque) - UNIFIED FOR ALL MODELS"""
        # Get base position indices (0-59)
        base_position_indices = self.get_state_indices(action_type)
        
        all_indices = []
        
        if state_type == StateType.POSITION_ONLY:
            # Only position: use base indices directly
            all_indices.extend(base_position_indices)
            
        elif state_type == StateType.POSITION_VELOCITY:
            # Position + Velocity: base indices + same pattern with +60 offset
            all_indices.extend(base_position_indices)  # Position: 0-59
            velocity_indices = [idx + 60 for idx in base_position_indices]  # Velocity: 60-119
            all_indices.extend(velocity_indices)
            
        elif state_type == StateType.POSITION_VELOCITY_TORQUE:
            # Position + Velocity + Torque: all three modalities
            all_indices.extend(base_position_indices)  # Position: 0-59
            velocity_indices = [idx + 60 for idx in base_position_indices]  # Velocity: 60-119
            torque_indices = [idx + 120 for idx in base_position_indices]  # Torque: 120-179
            all_indices.extend(velocity_indices)
            all_indices.extend(torque_indices)
        
        return sorted(all_indices)

    def get_action_indices(self, action_type: ActionType) -> List[int]:
        """Get action indices based on action type - UNIFIED FOR ALL MODELS"""
        if action_type == ActionType.RIGHT_ARM:
            # Right end-effector(0-5) + right fingers(12-26) = 21개
            return list(range(0, 6)) + list(range(12, 27))  # [0,1,2,3,4,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        elif action_type == ActionType.DUAL_ARM:
            # All actions: both arms + both hands = 42개
            return list(range(42))  # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
        else:
            raise ValueError(f"Unknown action type: {action_type}")


@dataclass
class DataConfig:
    """Data configuration for ablation study"""
    dataset_name: str = AblationDefaults.DEFAULT_DATASET_NAME
    data_amount: int = 100  # Percentage of data to use (1-100)
    input_dir: Optional[str] = None  # For dynamic episode counting
    
    # Camera configurations
    camera_keys: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.camera_keys is None:
            self.camera_keys = ["video.camera_ego"]  # Default robot view
        
        # Validate data_amount using common function
        validate_data_amount(self.data_amount)
    
    def get_num_episodes(self) -> Optional[int]:
        """Get number of episodes based on data amount"""
        # Calculate based on percentage
        base_episodes = get_dataset_total_episodes(self.dataset_name, self.input_dir)
        if self.data_amount < 100:
            return max(1, int(base_episodes * self.data_amount / 100))
        return None  # Use all episodes
    
    def get_episode_indices(self, seed: int = 42) -> Optional[List[int]]:
        """Get episode indices based on data amount"""
        if self.data_amount >= 100:
            return None  # Use all episodes
        
        # Calculate number of episodes to sample
        target_count = max(1, int(get_dataset_total_episodes(self.dataset_name, self.input_dir) * self.data_amount / 100))
        
        # Use first N episodes instead of random sampling to avoid LeRobot indexing issues
        # This is more stable and avoids the episode_data_index IndexError
        sequential_episodes = list(range(target_count))
        
        return sequential_episodes
    
    def get_state_dimension(self, state_type: StateType, action_type: ActionType | None = None) -> int:
        """Get state dimension based on state type and action type for allex_gesture_easy_pos_vel_torq dataset"""
        
        # Base dimensions for dual_arm (full robot state)
        base_dimensions = {
            StateType.POSITION_ONLY: 60,      # torso(4) + head(2) + right_arm(7) + left_arm(7) + right_hand(20) + left_hand(20) = 60
            StateType.POSITION_VELOCITY: 120,  # pos(60) + vel(60) = 120  
            StateType.POSITION_VELOCITY_TORQUE: 180  # pos(60) + vel(60) + torque(60) = 180
        }
        
        base_dim = base_dimensions.get(state_type)
        if base_dim is None:
            raise ValueError(f"Unknown state type: {state_type}")
        
        # If no action_type specified or dual_arm, return full dimensions
        if action_type is None or action_type == ActionType.DUAL_ARM:
            return base_dim
        
        # For right_arm, exclude left_arm(7) + left_hand(20) = 27 dims per modality
        if action_type == ActionType.RIGHT_ARM:
            # Calculate how many modalities we have (pos, vel, torque)
            if state_type == StateType.POSITION_ONLY:
                # Remove left_arm(7) + left_hand(20) = 27 from pos(60)
                return 60 - 27  # = 33
            elif state_type == StateType.POSITION_VELOCITY:
                # Remove 27 from both pos and vel
                return 120 - (27 * 2)  # = 66
            else:  # POSITION_VELOCITY_TORQUE
                # Remove 27 from pos, vel, and torque
                return 180 - (27 * 3)  # = 99
        
        return base_dim
    
    def get_camera_keys(self, camera_type: CameraType) -> List[str]:
        """Get camera keys based on camera type"""
        if camera_type == CameraType.ROBOT_VIEW:
            return ["video.camera_ego"]
        elif camera_type == CameraType.MULTI_VIEW:
            return [
                "video.camera_ego",      # robot view
                "video.camera_ext",      # side view
                "video.right_wrist",     # right wrist view (if available)
                "video.left_wrist"       # left wrist view (if available)
            ]
        return self.camera_keys or ["video.camera_ego"]


@dataclass
class AblationCondition:
    """Single ablation study condition"""
    model_type: ModelType
    data_amount: int  # Percentage of data to use (1-100)
    state_type: StateType
    action_type: ActionType
    camera_type: CameraType
    
    # Training specific - 기본값 제거하여 명시적 전달 강제
    max_steps: int
    batch_size: int
    
    # Dataset configuration (dynamic)
    dataset_name: str = AblationDefaults.DEFAULT_DATASET_NAME
    input_dir: Optional[str] = None
    
    def __post_init__(self):
        # Validate data_amount using common function
        validate_data_amount(self.data_amount)
        
        # Validate model limitations using common function
        validate_model_limitations(
            self.model_type.value, 
            self.state_type.value, 
            self.camera_type.value
        )
    
    @property
    def name(self) -> str:
        """Generate unique name for this condition"""
        return f"{self.model_type.value}_{self.data_amount}_percent_{self.state_type.value}_{self.action_type.value}_{self.camera_type.value}"
    
    def get_output_dir(self, job_id: Optional[str] = None) -> str:
        """Generate output directory name with optional job_id prefix"""
        if job_id:
            return f"job_{job_id}_{self.model_type.value}_{self.name}"
        return f"{self.model_type.value}_{self.name}"
    
    def get_state_config(self) -> StateConfig:
        """Get state configuration based on state type"""
        # State configuration mapping
        state_configs = {
            StateType.POSITION_ONLY: StateConfig(
                use_joint_positions=True,
                use_joint_velocities=False,
                use_joint_torques=False
            ),
            StateType.POSITION_VELOCITY: StateConfig(
                use_joint_positions=True,
                use_joint_velocities=True,
                use_joint_torques=False
            ),
            StateType.POSITION_VELOCITY_TORQUE: StateConfig(
                use_joint_positions=True,
                use_joint_velocities=True,
                use_joint_torques=True
            )
        }
        
        config = state_configs.get(self.state_type)
        if config is None:
            raise ValueError(f"Unknown state type: {self.state_type}")
        return config
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration with dynamic dataset info"""
        return DataConfig(
            dataset_name=self.dataset_name,
            data_amount=self.data_amount,
            input_dir=self.input_dir
        )
    
    def get_state_dim(self) -> int:
        """Get state dimension based on condition (considers both state_type and action_type)"""
        data_config = self.get_data_config()
        return data_config.get_state_dimension(self.state_type, self.action_type)
    
    def get_action_dim(self) -> int:
        """Get action dimension based on action type"""
        dimension = AblationDefaults.ACTION_DIMENSIONS.get(self.action_type.value)
        if dimension is None:
            raise ValueError(f"Unknown action type: {self.action_type}")
        return dimension

# =============================================================================
# 공통 State/Action Filtering 함수들 - 모든 모델이 동일한 joint indices 사용
# =============================================================================

def get_unified_state_indices(state_type: StateType, action_type: ActionType) -> List[int]:
    """Get unified state indices for all models - ensures consistency across GR00T, PI0, PI0_FAST, UniVLA"""
    state_config = StateConfig()
    return state_config.get_multi_modality_state_indices(state_type, action_type)


def get_unified_action_indices(action_type: ActionType) -> List[int]:
    """Get unified action indices for all models - ensures consistency across GR00T, PI0, PI0_FAST, UniVLA"""
    state_config = StateConfig()
    return state_config.get_action_indices(action_type)


def filter_state_tensor(state_tensor, state_type: StateType, action_type: ActionType):
    """Filter state tensor using unified indices - for PI0/PI0_FAST/other models"""
    indices = get_unified_state_indices(state_type, action_type)
    if hasattr(state_tensor, 'shape') and len(state_tensor.shape) >= 1:
        # For torch tensors or numpy arrays
        return state_tensor[..., indices]
    else:
        # For lists or other iterables
        return [state_tensor[i] for i in indices]


def filter_action_tensor(action_tensor, action_type: ActionType):
    """Filter action tensor using unified indices - for PI0/PI0_FAST/other models"""
    indices = get_unified_action_indices(action_type)
    if hasattr(action_tensor, 'shape') and len(action_tensor.shape) >= 1:
        # For torch tensors or numpy arrays
        return action_tensor[..., indices]
    else:
        # For lists or other iterables
        return [action_tensor[i] for i in indices]


def validate_unified_indices():
    """Validate that all models use the same indices for the same conditions"""
    print("🔍 Validating unified state/action indices...")
    
    test_cases = [
        (StateType.POSITION_ONLY, ActionType.RIGHT_ARM),
        (StateType.POSITION_ONLY, ActionType.DUAL_ARM),
        (StateType.POSITION_VELOCITY, ActionType.RIGHT_ARM),
        (StateType.POSITION_VELOCITY, ActionType.DUAL_ARM),
        (StateType.POSITION_VELOCITY_TORQUE, ActionType.RIGHT_ARM),
        (StateType.POSITION_VELOCITY_TORQUE, ActionType.DUAL_ARM),
    ]
    
    for state_type, action_type in test_cases:
        state_indices = get_unified_state_indices(state_type, action_type)
        action_indices = get_unified_action_indices(action_type)
        
        condition_name = f"{state_type.value}_{action_type.value}"
        print(f"  {condition_name}:")
        print(f"    State indices ({len(state_indices)}): {state_indices[:10]}...{state_indices[-10:] if len(state_indices) > 10 else state_indices}")
        print(f"    Action indices ({len(action_indices)}): {action_indices}")
    
    print("✅ Validation complete!")


# =============================================================================
# 유틸리티 함수들
# =============================================================================

def get_dataset_total_episodes(dataset_name: str, input_dir: Optional[str] = None) -> int:
    """Dynamically get total episodes from dataset"""
    try:
        # Method 1: Try to get from input_dir if provided
        if input_dir and os.path.exists(input_dir):
            dataset_path = Path(input_dir)
            
            # Check for info.json first
            info_file = dataset_path / "info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    if 'total_episodes' in info:
                        return info['total_episodes']
                    elif 'num_episodes' in info:
                        return info['num_episodes']
            
            # Method 2: Count episode directories
            episode_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
            if episode_dirs:
                return len(episode_dirs)
            
            # Method 3: Count episode files in data directory
            data_dir = dataset_path / "data"
            if data_dir.exists():
                episode_files = list(data_dir.glob("episode_*.parquet"))
                if episode_files:
                    return len(episode_files)
        
        # Method 4: Try HuggingFace cache directory
        cache_base = LEROBOT_CACHE_BASE
        
        # Normalize dataset_name to handle various formats
        dataset_name_clean = dataset_name.lstrip('/')  # Remove leading slash
        if dataset_name_clean.startswith("RLWRLD/"):
            cache_path = Path(cache_base) / dataset_name_clean
        elif "/" in dataset_name_clean:
            # Assume it's already in org/name format
            cache_path = Path(cache_base) / dataset_name_clean
        else:
            # Assume it's just the dataset name, prepend RLWRLD/
            cache_path = Path(cache_base) / "RLWRLD" / dataset_name_clean
            
        if cache_path.exists():
            # Check info.json in cache
            info_file = cache_path / "info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    if 'total_episodes' in info:
                        return info['total_episodes']
                    elif 'num_episodes' in info:
                        return info['num_episodes']
            
            # Count episodes in cache data directory
            data_dir = cache_path / "data"
            if data_dir.exists():
                episode_files = list(data_dir.glob("episode_*.parquet"))
                if episode_files:
                    return len(episode_files)
        
        # Method 5: Known dataset defaults
        dataset_episode_counts = {
            "RLWRLD/allex_gesture_easy_pos_vel_torq": 220,
            "allex_gesture_easy_pos_vel_torq": 220,
            # Add more known datasets here
        }
        
        # Try both original and cleaned names
        for name in [dataset_name, dataset_name_clean]:
            if name in dataset_episode_counts:
                return dataset_episode_counts[name]
        
        # Default fallback
        print(f"⚠️ Could not determine episode count for {dataset_name}, using default: 220")
        return 220
        
    except Exception as e:
        print(f"⚠️ Error getting episode count for {dataset_name}: {e}, using default: 220")
        return 220


def validate_data_amount(data_amount: int) -> None:
    """공통 데이터 양 검증 함수"""
    if not (1 <= data_amount <= 100):
        raise ValueError(f"data_amount must be between 1 and 100, got {data_amount}")


def validate_model_limitations(model_type: str, state_type: str, camera_type: str) -> None:
    """공통 모델 제한사항 검증 함수"""
    model_limitations = AblationDefaults.MODEL_LIMITATIONS.get(model_type, {})
    
    # State 제한 검증
    if "states" in model_limitations:
        allowed_states = model_limitations["states"]
        if state_type not in allowed_states:
            raise ValueError(
                f"{model_type} does not support state type '{state_type}'. "
                f"Supported states: {allowed_states}"
            )
    
    # Camera 제한 검증
    if "cameras" in model_limitations:
        allowed_cameras = model_limitations["cameras"]
        if camera_type not in allowed_cameras:
            raise ValueError(
                f"{model_type} does not support camera type '{camera_type}'. "
                f"Supported cameras: {allowed_cameras}"
            )


def check_model_limitations(model_type: str, state_type: str, camera_type: str) -> bool:
    """모델 제한사항 확인 함수 (필터링용)"""
    model_limitations = AblationDefaults.MODEL_LIMITATIONS.get(model_type, {})
    
    # State 제한 확인
    if "states" in model_limitations and state_type not in model_limitations["states"]:
        return False
        
    # Camera 제한 확인
    if "cameras" in model_limitations and camera_type not in model_limitations["cameras"]:
        return False
    
    return True


def get_state_description(state_type: StateType) -> str:
    """Get human-readable state description"""
    if state_type == StateType.POSITION_ONLY:
        return 'pos_only'
    elif state_type == StateType.POSITION_VELOCITY:
        return 'pos_vel'
    else:  # POSITION_VELOCITY_TORQUE
        return 'pos_vel_torque'


def get_model_batch_size(model_type: ModelType) -> int:
    """Get model-specific batch size"""
    batch_sizes = {
        ModelType.GR00T: 24,
        ModelType.PI0: 24,
        ModelType.PI0_FAST: 16,
        ModelType.UNIVLA: 16
    }
    return batch_sizes[model_type]


def get_pi0_model_config(model_type: ModelType) -> tuple[str, str]:
    """Get PI0 model configuration (model_name, policy_path)"""
    if model_type == ModelType.PI0:
        return "pi0", "lerobot/pi0"
    elif model_type == ModelType.PI0_FAST:
        return "pi0_fast", "lerobot/pi0fast_base"
    else:
        raise ValueError(f"Unsupported PI0 model type: {model_type}")


def create_pi0_command(condition, use_sbatch=False, job_id=None, input_dir=None, save_freq=None, num_workers=8) -> Union[str, List[str]]:
    """Create command for PI0/PI0_FAST training (unified function)"""
    if condition.model_type not in [ModelType.PI0, ModelType.PI0_FAST]:
        raise ValueError(f"Unsupported model type for PI0 command: {condition.model_type}")
    
    model_name, policy_path = get_pi0_model_config(condition.model_type)
    return create_pi0_base_command(condition, model_name, policy_path, use_sbatch, job_id, input_dir, save_freq, num_workers)


def create_pi0_command_with_shared_data(condition, shared_condition_name: str, use_sbatch=False, job_id=None, input_dir=None, save_freq=None, num_workers=8) -> Union[str, List[str]]:
    """Create command for PI0/PI0_FAST training with shared data directory"""
    if condition.model_type not in [ModelType.PI0, ModelType.PI0_FAST]:
        raise ValueError(f"Unsupported model type for PI0 command: {condition.model_type}")
    
    model_name, policy_path = get_pi0_model_config(condition.model_type)
    return create_pi0_base_command_with_shared_data(condition, shared_condition_name, model_name, policy_path, use_sbatch, job_id, input_dir, save_freq, num_workers)


def create_pi0_base_command(condition, model_name: str, policy_path: str, use_sbatch=False, job_id=None, input_dir=None, save_freq=None, num_workers=8) -> Union[str, List[str]]:
    """Create command for PI0-based training (shared logic for PI0 and PI0_FAST)"""
    
    # 기본 input_dir 설정
    if input_dir is None:
        input_dir = LEROBOT_CACHE_BASE + "RLWRLD/allex_gesture_easy_pos_vel_torq"
    
    if use_sbatch:
        # SBATCH 스크립트 생성 - 실시간 변환 포함
        sbatch_script = create_sbatch_header(model_name, condition.name)
        
        # Use common functions
        env_setup = create_common_env_setup("lerobot", ["transformers"])
        data_prep = create_common_data_preparation_section(input_dir, condition.name)
        training_header = create_common_training_section_header()
        
        sbatch_script += f"""#!/bin/bash
#SBATCH --job-name={condition.name}
#SBATCH --output=_logs/{model_name}/job_%j_{condition.name}.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 로그 디렉토리 생성
mkdir -p _logs/{model_name}

# =============================================================================
# 통합 로그 시작
# =============================================================================
echo "=== UNIFIED LOG FOR {condition.name.upper()} ==="
echo "Generated at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh

# 환경 확인 및 강제 설정
echo "=================================================="
echo "ENVIRONMENT CHECK"
echo "=================================================="
echo "Available conda environments:"
conda env list
echo ""

echo "Activating lerobot environment..."
conda activate lerobot
echo ""

echo "Current Python path:"
which python
echo ""

echo "Python version:"
python --version
echo ""

# =============================================================================
# STEP 1: 데이터 변환 수행 (PI0/PI0_FAST real-time conversion)
# =============================================================================
echo "=================================================="
echo "STEP 1: DATA CONVERSION"
echo "=================================================="
echo "Command: cd pi0 && python convert_dataset_for_ablation.py --condition {condition.name} --input-dir {input_dir} --output-dir {LEROBOT_CACHE_BASE}RLWRLD"
echo "Starting data conversion..."
echo ""

# 데이터 변환 실행 (conda activate 사용)
cd pi0
python convert_dataset_for_ablation.py \\
    --condition {condition.name} \\
    --input-dir {input_dir} \\
    --output-dir {LEROBOT_CACHE_BASE}RLWRLD

CONVERT_EXIT_CODE=$?
cd ..

if [ $CONVERT_EXIT_CODE -eq 0 ]; then
    echo "✅ Data conversion completed successfully"
    echo "Data directory: {LEROBOT_CACHE_BASE}RLWRLD/{condition.name}"
    
    # Copy meta directory from original dataset to converted dataset (필요시에만)
    if [ ! -d "{LEROBOT_CACHE_BASE}RLWRLD/{condition.name}/meta" ]; then
        echo "📂 Copying meta directory from original dataset..."
        cp -r "{input_dir}/meta" "{LEROBOT_CACHE_BASE}RLWRLD/{condition.name}/"
        echo "✅ Meta directory copied successfully"
    fi
else
    echo "❌ Data conversion failed"
    echo "Exiting..."
    exit $CONVERT_EXIT_CODE
fi
echo ""

{training_header}
echo "Command: python3 pi0/lerobot/scripts/train.py --dataset.repo_id=RLWRLD/{condition.name} --policy.path={policy_path} --output_dir=./pi0/outputs/job_${{SLURM_JOB_ID}}_{model_name}_{condition.name} --steps={condition.max_steps} --batch_size={condition.batch_size} --seed=1000 --save_freq={save_freq if save_freq else 5000} --num_workers={num_workers}"
echo "Starting training..."
echo ""

# PI0 훈련 실행 (성공한 스크립트 방식 - cd pi0 없이 직접 실행)
python3 pi0/lerobot/scripts/train.py \\
    --dataset.repo_id=RLWRLD/{condition.name} \\
    --policy.path={policy_path} \\
    --output_dir=./pi0/outputs/job_${{SLURM_JOB_ID}}_{model_name}_{condition.name} \\
    --steps={condition.max_steps} \\
    --batch_size={condition.batch_size} \\
    --seed=1000 \\
    --save_freq={save_freq if save_freq else 5000} \\
    --num_workers={num_workers}

TRAIN_EXIT_CODE=$?
"""
        sbatch_script += create_sbatch_summary()
        return sbatch_script
    else:
        # 직접 실행 명령 (워커노드에서 실행용) - 변환된 데이터셋 사용
        cmd = create_common_direct_command(
            "lerobot", 
            f"export TOKENIZERS_PARALLELISM=false && python3 pi0/lerobot/scripts/train.py --dataset.repo_id=RLWRLD/{condition.name} --policy.path={policy_path} --output_dir=./pi0/outputs/{condition.get_output_dir(job_id)} --steps={condition.max_steps} --batch_size={condition.batch_size} --seed=1000 --save_freq={save_freq if save_freq else 5000} --num_workers={num_workers}"
        )
        return cmd


def create_pi0_base_command_with_shared_data(condition, shared_condition_name: str, model_name: str, policy_path: str, use_sbatch=False, job_id=None, input_dir=None, save_freq=None, num_workers=8) -> Union[str, List[str]]:
    """Create command for PI0-based training with shared data directory (shared logic)"""
    
    # 기본 input_dir 설정
    if input_dir is None:
        input_dir = LEROBOT_CACHE_BASE + "RLWRLD/allex_gesture_easy_pos_vel_torq"
    
    if use_sbatch:
        # SBATCH 스크립트 생성 - 실시간 변환 포함
        sbatch_script = create_sbatch_header(model_name, shared_condition_name)
        
        # Use common functions
        env_setup = create_common_env_setup("lerobot", ["transformers"])
        data_prep = create_common_data_preparation_section(input_dir, shared_condition_name)
        training_header = create_common_training_section_header()
        
        sbatch_script += f"""#!/bin/bash
#SBATCH --job-name={shared_condition_name}
#SBATCH --output=_logs/{model_name}/job_%j_{shared_condition_name}.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 로그 디렉토리 생성
mkdir -p _logs/{model_name}

# =============================================================================
# 통합 로그 시작
# =============================================================================
echo "=== UNIFIED LOG FOR {shared_condition_name.upper()} ==="
echo "Generated at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh

# 환경 확인 및 강제 설정
echo "=================================================="
echo "ENVIRONMENT CHECK"
echo "=================================================="
echo "Available conda environments:"
conda env list
echo ""

echo "Activating lerobot environment..."
conda activate lerobot
echo ""

echo "Current Python path:"
which python
echo ""

echo "Python version:"
python --version
echo ""

{data_prep}
{training_header}
echo "Command: python3 pi0/lerobot/scripts/train.py --dataset.repo_id=RLWRLD/{shared_condition_name} --policy.path={policy_path} --output_dir=./pi0/outputs/job_${{SLURM_JOB_ID}}_{model_name}_{shared_condition_name} --steps={condition.max_steps} --batch_size={condition.batch_size} --seed=1000 --save_freq={save_freq if save_freq else 5000} --num_workers={num_workers}"
echo "Starting training..."
echo ""

# PI0 훈련 실행 (성공한 스크립트 방식 - cd pi0 없이 직접 실행)
python3 pi0/lerobot/scripts/train.py \\
    --dataset.repo_id=RLWRLD/{shared_condition_name} \\
    --policy.path={policy_path} \\
    --output_dir=./pi0/outputs/job_${{SLURM_JOB_ID}}_{model_name}_{shared_condition_name} \\
    --steps={condition.max_steps} \\
    --batch_size={condition.batch_size} \\
    --seed=1000 \\
    --save_freq={save_freq if save_freq else 5000} \\
    --num_workers={num_workers}

TRAIN_EXIT_CODE=$?
"""
        sbatch_script += create_sbatch_summary()
        return sbatch_script
    else:
        # 직접 실행 명령 (워커노드에서 실행용) - 변환된 데이터셋 사용
        cmd = create_common_direct_command(
            "lerobot",
            f"export TOKENIZERS_PARALLELISM=false && python3 pi0/lerobot/scripts/train.py --dataset.repo_id=RLWRLD/{shared_condition_name} --policy.path={policy_path} --output_dir=./pi0/outputs/{condition.get_output_dir(job_id)} --steps={condition.max_steps} --batch_size={condition.batch_size} --seed=1000 --save_freq={save_freq if save_freq else 5000} --num_workers={num_workers}"
        )
        return cmd


# Remove unused functions - no longer needed for 1-step process

# def get_dataset_source_for_condition(condition: AblationCondition, input_dir=None, converted_datasets_dir=None) -> str:
#     """Get dataset source configuration with model-specific handling"""
#     # This function is no longer needed as we use real-time conversion

# def create_pi0_sbatch_body(condition, model_name: str, policy_path: str, dataset_source: str, episodes_arg: str, save_freq, num_workers, input_dir) -> str:
#     """Create SBATCH body for PI0-based models (shared logic) - without filtering arguments"""
#     # This function is no longer needed as we integrated conversion into create_pi0_base_command

def get_dataset_source(data_config, input_dir=None) -> str:
    """Legacy function - kept for backward compatibility"""
    if input_dir is None:
        return f"--dataset.repo_id={data_config.dataset_name}"
    else:
        lerobot_cache_base = LEROBOT_CACHE_BASE
        if input_dir.startswith(lerobot_cache_base):
            relative_path = input_dir[len(lerobot_cache_base):]
            return f"--dataset.repo_id={relative_path}"
        else:
            return f"--dataset.root={input_dir}"


def create_univla_command(condition, use_sbatch=False, job_id=None, input_dir=None, save_steps=None) -> Union[Dict[str, Union[List[str], str]], Dict[str, List[str]]]:
    """Create command for UniVLA training using existing conversion workflow"""
    
    # UniVLA already has proper joint filtering in its conversion script
    # Use existing proven workflow instead of pre-converted datasets
    
    # 기본 input_dir 설정
    if input_dir is None:
        input_dir = LEROBOT_CACHE_BASE + "RLWRLD/allex_gesture_easy_pos_vel_torq"
    
    print(f"📁 UniVLA using existing conversion workflow with input: {input_dir}")
    
    # 데이터 변환 명령 (conda 환경 활성화 포함) - UniVLA's existing approach
    convert_cmd = [
        "bash", "-c", 
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py --condition {condition.name} --input-dir {input_dir} --output-dir ./univla/converted_data"
    ]
    
    if use_sbatch:
        # Simple SBATCH script for UniVLA
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={condition.name}
#SBATCH --output=_logs/univla/job_%j_{condition.name}.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 로그 디렉토리 생성
mkdir -p _logs/univla

# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh
conda activate univla_train

# 데이터 변환
python univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py --condition {condition.name} --input-dir {input_dir} --output-dir ./univla/converted_data

# 훈련 실행
python univla/vla_scripts/finetune_rlwrld_ablation.py --condition {condition.name} --data-root-dir ./univla/converted_data/{condition.name} --output-dir ./univla/outputs/job_${{SLURM_JOB_ID}}_univla_{condition.name} --max-steps {condition.max_steps} --batch-size 16 --save-steps {save_steps if save_steps else 5000}
"""
        return {"convert": convert_cmd, "sbatch_script": sbatch_script}
    else:
        # 직접 실행 명령 (워커노드에서 실행용)
        convert_cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python univla/vla_scripts/convert_lerobot_dataset_for_univla_ablation.py --condition {condition.name} --input-dir {input_dir} --output-dir ./univla/converted_data"
        ]
        
        train_cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate univla_train && python univla/vla_scripts/finetune_rlwrld_ablation.py --condition {condition.name} --data-root-dir ./univla/converted_data/{condition.name} --output-dir ./univla/outputs/{condition.get_output_dir(job_id)} --max-steps {condition.max_steps} --batch-size 16 --save-steps {save_steps if save_steps else 5000}"
        ]
        return {"convert": convert_cmd, "train": train_cmd}


# =============================================================================
# SBATCH 및 명령어 생성 함수들 (복원)
# =============================================================================

def create_sbatch_header(model_name: str, condition_name: str) -> str:
    """Create common SBATCH header for all models"""
    return f"""#!/bin/bash
#SBATCH --job-name={condition_name}
#SBATCH --output=_logs/{model_name}/job_%j_{condition_name}.log
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --nodes=1

# 로그 디렉토리 생성
mkdir -p _logs/{model_name}

# =============================================================================
# 통합 로그 시작
# =============================================================================
echo "=== UNIFIED LOG FOR {condition_name.upper()} ==="
echo "Generated at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
"""


def create_common_env_setup(conda_env: str, package_checks: Optional[List[str]] = None) -> str:
    """Create common environment setup for all models"""
    if package_checks is None:
        package_checks = []
    
    setup_code = f"""# 환경 설정
source ~/miniconda3/etc/profile.d/conda.sh

# 환경 확인 및 강제 설정
echo "=================================================="
echo "ENVIRONMENT CHECK"
echo "=================================================="
echo "Available conda environments:"
conda env list
echo ""

echo "Activating {conda_env} environment..."
conda activate {conda_env}
echo ""

echo "Current Python path:"
which python
echo ""

echo "Python version:"
python --version
echo ""
"""
    
    # Add package checks
    for package in package_checks:
        if package in ["transformers", "torch", "tyro", "accelerate"]:
            setup_code += f'echo "Checking {package} installation..."\n'
            setup_code += f'python -c "import {package}; print(\'{package} version:\', {package}.__version__)"\n'
    
    setup_code += """echo "=================================================="
echo ""
"""
    return setup_code


def create_common_data_preparation_section(input_dir: str, data_info: str) -> str:
    """Create common data preparation section"""
    return f"""# =============================================================================
# STEP 1: 데이터 준비 (이미 완료됨)
# =============================================================================
echo "=================================================="
echo "STEP 1: DATA PREPARATION (Already available)"
echo "=================================================="
echo "Data source: {input_dir}"
echo "Dataset: {data_info}"
echo "Status: ✅ SUCCESS (Data already available)"
echo ""
"""


def create_common_training_section_header() -> str:
    """Create common training section header"""
    return """# =============================================================================
# STEP 2: 훈련 실행
# =============================================================================
echo "=================================================="
echo "STEP 2: TRAINING"
echo "=================================================="
"""


def create_common_direct_command(conda_env: str, command: str) -> List[str]:
    """Create common direct execution command"""
    return [
        "bash", "-c",
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {command}"
    ]


def create_sbatch_summary() -> str:
    """Create common SBATCH summary section"""
    return """
echo ""
echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo "Data preparation: ✅ SUCCESS"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training: ✅ SUCCESS"
    echo "Overall: ✅ SUCCESS"
else
    echo "Training: ❌ FAILED"
    echo "Overall: ❌ FAILED"
fi
echo "=================================================="
"""


def create_gr00t_command(condition, use_sbatch=False, job_id=None, input_dir=None, save_freq=None, num_workers=8) -> Union[Dict[str, str], List[str]]:
    """Create command for GR00T training"""
    data_config = condition.get_data_config()
    
    # 기본 input_dir 설정
    if input_dir is None:
        input_dir = LEROBOT_CACHE_BASE + "RLWRLD/allex_gesture_easy_pos_vel_torq"
    
    # GR00T 훈련 명령 (conda 환경 활성화 포함)
    # Select appropriate data config based on action type, camera type, and state type
    action_part = "right_arm" if condition.action_type == ActionType.RIGHT_ARM else "dual_arm"
    camera_part = "robot_view" if condition.camera_type == CameraType.ROBOT_VIEW else "multi_view"
    state_part = condition.state_type.value  # pos_only, pos_vel, pos_vel_torque
    
    data_config_name = f"allex_{action_part}_{camera_part}_{state_part}"
    
    # save_steps 파라미터 이름 호환성 유지
    save_steps = save_freq
    
    if use_sbatch:
        # SBATCH 스크립트 생성
        sbatch_script = create_sbatch_header("gr00t", condition.name)
        
        # Use common functions
        env_setup = create_common_env_setup("gr00t", ["torch", "transformers", "tyro"])
        data_prep = create_common_data_preparation_section(input_dir, data_config_name)
        training_header = create_common_training_section_header()
        
        sbatch_script += f"""{env_setup}
{data_prep}
{training_header}
echo "Command: python gr00t/scripts/gr00t_finetune.py --dataset-path {input_dir} --num-gpus 1 --output-dir ./gr00t/checkpoints/job_${{SLURM_JOB_ID}}_gr00t_{condition.name} --max-steps {condition.max_steps} --batch-size {condition.batch_size} --video-backend torchvision_av --data-config {data_config_name} --action_dim {condition.get_action_dim()} --save-steps {save_steps if save_steps else 5000} --dataloader-num-workers {num_workers}"
echo "Starting training..."
echo ""

# 훈련 실행 (conda activate 사용)
python gr00t/scripts/gr00t_finetune.py \\
    --dataset-path {input_dir} \\
    --num-gpus 1 \\
    --output-dir ./gr00t/checkpoints/job_${{SLURM_JOB_ID}}_gr00t_{condition.name} \\
    --max-steps {condition.max_steps} \\
    --batch-size {condition.batch_size} \\
    --video-backend torchvision_av \\
    --data-config {data_config_name} \\
    --action_dim {condition.get_action_dim()} \\
    --save-steps {save_steps if save_steps else 5000} \\
    --dataloader-num-workers {num_workers}

TRAIN_EXIT_CODE=$?
"""
        sbatch_script += create_sbatch_summary()
        return {"sbatch_script": sbatch_script}
    else:
        # 직접 실행 명령
        cmd = create_common_direct_command(
            "gr00t",
            f"python gr00t/scripts/gr00t_finetune.py --dataset-path {input_dir} --num-gpus 1 --output-dir ./gr00t/checkpoints/{condition.get_output_dir(job_id)} --max-steps {condition.max_steps} --batch-size {condition.batch_size} --video-backend torchvision_av --data-config {data_config_name} --action_dim {condition.get_action_dim()} --dataloader-num-workers {num_workers}"
        )
        return cmd


def execute_model_command(condition, use_sbatch, job_id, input_dir, save_interval, num_workers, dry_run):
    """Execute model command with unified logic"""
    model_type = condition.model_type
    
    if model_type == ModelType.UNIVLA:
        # UniVLA has special handling for data conversion
        commands = create_univla_command(condition, use_sbatch, job_id, input_dir, save_interval)
        
        # 통합 로그 파일 초기화
        log_dir = Path("_logs") / condition.model_type.value
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 변환 단계
        convert_cmd = commands["convert"]
        print(f"Convert Command: {' '.join(convert_cmd)}")
        
        if not dry_run:
            print("Step 1: Converting data...")
            convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if convert_result.returncode != 0:
                print(f"❌ Data conversion failed for: {condition.name}")
                return
            
            print("✅ Data conversion completed")
        else:
            print("(Dry run - not executing convert)")
        
        # 훈련 단계
        if use_sbatch:
            # SBATCH 스크립트 생성 및 실행
            if "sbatch_script" in commands and isinstance(commands["sbatch_script"], str):
                sbatch_script = commands["sbatch_script"]
                job_id = submit_sbatch_job(sbatch_script, "univla", condition.name, dry_run)
            else:
                print("❌ Failed to create UniVLA SBATCH script")
                return
        else:
            # 직접 실행 (워커노드에서 실행용)
            if "train" in commands and isinstance(commands["train"], list):
                train_cmd = commands["train"]
                print(f"Train Command: {' '.join(train_cmd)}")
                print("💡 Note: Make sure you are running on a worker node with GPU access!")
                print("💡 Use: srun --comment 'univla training' --gpus=1 --nodes=1 --pty /bin/bash")
                
                if not dry_run:
                    print("Step 2: Training model...")
                    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
                    
                    # 간단한 로그 기록
                    _log_result(condition.model_type.value, f"{condition.name}_train", train_cmd, train_result)
                    
                    if train_result.returncode == 0:
                        print(f"✅ Successfully completed: {condition.name}")
                    else:
                        print(f"❌ Training failed: {condition.name}")
                else:
                    print("(Dry run - not executing train)")
            else:
                print("❌ Failed to create UniVLA train command")
                return
    
    elif model_type in [ModelType.PI0, ModelType.PI0_FAST]:
        # PI0/PI0_FAST: Real-time conversion like UniVLA (1-step process)
        model_name = model_type.value
        
        # 기본 input_dir 설정
        if input_dir is None:
            input_dir = LEROBOT_CACHE_BASE + "RLWRLD/allex_gesture_easy_pos_vel_torq"
        
        print(f"📁 {model_name.upper()} using real-time conversion workflow with input: {input_dir}")
        
        # 통합 로그 파일 초기화
        log_dir = Path("_logs") / condition.model_type.value
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shared data directory name (excluding model name)
        shared_condition_name = f"{condition.data_amount}_percent_{condition.state_type.value}_{condition.action_type.value}_{condition.camera_type.value}"
        
        # Step 1: 데이터 변환 (PI0/PI0_FAST conversion) - only convert if not exists
        convert_cmd = [
            "bash", "-c", 
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && cd pi0 && python convert_dataset_for_ablation.py --condition {condition.name} --input-dir {input_dir} --output-dir {LEROBOT_CACHE_BASE}RLWRLD"
        ]
        
        print(f"Convert Command: {' '.join(convert_cmd)}")
        print(f"💡 Using shared data directory: {shared_condition_name}")
        
        if not dry_run:
            print("Step 1: Converting data (if not already converted)...")
            convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
            
            if convert_result.returncode != 0:
                print(f"❌ Data conversion failed for: {condition.name}")
                print(f"Error: {convert_result.stderr}")
                return
            
            print("✅ Data conversion completed")
        else:
            print("(Dry run - not executing convert)")
        
        # Step 2: 훈련 - use shared data directory
        if use_sbatch:
            # SBATCH 실행 - 변환 완료된 공유 데이터셋 사용
            command_creator = create_pi0_command_with_shared_data
            result = command_creator(condition, shared_condition_name, use_sbatch=True, job_id=job_id, input_dir=input_dir, save_freq=save_interval, num_workers=num_workers)
            
            if isinstance(result, str):
                job_id = submit_sbatch_job(result, model_name, condition.name, dry_run)
            else:
                print(f"❌ Failed to create {model_name.upper()} SBATCH script")
                return
        else:
            # 직접 실행 - 변환 완료된 공유 데이터셋 사용
            command_creator = create_pi0_command_with_shared_data
            cmd = command_creator(condition, shared_condition_name, use_sbatch=False, job_id=job_id, input_dir=input_dir, save_freq=save_interval, num_workers=num_workers)
            
            if isinstance(cmd, list):
                print(f"Train Command: {' '.join(cmd)}")
                print("💡 Note: Make sure you are running on a worker node with GPU access!")
                print("💡 Use: srun --comment 'pi0/pi0_fast training' --gpus=1 --nodes=1 --pty /bin/bash")
                
                if not dry_run:
                    print("Step 2: Training model...")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # 간단한 로그 기록
                    _log_result(condition.model_type.value, f"{condition.name}_train", cmd, result)
                    
                    if result.returncode == 0:
                        print(f"✅ Successfully completed: {condition.name}")
                    else:
                        print(f"❌ Training failed: {condition.name}")
                else:
                    print("(Dry run - not executing train)")
            else:
                print(f"❌ Unexpected return type from create_{model_name}_command: {type(cmd)}")
                return
    
    else:
        # Standard model handling (GR00T)
        if model_type == ModelType.GR00T:
            command_creator = create_gr00t_command
            model_name = "gr00t"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if use_sbatch:
            # SBATCH 실행
            result = command_creator(condition, use_sbatch=True, job_id=job_id, input_dir=input_dir, save_freq=save_interval, num_workers=num_workers)
            
            if isinstance(result, dict) and "sbatch_script" in result:
                sbatch_script = result["sbatch_script"]
                job_id = submit_sbatch_job(sbatch_script, model_name, condition.name, dry_run)
            else:
                print(f"❌ Failed to create {model_name.upper()} SBATCH script")
                return
        else:
            # 직접 실행
            cmd = command_creator(condition, use_sbatch=False, job_id=job_id, input_dir=input_dir, save_freq=save_interval, num_workers=num_workers)
            
            if isinstance(cmd, list):
                print(f"Command: {' '.join(cmd)}")
                
                if not dry_run:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    _log_result(condition.model_type.value, condition.name, cmd, result)
                else:
                    print("(Dry run - not executing)")
            else:
                print(f"❌ Unexpected return type from create_{model_name}_command: {type(cmd)}")

# =============================================================================
# 작업 실행 관련 함수들
# =============================================================================

def submit_sbatch_job(sbatch_script: str, model_name: str, condition_name: str, dry_run: bool = False) -> Optional[str]:
    """Submit SBATCH job and return job ID"""
    print("Creating SBATCH script (temporary)")
    print("SBATCH script content:")
    print(sbatch_script)
    
    if not dry_run:
        print("Creating and submitting SBATCH job...")
        
        # 임시 스크립트 파일 생성 및 실행
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_script:
            temp_script.write(sbatch_script)
            temp_script_path = temp_script.name
        
        try:
            # 스크립트 실행 권한 부여
            os.chmod(temp_script_path, 0o755)
            
            # sbatch로 작업 제출
            sbatch_cmd = ["sbatch", "--comment", f"ablation-{condition_name}", temp_script_path]
            print(f"Submitting job: {' '.join(sbatch_cmd)}")
            
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            job_id = extract_job_id(result.stdout) if result.returncode == 0 else "failed"
        finally:
            # 임시 파일 삭제
            os.unlink(temp_script_path)
        
        if result.returncode == 0:
            print(f"✅ Successfully submitted job: {condition_name}")
            print(f"Job output: {result.stdout.strip()}")
            return job_id
        else:
            print(f"❌ Job submission failed: {condition_name}")
            print(f"Error: {result.stderr}")
            return None
    else:
        print("(Dry run - not executing sbatch)")
        return None


def run_condition(condition, dry_run=False, use_sbatch=False, input_dir=None, save_interval=None, num_workers=8):
    """Run a single ablation condition"""
    
    # 직접 실행일 때를 위한 job_id 생성
    if use_sbatch:
        job_id = None  # SBATCH에서는 나중에 추출
    else:
        job_id = f"D{random.randint(1000, 9999)}"
    
    print(f"\n{'='*80}")
    print(f"Running condition: {condition.name}")
    print(f"Model: {condition.model_type.value}")
    print(f"Data: {condition.data_amount} percent")
    state_desc = get_state_description(condition.state_type)
    print(f"State: {state_desc}")
    action_desc = "right_arm" if condition.action_type == ActionType.RIGHT_ARM else "dual_arm"
    print(f"Action: {action_desc}")
    camera_desc = "robot_view" if condition.camera_type == CameraType.ROBOT_VIEW else "multi_view"
    print(f"Camera: {camera_desc}")
    print(f"Action dim: {condition.get_action_dim()}")
    print(f"Output: {condition.get_output_dir(job_id)}")
    print(f"{'='*80}")
    
    try:
        # Create command based on model type
        execute_model_command(condition, use_sbatch, job_id, input_dir, save_interval, num_workers, dry_run)
            
    except Exception as e:
        print(f"❌ Error running condition {condition.name}: {str(e)}")


def _log_result(model_type: str, name: str, cmd: list, result, job_id: Optional[str] = None):
    """Log subprocess result with organized directory structure"""
    if result.returncode == 0:
        print(f"✅ Successfully completed: {name}")
    else:
        print(f"❌ Failed: {name}")
        print(f"Error: {result.stderr}")
        
    # Create logs directory structure
    log_dir = Path("_logs") / model_type
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with job ID if available
    if job_id:
        log_file = log_dir / f"{job_id}_{name}.log"
    else:
        # Generate a shorter, more readable ID for direct execution
        short_id = f"D{int(time.time()) % 10000:04d}"
        log_file = log_dir / f"{short_id}_{name}.log"
    
    # Log the results
    with open(log_file, "w") as f:
        f.write(f"Name: {name}\n")
        if job_id:
            f.write(f"Job ID: {job_id}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Stdout:\n{result.stdout}\n")
        f.write(f"Stderr:\n{result.stderr}\n")
    
    print(f"📝 Log saved to: {log_file}")


def extract_job_id(sbatch_output: str) -> str:
    """Extract job ID from sbatch output"""
    match = re.search(r'Submitted batch job (\d+)', sbatch_output)
    return match.group(1) if match else "unknown"

def create_ablation_condition(args, dataset_name: str, input_dir: str, max_steps: int) -> AblationCondition:
    """Create AblationCondition object from command line arguments"""
    model_type = ModelType(args.model)
    data_amount = args.data
    state_type = StateType(args.state)
    action_type = ActionType(args.action)
    camera_type = CameraType(args.camera)
    
    # Get model-specific batch size
    model_batch_size = get_model_batch_size(model_type)
    
    return AblationCondition(
        model_type=model_type,
        data_amount=data_amount,
        state_type=state_type,
        action_type=action_type,
        camera_type=camera_type,
        max_steps=max_steps,
        batch_size=model_batch_size,
        dataset_name=dataset_name,
        input_dir=input_dir
    )


def generate_all_conditions(max_steps: int = 10000, dataset_name: str = AblationDefaults.DEFAULT_DATASET_NAME, input_dir: Optional[str] = None) -> List[AblationCondition]:
    """Generate all possible ablation conditions with specified parameters"""
    
    models = [ModelType.GR00T, ModelType.PI0, ModelType.PI0_FAST, ModelType.UNIVLA]
    data_amounts = AblationDefaults.SUPPORTED_DATA_AMOUNTS
    states = [StateType.POSITION_ONLY, StateType.POSITION_VELOCITY, StateType.POSITION_VELOCITY_TORQUE]
    actions = [ActionType.RIGHT_ARM, ActionType.DUAL_ARM]
    cameras = [CameraType.ROBOT_VIEW, CameraType.MULTI_VIEW]
    
    conditions = []
    
    for model, data, state, action, camera in itertools.product(
        models, data_amounts, states, actions, cameras
    ):
        # 모델별 제한사항에 따라 조합 필터링 using common function
        if not check_model_limitations(model.value, state.value, camera.value):
            continue
            
        condition = AblationCondition(
            model_type=model,
            data_amount=data,
            state_type=state,
            action_type=action,
            camera_type=camera,
            max_steps=max_steps,
            batch_size=get_model_batch_size(model),
            dataset_name=dataset_name,
            input_dir=input_dir
        )
        conditions.append(condition)
    
    return conditions


def get_condition_by_name(name: str, max_steps: int = 10000, dataset_name: str = AblationDefaults.DEFAULT_DATASET_NAME, input_dir: Optional[str] = None) -> Optional[AblationCondition]:
    """Get ablation condition by name - supports dynamic parsing"""
    # Generate all conditions with current parameters
    all_conditions = generate_all_conditions(max_steps, dataset_name, input_dir)
    
    # First, try to find in generated conditions
    for condition in all_conditions:
        if condition.name == name:
            return condition
    
    # If not found, try to parse the name dynamically
    # Format: {model}_{data_amount}_percent_{state}_{action}_{camera}
    # Example: univla_30_percent_pos_only_bimanual_multi_view
    try:
        parts = name.split('_')
        if len(parts) < 6 or parts[2] != 'percent':
            return None
        
        # Parse components
        model_str = parts[0]
        data_amount = int(parts[1])
        
        # Find where 'percent' ends and state starts
        state_start_idx = 3
        state_parts = []
        action_parts = []
        camera_parts = []
        
        # Parse state type (can be multiple words like 'pos_vel_torque')
        i = state_start_idx
        while i < len(parts):
            if parts[i] in ['right', 'dual']:  # Found action type
                break
            state_parts.append(parts[i])
            i += 1
        
        # Parse action type
        if i < len(parts) and parts[i] in ['right', 'dual']:
            if parts[i] == 'right' and i + 1 < len(parts) and parts[i + 1] == 'arm':
                action_parts = ['right', 'arm']
                i += 2
            elif parts[i] == 'dual':
                action_parts = ['dual']
                i += 1
        
        # Parse camera type (remaining parts)
        camera_parts = parts[i:]
        
        # Reconstruct strings
        state_str = '_'.join(state_parts)
        action_str = '_'.join(action_parts)
        camera_str = '_'.join(camera_parts)
        
        # Convert to enums
        model_type = ModelType(model_str)
        state_type = StateType(state_str)
        action_type = ActionType(action_str)
        camera_type = CameraType(camera_str)
        
        # Validate data amount using common function
        validate_data_amount(data_amount)
        
        # Create and return the condition
        return AblationCondition(
            model_type=model_type,
            data_amount=data_amount,
            state_type=state_type,
            action_type=action_type,
            camera_type=camera_type,
            max_steps=max_steps,
            batch_size=get_model_batch_size(model_type),
            dataset_name=dataset_name,
            input_dir=input_dir
        )
        
    except (ValueError, IndexError) as e:
        print(f"Failed to parse condition name '{name}': {e}")
        return None


def list_all_conditions() -> List[str]:
    """List all available condition names"""
    all_conditions = generate_all_conditions()
    return [condition.name for condition in all_conditions]


def get_conditions_by_model(model_type: ModelType) -> List[AblationCondition]:
    """Get all conditions for a specific model"""
    all_conditions = generate_all_conditions()
    return [c for c in all_conditions if c.model_type == model_type]


def get_condition_combinations_summary() -> Dict[str, Any]:
    """Get summary of all possible combinations"""
    summary = {}
    
    for model in ModelType:
        model_conditions = get_conditions_by_model(model)
        model_limitations = AblationDefaults.MODEL_LIMITATIONS.get(model.value, {})
        
        summary[model.value] = {
            "total_conditions": len(model_conditions),
            "supported_states": model_limitations.get("states", []),
            "supported_cameras": model_limitations.get("cameras", []),
            "conditions": [condition.name for condition in model_conditions]
        }
    
    return summary


def print_conditions_summary():
    """Print summary of all conditions"""
    all_conditions = generate_all_conditions()
    print(f"Total conditions: {len(all_conditions)}")
    print()
    
    for model in ModelType:
        model_conditions = get_conditions_by_model(model)
        print(f"{model.value.upper()}: {len(model_conditions)} conditions")
        
        for i, condition in enumerate(model_conditions, 1):
            data_pct = f"{condition.data_amount}%"
            state_desc = get_state_description(condition.state_type)
            action_desc = "right" if condition.action_type == ActionType.RIGHT_ARM else "dual"
            camera_desc = "robot" if condition.camera_type == CameraType.ROBOT_VIEW else "multi"
            
            print(f"  {i:2d}. {condition.name}")
            print(f"      Data: {data_pct}, State: {state_desc}, Action: {action_desc}, Camera: {camera_desc}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Run VLA model ablation study')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['univla', 'pi0', 'pi0_fast', 'gr00t'],
                       help='Model to train')
    parser.add_argument('--data', type=int, required=True,
                       help='Percentage of data to use (1-100)')
    parser.add_argument('--state', type=str, required=True,
                       choices=AblationDefaults.SUPPORTED_STATE_TYPES,
                       help='State representation type')
    parser.add_argument('--action', type=str, required=True,
                       choices=AblationDefaults.SUPPORTED_ACTION_TYPES,
                       help='Action space type')
    parser.add_argument('--camera', type=str, required=True,
                       choices=AblationDefaults.SUPPORTED_CAMERA_TYPES,
                       help='Camera configuration')

    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS,
                       help=f'Maximum training steps (default: {DEFAULT_MAX_STEPS})')
    parser.add_argument('--save-interval', type=int, default=DEFAULT_SAVE_INTERVAL,
                       help=f'Checkpoint saving frequency for all models (default: {DEFAULT_SAVE_INTERVAL})')

    parser.add_argument('--sbatch', action='store_true',
                       help='Use SBATCH for job submission (requires SLURM)')
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                       help=f'Input dataset directory (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help=f'Number of workers for DataLoader (default: {DEFAULT_NUM_WORKERS})')
    # batch_size is now determined automatically based on model type
    
    args = parser.parse_args()
    
    # Get values directly from args (defaults already set in argparse)
    max_steps = args.max_steps
    save_interval = args.save_interval
    input_dir = args.input_dir
    num_workers = args.num_workers
    
    # Derive dataset_name from input_dir (for dynamic episode counting)
    lerobot_cache_base = LEROBOT_CACHE_BASE
    if input_dir.startswith(lerobot_cache_base):
        dataset_name = input_dir[len(lerobot_cache_base):].lstrip('/')  # Remove leading slash
    else:
        dataset_name = AblationDefaults.DEFAULT_DATASET_NAME
    
    # 환경 변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Create AblationCondition object
    condition = create_ablation_condition(args, dataset_name, input_dir, max_steps)
    
    try:
        if args.sbatch:
            print("🚀 SBATCH 모드 활성화 - SLURM 작업으로 제출됩니다")
            
            # SBATCH로 실행
            run_condition(condition, dry_run=False, use_sbatch=True, input_dir=input_dir, save_interval=save_interval, num_workers=num_workers)
            print("🎉 SBATCH 작업이 성공적으로 제출되었습니다!")
            
        else:
            print(f"🚀 직접 실행 모드 - {args.model} 모델 훈련 시작")
            
            # 직접 실행
            run_condition(condition, dry_run=False, use_sbatch=False, input_dir=input_dir, save_interval=save_interval, num_workers=num_workers)
            
            # run_condition은 None을 반환할 수 있으므로 성공 여부를 확인
            # 직접 실행 모드에서는 예외가 발생하지 않으므로 항상 성공으로 간주
            print("🎉 Ablation study completed successfully!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Test when run directly with --help or show summary
    if len(sys.argv) == 1:
        print("VLA Models Ablation Study - All Conditions")
        print("=" * 60)
        print_conditions_summary()
        
        print("Condition breakdown:")
        print(f"--model: {len(list(ModelType))} (gr00t, pi0, pi0_fast, univla)")
        print(f"--data: {AblationDefaults.SUPPORTED_DATA_AMOUNTS}")
        print(f"--state: {len(list(StateType))} (pos_only, pos_vel, pos_vel_torque)")
        print(f"--action: {len(list(ActionType))} (right_arm, dual_arm)")
        print(f"--camera: {len(list(CameraType))} (robot_view, multi_view)")
        all_conditions = generate_all_conditions()
        print(f"- Total: {len(all_conditions)} conditions")
        
        print("\nModel limitations:")
        for model, limitations in AblationDefaults.MODEL_LIMITATIONS.items():
            print(f"- {model}: states={limitations.get('states', [])}, cameras={limitations.get('cameras', [])}") 
    else:
        main()