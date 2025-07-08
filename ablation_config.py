#!/usr/bin/env python
"""
Ablation Study Configuration for VLA Models
Supports gr00t, pi0, and univla with different experimental conditions
Complete factorial design: 3 models × 2 data × 2 state × 2 action × 2 camera = 48 experiments
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import itertools


class ModelType(Enum):
    GR00T = "gr00t"
    PI0 = "pi0"
    UNIVLA = "univla"


class DataAmount(Enum):
    PERCENT_20 = "20_percent"    # 20% 데이터
    PERCENT_100 = "100_percent"  # 100% 데이터


class StateType(Enum):
    POSITION_ONLY = "pos_only"        # joint position만 (torque&velocity X)
    FULL_STATE = "full_state"         # position + torque + velocity (torque&velocity O)


class ActionType(Enum):
    SINGLE_ARM = "single_arm"  # 몸+오른팔+오른손
    BIMANUAL = "bimanual"      # 몸+오른팔+왼팔+오른손+왼손


class CameraType(Enum):
    ROBOT_VIEW = "robot_view"     # robotview만
    MULTI_VIEW = "multi_view"     # robotview+sideview+wrist_views


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
        if self.torso_indices is None:
            self.torso_indices = list(range(0, 3))      # 0-2
        if self.head_indices is None:
            self.head_indices = list(range(3, 5))       # 3-4
        if self.right_arm_indices is None:
            self.right_arm_indices = list(range(5, 12)) # 5-11
        if self.left_arm_indices is None:
            self.left_arm_indices = list(range(12, 19)) # 12-18
        if self.right_hand_indices is None:
            self.right_hand_indices = list(range(19, 40)) # 19-39
        if self.left_hand_indices is None:
            self.left_hand_indices = list(range(40, 60))  # 40-59
    
    def get_state_indices(self, action_type: ActionType) -> List[int]:
        """Get state indices based on action type"""
        indices = []
        
        # Always include torso
        if self.torso_indices:
            indices.extend(self.torso_indices)
        
        if action_type == ActionType.SINGLE_ARM:
            # 몸+오른팔+오른손
            if self.right_arm_indices:
                indices.extend(self.right_arm_indices)
            if self.right_hand_indices:
                indices.extend(self.right_hand_indices)
        elif action_type == ActionType.BIMANUAL:
            # 몸+오른팔+왼팔+오른손+왼손
            if self.right_arm_indices:
                indices.extend(self.right_arm_indices)
            if self.left_arm_indices:
                indices.extend(self.left_arm_indices)
            if self.right_hand_indices:
                indices.extend(self.right_hand_indices)
            if self.left_hand_indices:
                indices.extend(self.left_hand_indices)
        
        return sorted(indices)


@dataclass
class DataConfig:
    """Data configuration for ablation study"""
    dataset_name: str = "RLWRLD/allex_cube"
    data_amount: DataAmount = DataAmount.PERCENT_100
    
    # Camera configurations
    camera_keys: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.camera_keys is None:
            self.camera_keys = ["video.camera_ego"]  # Default robot view
    
    def get_num_episodes(self) -> Optional[int]:
        """Get number of episodes based on data amount"""
        if self.data_amount == DataAmount.PERCENT_20:
            return 12  # 20% of ~60 episodes
        elif self.data_amount == DataAmount.PERCENT_100:
            return None  # Use all episodes
        return None
    
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
    data_amount: DataAmount
    state_type: StateType
    action_type: ActionType
    camera_type: CameraType
    
    # Training specific
    max_steps: int = 10000
    batch_size: int = 8
    learning_rate: float = 1e-4
    
    @property
    def name(self) -> str:
        """Generate unique name for this condition"""
        return f"{self.model_type.value}_{self.data_amount.value}_{self.state_type.value}_{self.action_type.value}_{self.camera_type.value}"
    
    def get_output_dir(self) -> str:
        """Generate output directory name"""
        return self.name
    
    def get_state_config(self) -> StateConfig:
        """Get state configuration based on state type"""
        if self.state_type == StateType.POSITION_ONLY:
            return StateConfig(
                use_joint_positions=True,
                use_joint_velocities=False,
                use_joint_torques=False
            )
        elif self.state_type == StateType.FULL_STATE:
            return StateConfig(
                use_joint_positions=True,
                use_joint_velocities=True,
                use_joint_torques=True
            )
        else:
            raise ValueError(f"Unknown state type: {self.state_type}")
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        return DataConfig(data_amount=self.data_amount)
    
    def get_action_dim(self) -> int:
        """Get action dimension based on action type"""
        if self.action_type == ActionType.SINGLE_ARM:
            return 32  # 몸+오른팔+오른손
        elif self.action_type == ActionType.BIMANUAL:
            return 42  # 몸+양팔+양손
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")


def generate_all_conditions() -> List[AblationCondition]:
    """Generate all possible ablation conditions"""
    
    models = [ModelType.GR00T, ModelType.PI0, ModelType.UNIVLA]
    data_amounts = [DataAmount.PERCENT_20, DataAmount.PERCENT_100]
    state_types = [StateType.POSITION_ONLY, StateType.FULL_STATE]
    action_types = [ActionType.SINGLE_ARM, ActionType.BIMANUAL]
    camera_types = [CameraType.ROBOT_VIEW, CameraType.MULTI_VIEW]
    
    conditions = []
    
    for model, data, state, action, camera in itertools.product(
        models, data_amounts, state_types, action_types, camera_types
    ):
        condition = AblationCondition(
            model_type=model,
            data_amount=data,
            state_type=state,
            action_type=action,
            camera_type=camera
        )
        conditions.append(condition)
    
    return conditions


# Generate all conditions
ABLATION_CONDITIONS = generate_all_conditions()


def get_condition_by_name(name: str) -> Optional[AblationCondition]:
    """Get ablation condition by name"""
    for condition in ABLATION_CONDITIONS:
        if condition.name == name:
            return condition
    return None


def list_all_conditions() -> List[str]:
    """List all available condition names"""
    return [condition.name for condition in ABLATION_CONDITIONS]


def get_conditions_by_model(model_type: ModelType) -> List[AblationCondition]:
    """Get all conditions for a specific model"""
    return [c for c in ABLATION_CONDITIONS if c.model_type == model_type]


def print_conditions_summary():
    """Print summary of all conditions"""
    print(f"Total conditions: {len(ABLATION_CONDITIONS)}")
    print()
    
    for model in ModelType:
        model_conditions = get_conditions_by_model(model)
        print(f"{model.value.upper()}: {len(model_conditions)} conditions")
        
        for i, condition in enumerate(model_conditions, 1):
            data_pct = "20%" if condition.data_amount == DataAmount.PERCENT_20 else "100%"
            state_desc = "pos_only" if condition.state_type == StateType.POSITION_ONLY else "full_state"
            action_desc = "single" if condition.action_type == ActionType.SINGLE_ARM else "bimanual"
            camera_desc = "robot" if condition.camera_type == CameraType.ROBOT_VIEW else "multi"
            
            print(f"  {i:2d}. {condition.name}")
            print(f"      Data: {data_pct}, State: {state_desc}, Action: {action_desc}, Camera: {camera_desc}")
        print()


if __name__ == "__main__":
    print("VLA Models Ablation Study - All Conditions")
    print("=" * 60)
    print_conditions_summary()
    
    print("Condition breakdown:")
    print(f"- Models: {len(list(ModelType))} (gr00t, pi0, univla)")
    print(f"- Data amounts: {len(list(DataAmount))} (20%, 100%)")
    print(f"- State types: {len(list(StateType))} (pos_only, full_state)")
    print(f"- Action types: {len(list(ActionType))} (single_arm, bimanual)")
    print(f"- Camera types: {len(list(CameraType))} (robot_view, multi_view)")
    print(f"- Total: 3 × 2 × 2 × 2 × 2 = {len(ABLATION_CONDITIONS)} conditions") 