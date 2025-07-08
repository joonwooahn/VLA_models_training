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
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import ablation config
import sys
sys.path.append('/virtual_lab/rlwrld/david/VLA_models_training')
from ablation_config import (
    AblationCondition, StateType, ActionType, CameraType, DataAmount,
    get_condition_by_name
)


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset for UniVLA ablation study")
    parser.add_argument("--condition", type=str, required=True, 
                       help="Ablation condition name")
    parser.add_argument("--input-dir", type=str, 
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_cube",
                       help="Input directory path")
    parser.add_argument("--output-dir", type=str,
                       default="/virtual_lab/rlwrld/david/VLA_models_training/univla/converted_data",
                       help="Output directory path")
    
    args = parser.parse_args()
    
    # 조건 가져오기
    condition = get_condition_by_name(args.condition)
    if not condition:
        print(f"Error: Condition '{args.condition}' not found")
        return
    
    print(f"✅ Data conversion for condition: {args.condition}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Condition: {condition.name}")

    # if condition.state_type == StateType.FULL_STATE:
    #     torque_indices = [i + 60 for i in indices]  # 60-119
    #     velocity_indices = [i + 120 for i in indices]  # 120-179
    #     indices.extend(torque_indices)
    #     indices.extend(velocity_indices)


if __name__ == "__main__":
    main()
