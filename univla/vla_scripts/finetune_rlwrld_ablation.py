#!/usr/bin/env python
"""
Ablation Study를 위한 UniVLA 훈련 스크립트
"""

import os
import argparse
import sys
sys.path.append('/virtual_lab/rlwrld/david/VLA_models_training')
from ablation_config import get_condition_by_name


def main():
    parser = argparse.ArgumentParser(description="UniVLA Ablation Study Training")
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--data-root-dir", type=str, default="./converted_data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    condition = get_condition_by_name(args.condition)
    if not condition:
        print(f"Error: Condition '{args.condition}' not found")
        return
    
    print(f"✅ Training for condition: {args.condition}")
    print(f"Data: {args.data_root_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")


if __name__ == "__main__":
    main()
