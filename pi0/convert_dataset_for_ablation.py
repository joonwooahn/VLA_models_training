#!/usr/bin/env python
"""
PI0/PI0_FAST Dataset Conversion for VLA Models Ablation Study
Converts LeRobot format dataset to ablation-specific datasets with proper state/action filtering.
Primarily designed for PI0/PI0_FAST models which use simple dimension cutting.
GR00T and UniVLA already have proper joint filtering implemented.
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
import shutil
import random
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import ablation utilities from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from run_ablation import (
    AblationCondition, StateType, ActionType, CameraType,
    get_unified_state_indices, get_unified_action_indices,
    get_condition_by_name, generate_all_conditions
)


def convert_dataset_for_condition(input_dir: str, output_base_dir: str, condition: AblationCondition, keep_percentage: int = 100):
    """Convert LeRobot dataset for a specific ablation condition"""
    
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)
    
    # Create shared output directory name (excluding model name)
    # Format: {data_amount}_percent_{state_type}_{action_type}_{camera_type}
    shared_condition_name = f"{condition.data_amount}_percent_{condition.state_type.value}_{condition.action_type.value}_{condition.camera_type.value}"
    output_path = output_base_path / shared_condition_name
    
    # Check if conversion already exists and is complete
    if output_path.exists():
        print(f"🔍 Found existing converted data at: {output_path}")
        # Quick check for completion
        data_dir = output_path / "data"
        if data_dir.exists():
            existing_episodes = list(data_dir.glob("episode_*.parquet"))
            if existing_episodes:
                print(f"📂 {len(existing_episodes)} episodes already converted")
                print(f"✅ Skipping conversion (data already exists)")
                print(f"💡 This shared data can be used by both PI0 and PI0_FAST models")
                return str(output_path)
    
    print(f"🔄 Converting LeRobot dataset for shared PI0/PI0_FAST usage...")
    print(f"   - Input: {input_path}")
    print(f"   - Output: {output_path}")
    print(f"   - Shared condition: {shared_condition_name}")
    print(f"   - State type: {condition.state_type.value}")
    print(f"   - Action type: {condition.action_type.value}")
    print(f"   - Camera type: {condition.camera_type.value}")
    print(f"   - Data percentage: {condition.data_amount}%")
    print(f"💡 This data will be shared between PI0 and PI0_FAST models")
    
    # Get unified indices for consistent filtering across all models
    state_indices = get_unified_state_indices(condition.state_type, condition.action_type)
    action_indices = get_unified_action_indices(condition.action_type)
    
    state_dim = len(state_indices)
    action_dim = len(action_indices)
    
    print(f"   - State indices: {len(state_indices)} dims - {state_indices[:10]}...{state_indices[-5:] if len(state_indices) > 15 else state_indices}")
    print(f"   - Action indices: {len(action_indices)} dims - {action_indices}")
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    # Read original data
    data_dir = input_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all episode files - handle both old and new LeRobot formats
    episode_files = []
    
    # Try old format first (data/*.parquet)
    episode_files = sorted(list(data_dir.glob("episode_*.parquet")))
    
    # If no files found, try new chunk format (data/chunk-*/*.parquet)
    if not episode_files:
        chunk_dirs = sorted(list(data_dir.glob("chunk-*")))
        for chunk_dir in chunk_dirs:
            if chunk_dir.is_dir():
                chunk_episodes = sorted(list(chunk_dir.glob("episode_*.parquet")))
                episode_files.extend(chunk_episodes)
    
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {data_dir} or its chunk subdirectories")
    
    print(f"📊 Found {len(episode_files)} episodes in source dataset")
    
    # Apply data percentage filtering
    if condition.data_amount < 100:
        num_episodes_to_use = max(1, int(len(episode_files) * condition.data_amount / 100))
        episode_files = episode_files[:num_episodes_to_use]  # Use first N episodes
        print(f"📉 Using {len(episode_files)} episodes ({condition.data_amount}% of total)")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    output_data_dir = output_path / "data"
    output_data_dir.mkdir(exist_ok=True)
    
    # Copy meta directory
    print("📂 Copying meta directory...")
    input_meta_dir = input_path / "meta"
    output_meta_dir = output_path / "meta"
    if input_meta_dir.exists():
        if output_meta_dir.exists():
            shutil.rmtree(output_meta_dir)
        shutil.copytree(input_meta_dir, output_meta_dir)
        print(f"✅ Meta directory copied to: {output_meta_dir}")
    else:
        print(f"⚠️  Meta directory not found at: {input_meta_dir}")
    
    # Copy videos directory
    print("🎥 Copying videos directory...")
    input_videos_dir = input_path / "videos"
    output_videos_dir = output_path / "videos"
    if input_videos_dir.exists():
        if output_videos_dir.exists():
            shutil.rmtree(output_videos_dir)
        shutil.copytree(input_videos_dir, output_videos_dir)
        print(f"✅ Videos directory copied to: {output_videos_dir}")
    else:
        print(f"⚠️  Videos directory not found at: {input_videos_dir}")
    
    # Copy and modify info.json
    original_info_file = input_path / "info.json"
    if original_info_file.exists():
        with open(original_info_file, 'r') as f:
            info_data = json.load(f)
        
        # Update info for the converted dataset
        info_data['total_episodes'] = len(episode_files)
        info_data['state_dim'] = state_dim
        info_data['action_dim'] = action_dim
        info_data['ablation_condition'] = shared_condition_name
        info_data['shared_for_models'] = ['pi0', 'pi0_fast']
        info_data['original_condition_example'] = condition.name
        info_data['state_indices'] = state_indices
        info_data['action_indices'] = action_indices
        
        # Update shapes in features
        if 'features' in info_data:
            if 'observation.state' in info_data['features']:
                info_data['features']['observation.state']['shape'] = [state_dim]
            if 'action' in info_data['features']:
                info_data['features']['action']['shape'] = [action_dim]
        
        with open(output_path / "info.json", 'w') as f:
            json.dump(info_data, f, indent=2)
        print(f"✅ Info file updated at: {output_path / 'info.json'}")
    else:
        print(f"⚠️  Info file not found at: {original_info_file}")
    
    # Copy any other files in the root directory (e.g., README, etc.)
    for item in input_path.iterdir():
        if item.is_file() and item.name not in ['info.json']:
            shutil.copy2(item, output_path / item.name)
            print(f"📄 Copied additional file: {item.name}")
    
    # Convert each episode
    print(f"🔄 Converting {len(episode_files)} episodes...")
    
    for episode_file in tqdm(episode_files, desc="Converting episodes"):
        convert_episode(episode_file, output_data_dir, state_indices, action_indices)
    
    print(f"✅ Dataset conversion completed!")
    print(f"📁 Converted dataset saved to: {output_path}")
    print(f"📊 Episodes: {len(episode_files)}, State dim: {state_dim}, Action dim: {action_dim}")
    print(f"📂 Structure: data/ ({len(episode_files)} episodes), meta/, videos/, info.json")
    
    return str(output_path)


def convert_episode(episode_file: Path, output_data_dir: Path, state_indices: List[int], action_indices: List[int]):
    """Convert a single episode with proper state/action filtering"""
    
    # Read original episode data
    df = pd.read_parquet(episode_file)
    
    if df.empty:
        print(f"⚠️  Empty episode file: {episode_file.name}, skipping")
        return
    
    # Apply state filtering
    observation_state_filtered = []
    for _, row in df.iterrows():
        full_state = np.array(row['observation.state'])
        filtered_state = full_state[state_indices]
        observation_state_filtered.append(filtered_state)
    
    # Apply action filtering  
    action_filtered = []
    for _, row in df.iterrows():
        full_action = np.array(row['action'])
        filtered_action = full_action[action_indices]
        action_filtered.append(filtered_action)
    
    # Create new dataframe with filtered data
    new_df = df.copy()
    new_df['observation.state'] = observation_state_filtered
    new_df['action'] = action_filtered
    
    # Save filtered episode
    output_file = output_data_dir / episode_file.name
    new_df.to_parquet(output_file, index=False)


def convert_all_conditions(input_dir: str, output_base_dir: str, models: Optional[List[str]] = None):
    """Convert dataset for ablation conditions (primarily for PI0/PI0_FAST)"""
    
    print("🚀 Starting dataset conversion for ablation conditions")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output base: {output_base_dir}")
    
    # Default to PI0/PI0_FAST if no models specified
    if models is None:
        models = ['pi0', 'pi0_fast']
        print(f"🔍 Using default models for conversion: {models}")
    
    # Generate all conditions
    all_conditions = generate_all_conditions()
    
    # Filter by models
    all_conditions = [c for c in all_conditions if c.model_type.value in models]
    print(f"🔍 Filtering for models: {models}")
    
    # Create unique combinations excluding model name to avoid duplicates
    # Since PI0 and PI0_FAST use the same converted data for identical state/action/camera/data conditions
    unique_combinations = set()
    unique_conditions = []
    
    for condition in all_conditions:
        # Create unique key excluding model name
        unique_key = (condition.data_amount, condition.state_type.value, condition.action_type.value, condition.camera_type.value)
        
        if unique_key not in unique_combinations:
            unique_combinations.add(unique_key)
            unique_conditions.append(condition)
    
    print(f"📊 Total unique combinations to convert: {len(unique_conditions)} (reduced from {len(all_conditions)} to avoid duplicates)")
    print(f"💡 Each converted dataset will be shared between PI0 and PI0_FAST models")
    
    converted_datasets = {}
    
    for i, condition in enumerate(unique_conditions, 1):
        # Create shared condition name for display
        shared_name = f"{condition.data_amount}_percent_{condition.state_type.value}_{condition.action_type.value}_{condition.camera_type.value}"
        
        print(f"\n{'='*80}")
        print(f"Converting combination {i}/{len(unique_conditions)}: {shared_name}")
        print(f"Original condition example: {condition.name}")
        print(f"{'='*80}")
        
        try:
            dataset_path = convert_dataset_for_condition(input_dir, output_base_dir, condition)
            converted_datasets[shared_name] = dataset_path
            print(f"✅ Combination {i} completed: {shared_name}")
            
        except Exception as e:
            print(f"❌ Failed to convert combination {shared_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Dataset conversion completed!")
    print(f"✅ Successfully converted: {len(converted_datasets)}/{len(unique_conditions)} unique combinations")
    print(f"📁 Converted datasets available at: {output_base_dir}")
    print(f"💡 Each dataset can be used by both PI0 and PI0_FAST models")
    
    # Save conversion summary
    summary = {
        'total_unique_combinations': len(unique_conditions),
        'total_original_conditions': len(all_conditions),
        'successful_conversions': len(converted_datasets),
        'converted_datasets': converted_datasets,
        'input_dataset': input_dir,
        'output_base_dir': output_base_dir,
        'target_models': models,
        'shared_data_info': 'Each converted dataset can be used by both PI0 and PI0_FAST models'
    }
    
    summary_file = Path(output_base_dir) / "conversion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Conversion summary saved to: {summary_file}")
    
    return converted_datasets


def list_converted_datasets(output_base_dir: str):
    """List all converted datasets"""
    
    base_path = Path(output_base_dir)
    if not base_path.exists():
        print(f"❌ Output directory does not exist: {output_base_dir}")
        return
    
    converted_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d / "data").exists()]
    
    print(f"📊 Found {len(converted_dirs)} converted datasets in {output_base_dir}:")
    
    for dataset_dir in sorted(converted_dirs):
        info_file = dataset_dir / "info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            episodes = info.get('total_episodes', 'unknown')
            state_dim = info.get('state_dim', 'unknown') 
            action_dim = info.get('action_dim', 'unknown')
            
            print(f"  📁 {dataset_dir.name}")
            print(f"     Episodes: {episodes}, State: {state_dim}D, Action: {action_dim}D")
        else:
            print(f"  📁 {dataset_dir.name} (info missing)")


def main():
    parser = argparse.ArgumentParser(description="Dataset conversion for PI0/PI0_FAST models in VLA ablation study")
    parser.add_argument("--input-dir", type=str, 
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq",
                       help="Input LeRobot dataset directory")
    parser.add_argument("--output-dir", type=str,
                       default="../converted_datasets_ablation",
                       help="Output directory for converted datasets (relative to parent dir)")
    parser.add_argument("--condition", type=str,
                       help="Convert only specific condition (e.g., 'pi0_20_percent_pos_only_right_arm_robot_view')")
    parser.add_argument("--models", type=str, nargs='+',
                       choices=['gr00t', 'pi0', 'pi0_fast', 'univla'],
                       default=['pi0', 'pi0_fast'],
                       help="Convert only for specific models (default: pi0 pi0_fast)")
    parser.add_argument("--list", action="store_true",
                       help="List existing converted datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_converted_datasets(args.output_dir)
        return
    
    print("🔧 PI0/PI0_FAST Dataset Conversion Tool")
    print("=" * 60)
    print("💡 Note: GR00T and UniVLA already have proper joint filtering")
    print("💡 This tool is primarily for PI0/PI0_FAST which use simple dimension cutting")
    print()
    
    if args.condition:
        # Convert single condition
        condition = get_condition_by_name(args.condition)
        if not condition:
            print(f"❌ Error: Condition '{args.condition}' not found")
            return
        
        # Check if this is really needed for the model
        if condition.model_type.value in ['gr00t', 'univla']:
            print(f"⚠️  Note: {condition.model_type.value.upper()} already has proper joint filtering")
            print(f"⚠️  Consider using the model's existing approach instead")
            
        print(f"🎯 Converting single condition: {args.condition}")
        dataset_path = convert_dataset_for_condition(args.input_dir, args.output_dir, condition)
        print(f"🎉 Single condition conversion completed!")
        print(f"📁 Dataset available at: {dataset_path}")
        
    else:
        # Convert all conditions (or filtered by models)
        if args.models != ['pi0', 'pi0_fast']:
            print(f"🔍 Converting for models: {args.models}")
            if any(model in ['gr00t', 'univla'] for model in args.models):
                print(f"⚠️  Note: GR00T and UniVLA already have proper joint filtering implemented")
                print(f"⚠️  You may want to use --models pi0 pi0_fast instead")
        
        convert_all_conditions(args.input_dir, args.output_dir, args.models)


if __name__ == "__main__":
    main() 