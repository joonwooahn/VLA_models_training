#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

def create_simple_stats():
    """Create a simple stats.json with proper dimensions for gr00t."""
    
    stats_file = Path("/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq/meta/stats.json")
    episodes_stats_file = Path("/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/allex_gesture_easy_pos_vel_torq/meta/episodes_stats.jsonl")
    
    print("Reading first episode to get data structure...")
    with open(episodes_stats_file, 'r') as f:
        first_episode = json.loads(f.readline())
        
    obs_state_stats = first_episode['stats']['observation.state']
    action_stats = first_episode['stats']['action']
    
    # Create stats with proper dimensions
    stats = {
        "observation.state": {
            "mean": obs_state_stats['mean'],  # Should be 180-dim
            "std": obs_state_stats['std'],
            "min": obs_state_stats['min'],
            "max": obs_state_stats['max'],
            "q01": obs_state_stats['min'],  # Use min as q01 approximation
            "q99": obs_state_stats['max']   # Use max as q99 approximation
        },
        "action": {
            "mean": action_stats['mean'],  # Should be 42-dim
            "std": action_stats['std'],
            "min": action_stats['min'],
            "max": action_stats['max'],
            "q01": action_stats['min'],  # Use min as q01 approximation
            "q99": action_stats['max']   # Use max as q99 approximation
        }
    }
    
    # Verify dimensions
    print(f"observation.state dimensions: {len(stats['observation.state']['mean'])}")
    print(f"action dimensions: {len(stats['action']['mean'])}")
    
    # Write to file
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Created simple stats.json at: {stats_file}")
    return stats

if __name__ == "__main__":
    create_simple_stats() 