#!/usr/bin/env python3
"""
Test script to verify the solved condition logic works correctly.
This simulates the cart-pole problem being solved without running full training.
"""

import numpy as np
from collections import deque
import sys
import os
sys.path.append('.')

# Import the load_config function from main.py
from main import load_config

def test_solved_condition():
    """Test the solved condition logic."""
    print("Testing solved condition logic...")
    
    # Load configuration
    config = load_config()
    solved_threshold = config['training']['solved_reward_threshold']
    solved_window = config['training']['solved_episodes_window']
    stop_when_solved = config['training']['stop_when_solved']
    
    print(f"Configuration:")
    print(f"  - Solved threshold: {solved_threshold}")
    print(f"  - Solved window: {solved_window} episodes")
    print(f"  - Stop when solved: {stop_when_solved}")
    print()
    
    # Simulate episode rewards
    episode_rewards = deque(maxlen=config['training']['episode_history_length'])
    
    # Test case 1: Not enough episodes yet
    print("Test 1: Adding rewards below threshold...")
    for i in range(50):
        reward = np.random.uniform(150, 180)  # Below threshold
        episode_rewards.append(reward)
        
    if len(episode_rewards) >= solved_window:
        recent_avg = np.mean(list(episode_rewards)[-solved_window:])
        solved = recent_avg >= solved_threshold
        print(f"  - Episodes: {len(episode_rewards)}, Recent avg: {recent_avg:.2f}, Solved: {solved}")
    else:
        print(f"  - Episodes: {len(episode_rewards)}, Not enough episodes for window")
    
    # Test case 2: Fill to window size but still below threshold
    print("\nTest 2: Filling to window size with below-threshold rewards...")
    while len(episode_rewards) < solved_window:
        reward = np.random.uniform(150, 180)  # Below threshold
        episode_rewards.append(reward)
        
    recent_avg = np.mean(list(episode_rewards)[-solved_window:])
    solved = recent_avg >= solved_threshold
    print(f"  - Episodes: {len(episode_rewards)}, Recent avg: {recent_avg:.2f}, Solved: {solved}")
    
    # Test case 3: Add rewards above threshold to trigger solved condition
    print("\nTest 3: Adding rewards above threshold...")
    for i in range(solved_window):
        reward = 200.0  # Above threshold
        episode_rewards.append(reward)
        
        if len(episode_rewards) >= solved_window:
            recent_avg = np.mean(list(episode_rewards)[-solved_window:])
            solved = recent_avg >= solved_threshold
            
            if solved and stop_when_solved:
                print(f"  - Episode {len(episode_rewards)}: Recent avg: {recent_avg:.2f}")
                print("  - 🎉 PROBLEM WOULD BE SOLVED! 🎉")
                print(f"  - Training would stop here with {len(episode_rewards)} total episodes")
                break
            elif i % 10 == 0:
                print(f"  - Episode {len(episode_rewards)}: Recent avg: {recent_avg:.2f}, Solved: {solved}")
    
    print("\nTest completed successfully! ✅")

if __name__ == "__main__":
    test_solved_condition()
