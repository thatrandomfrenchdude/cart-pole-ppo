#!/usr/bin/env python3
"""
Test script to demonstrate the modified pendulum environment behavior.
"""

from src.environments import EnvironmentFactory
import yaml
import numpy as np

def test_pendulum_modifications():
    """Test the pendulum environment modifications."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set environment to pendulum
    config['game']['environment'] = 'pendulum'
    env = EnvironmentFactory.create_environment(config)
    
    print("Pendulum Environment Modifications Test")
    print("=" * 50)
    
    # Test 1: Reset behavior
    print("\n1. Testing Reset Behavior:")
    print("-" * 30)
    for i in range(5):
        state = env.reset()
        theta = env.state[0]
        print(f"Reset {i+1}: θ = {theta:.3f} rad ({theta * 180 / np.pi:.1f}°)")
    
    # Test 2: Reward function
    print("\n2. Testing Reward Function:")
    print("-" * 30)
    angles = np.linspace(-np.pi, np.pi, 13)
    rewards = []
    
    for angle in angles:
        env.state = np.array([angle, 0.0])  # Zero velocity
        _, reward, _ = env.step(0.0)  # No action
        rewards.append(reward)
        
        # Format angle for display
        deg = angle * 180 / np.pi
        position = "hanging down" if abs(abs(angle) - np.pi) < 0.1 else "upright" if abs(angle) < 0.1 else f"{deg:.0f}°"
        print(f"θ = {angle:6.2f} rad ({deg:6.1f}°) -> reward = {reward:6.3f}")
    
    # Test 3: Reward continuity
    print("\n3. Reward Function Properties:")
    print("-" * 30)
    print(f"Maximum reward (upright): {max(rewards):.3f}")
    print(f"Minimum reward (hanging): {min(rewards):.3f}")
    print(f"Reward range: {max(rewards) - min(rewards):.3f}")
    print(f"Continuous function: {'✓' if len(set([round(r, 6) for r in rewards])) == len(rewards) else '✗'}")
    
    # Test 4: Visual representation of reward function
    print("\n4. Visual Representation of Reward Function:")
    print("-" * 50)
    print("Angle (deg) | Reward | Visual")
    print("-" * 50)
    
    for angle in np.linspace(-180, 180, 9):
        angle_rad = angle * np.pi / 180
        reward = np.cos(abs(angle_rad))
        # Create a simple ASCII bar chart
        bar_length = int((reward + 1) * 15)  # Scale to 0-30 characters
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{angle:8.0f}° | {reward:6.3f} | {bar}")
    
    return angles, rewards


if __name__ == "__main__":
    test_pendulum_modifications()
