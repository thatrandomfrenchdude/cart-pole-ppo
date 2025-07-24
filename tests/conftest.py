"""
Test configuration and fixtures for the cart-pole PPO project.
"""
import pytest
import os
import sys
import tempfile
import shutil
from collections import deque
from unittest.mock import MagicMock

# Add src to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'game': {
            'environment': 'cartpole'
        },
        'environment': {
            'gravity': 9.8,
            'cart_mass': 1.0,
            'pole_mass': 0.1,
            'pole_half_length': 0.5,
            'force_magnitude': 10.0,
            'time_step': 0.02,
            'position_threshold': 2.4,
            'angle_threshold_degrees': 12
        },
        'mountain_car': {
            'min_position': -1.2,
            'max_position': 0.6,
            'max_speed': 0.07,
            'goal_position': 0.5,
            'goal_velocity': 0.0,
            'force': 0.001,
            'gravity': 0.0025,
            'time_step': 0.02
        },
        'pendulum': {
            'max_speed': 8.0,
            'max_torque': 2.0,
            'time_step': 0.05,
            'gravity': 10.0,
            'mass': 1.0,
            'length': 1.0
        },
        'acrobot': {
            'link_length_1': 1.0,
            'link_length_2': 1.0,
            'link_mass_1': 1.0,
            'link_mass_2': 1.0,
            'link_com_pos_1': 0.5,
            'link_com_pos_2': 0.5,
            'link_moi': 1.0,
            'max_vel_1': 12.566,
            'max_vel_2': 28.274,
            'torque_noise_max': 0.0,
            'time_step': 0.05,
            'gravity': 9.8
        },
        'network': {
            'input_dim': 4,
            'hidden_dim': 128,
            'output_dim': 2
        },
        'ppo': {
            'learning_rate': 0.0003,
            'discount_factor': 0.99,
            'clip_ratio': 0.2,
            'update_epochs': 4,
            'update_frequency': 200
        },
        'training': {
            'simulation_speed': 0.01,  # Fast for tests
            'reward_history_length': 100,
            'episode_history_length': 50,
            'model_save_paths': {
                'cartpole': 'test_cartpole_model.pth',
                'mountain_car': 'test_mountain_car_model.pth',
                'pendulum': 'test_pendulum_model.pth',
                'acrobot': 'test_acrobot_model.pth'
            },
            'save_frequency': 10,
            'example_mode': False,
            'example_model_paths': {
                'cartpole': 'test_example_cartpole.pth',
                'mountain_car': 'test_example_mountain_car.pth',
                'pendulum': 'test_example_pendulum.pth',
                'acrobot': 'test_example_acrobot.pth'
            },
            'solved_reward_thresholds': {
                'cartpole': 195.0,
                'mountain_car': -110.0,
                'pendulum': -200.0,
                'acrobot': -100.0
            },
            'solved_episodes_window': 100,
            'stop_when_solved': False  # Don't stop during tests
        },
        'server': {
            'host': '127.0.0.1',
            'port': 8081,
            'debug': False
        },
        'logging': {
            'level': 'WARNING',  # Reduce noise in tests
            'format': '%(asctime)s - %(message)s',
            'episode_summary_frequency': 5,
            'log_file': 'test_training.log'
        }
    }

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_shared_state():
    """Mock shared state objects for training tests."""
    current_state = {
        "position": 0, "velocity": 0, "angle": 0, 
        "angular_velocity": 0, "reward": 0, "episode": 0, "timestep": 0
    }
    reward_history = deque(maxlen=100)
    episode_rewards = deque(maxlen=50)
    running_flag = {'value': True}
    
    return current_state, reward_history, episode_rewards, running_flag
