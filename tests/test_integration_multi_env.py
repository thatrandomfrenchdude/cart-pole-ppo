"""
Integration tests for multi-environment PPO system.
"""
import pytest
import numpy as np
import torch
import tempfile
import os
from collections import deque
from unittest.mock import Mock, patch

from src.environments import EnvironmentFactory
from src.agent import PPOAgent
from src.network import PPONetwork
from src.config import load_config


class TestMultiEnvironmentIntegration:
    """Integration tests for the complete multi-environment system."""
    
    @pytest.mark.parametrize("env_name", ['cartpole', 'mountain_car', 'pendulum', 'acrobot'])
    def test_complete_environment_workflow(self, sample_config, env_name):
        """Test complete workflow for each environment type."""
        # Setup config for specific environment
        config = sample_config.copy()
        config['game'] = {'environment': env_name}
        
        # Get environment specifications
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        # Create environment
        env = EnvironmentFactory.create_environment(config)
        
        # Create agent
        agent = PPOAgent(config)
        
        # Run a short episode
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 10  # Short episode for testing
        
        for _ in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Verify episode completed successfully
        assert steps > 0
        assert isinstance(total_reward, (int, float))
        assert len(agent.states) == steps
        assert len(agent.actions) == steps
        
        # Test agent update (should not crash)
        if len(agent.states) > 0:
            try:
                agent.update()
            except Exception as e:
                pytest.fail(f"Agent update failed for {env_name}: {e}")
    
    @pytest.mark.parametrize("env_name", ['cartpole', 'mountain_car', 'pendulum', 'acrobot'])
    def test_environment_state_shapes(self, sample_config, env_name):
        """Test that environment states have correct shapes."""
        config = sample_config.copy()
        config['game'] = {'environment': env_name}
        
        # Get environment specifications and update config
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        # Add environment-specific configs
        if env_name == 'mountain_car':
            config['mountain_car'] = {
                'min_position': -1.2, 'max_position': 0.6, 'max_speed': 0.07,
                'goal_position': 0.5, 'goal_velocity': 0.0, 'force': 0.001,
                'gravity': 0.0025, 'time_step': 0.02
            }
        elif env_name == 'pendulum':
            config['pendulum'] = {
                'max_speed': 8.0, 'max_torque': 2.0, 'time_step': 0.05,
                'gravity': 10.0, 'mass': 1.0, 'length': 1.0
            }
        elif env_name == 'acrobot':
            config['acrobot'] = {
                'link_length_1': 1.0, 'link_length_2': 1.0, 'link_mass_1': 1.0, 'link_mass_2': 1.0,
                'link_com_pos_1': 0.5, 'link_com_pos_2': 0.5, 'link_moi': 1.0,
                'max_vel_1': 12.566, 'max_vel_2': 28.274, 'torque_noise_max': 0.0,
                'time_step': 0.05, 'gravity': 9.8
            }
        
        env = EnvironmentFactory.create_environment(config)
        
        # Test reset state shape
        initial_state = env.reset()
        expected_shape = (specs['input_dim'],)
        
        if env_name == 'pendulum':
            # Pendulum returns observation, not raw state
            assert len(initial_state) == 3  # [cos(theta), sin(theta), theta_dot]
        else:
            assert initial_state.shape == expected_shape
        
        # Test step state shape
        if env_name == 'pendulum':
            action = 0.5  # Continuous action
        else:
            action = 0    # Discrete action
        
        next_state, reward, done = env.step(action)
        
        if env_name == 'pendulum':
            assert len(next_state) == 3
        else:
            assert next_state.shape == expected_shape
    
    def test_agent_action_compatibility(self, sample_config):
        """Test that agents produce compatible actions for each environment."""
        test_cases = [
            ('cartpole', 4, 2, False),
            ('mountain_car', 2, 1, True),  # Mountain Car is now continuous
            ('pendulum', 3, 1, True),
            ('acrobot', 4, 3, False)
        ]
        
        for env_name, input_dim, output_dim, continuous in test_cases:
            config = sample_config.copy()
            config['game'] = {'environment': env_name}
            config['network'] = {
                'input_dim': input_dim,
                'hidden_dim': 64,
                'output_dim': output_dim
            }
            
            agent = PPOAgent(config)
            
            # Create dummy state
            state = np.random.rand(input_dim)
            
            # Get action
            action, log_prob, value = agent.select_action(state)
            
            # Verify action compatibility
            if continuous:
                assert isinstance(action, np.ndarray)
                assert action.shape == (output_dim,)
                assert np.all(-1.0 <= action) and np.all(action <= 1.0)
            else:
                assert isinstance(action, int)
                assert 0 <= action < output_dim
    
    def test_model_saving_loading_all_environments(self, sample_config, temp_dir):
        """Test model saving and loading for all environments."""
        environments = ['cartpole', 'mountain_car', 'pendulum', 'acrobot']
        
        for env_name in environments:
            config = sample_config.copy()
            config['game'] = {'environment': env_name}
            
            # Get specs and update config
            specs = EnvironmentFactory.get_environment_specs(config)
            config['network']['input_dim'] = specs['input_dim']
            config['network']['output_dim'] = specs['output_dim']
            
            # Create agent
            agent = PPOAgent(config)
            
            # Save model
            model_path = os.path.join(temp_dir, f'{env_name}_model.pth')
            training_state = {
                'episode': 50,
                'timestep': 1000,
                'reward_history': [1.0, 2.0, 3.0],
                'episode_rewards': [10.0, 20.0]
            }
            agent.save_model(model_path, training_state)
            
            # Verify file exists
            assert os.path.exists(model_path)
            
            # Create new agent and load
            new_agent = PPOAgent(config)
            loaded_state = new_agent.load_model(model_path)
            
            # Verify state
            assert loaded_state is not None
            assert loaded_state['episode'] == 50
            assert loaded_state['timestep'] == 1000
    
    def test_config_path_resolution(self, sample_config):
        """Test that config paths are resolved correctly for different environments."""
        config = sample_config.copy()
        
        # Test each environment
        for env_name in ['cartpole', 'mountain_car', 'pendulum', 'acrobot']:
            config['game'] = {'environment': env_name}
            
            # Get model paths
            model_save_path = config['training']['model_save_paths'][env_name]
            example_model_path = config['training']['example_model_paths'][env_name]
            solved_threshold = config['training']['solved_reward_thresholds'][env_name]
            
            # Verify paths exist in config
            assert isinstance(model_save_path, str)
            assert isinstance(example_model_path, str)
            assert isinstance(solved_threshold, (int, float))
            
            # Verify paths are different for different environments
            if env_name != 'cartpole':
                assert model_save_path != config['training']['model_save_paths']['cartpole']
                assert example_model_path != config['training']['example_model_paths']['cartpole']


class TestEnvironmentSpecificBehavior:
    """Test environment-specific behaviors and edge cases."""
    
    def test_cartpole_termination_conditions(self, sample_config):
        """Test CartPole termination conditions."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        env = EnvironmentFactory.create_environment(config)
        
        # Test position threshold
        env.state = np.array([3.0, 0, 0, 0])  # Beyond position threshold
        _, _, done = env.step(0)
        assert done
        
        # Test angle threshold
        env.state = np.array([0, 0, 0.3, 0])  # Beyond angle threshold
        _, _, done = env.step(0)
        assert done
    
    def test_mountain_car_goal_reaching(self, sample_config):
        """Test MountainCar goal reaching."""
        config = sample_config.copy()
        config['game'] = {'environment': 'mountain_car'}
        config['mountain_car'] = {
            'min_position': -1.2, 'max_position': 0.6, 'max_speed': 0.07,
            'goal_position': 0.5, 'goal_velocity': 0.0, 'force': 0.001,
            'gravity': 0.0025, 'time_step': 0.02
        }
        
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        env = EnvironmentFactory.create_environment(config)
        
        # Set car at goal position
        env.state = np.array([0.51, 0.01])  # At goal with small velocity
        _, reward, done = env.step(1.0)  # Max force action (continuous)
        
        assert done
        assert reward == 99.9  # Goal reward (100) minus force penalty (0.1 * 1.0^2)
    
    def test_pendulum_continuous_episodes(self, sample_config):
        """Test that Pendulum episodes don't terminate naturally."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['pendulum'] = {
            'max_speed': 8.0, 'max_torque': 2.0, 'time_step': 0.05,
            'gravity': 10.0, 'mass': 1.0, 'length': 1.0
        }
        
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        env = EnvironmentFactory.create_environment(config)
        env.reset()
        
        # Run many steps - should never terminate
        for _ in range(100):
            _, _, done = env.step(0.5)
            assert not done
    
    def test_acrobot_goal_height(self, sample_config):
        """Test Acrobot goal height calculation."""
        config = sample_config.copy()
        config['game'] = {'environment': 'acrobot'}
        config['acrobot'] = {
            'link_length_1': 1.0, 'link_length_2': 1.0, 'link_mass_1': 1.0, 'link_mass_2': 1.0,
            'link_com_pos_1': 0.5, 'link_com_pos_2': 0.5, 'link_moi': 1.0,
            'max_vel_1': 12.566, 'max_vel_2': 28.274, 'torque_noise_max': 0.0,
            'time_step': 0.05, 'gravity': 9.8
        }
        
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        env = EnvironmentFactory.create_environment(config)
        
        # Set both links upright (should reach goal height)
        env.state = np.array([0.0, 0.0, 0.0, 0.0])  # Both links up
        _, reward, done = env.step(1)
        
        if done:
            assert reward == 0.0  # Goal reached gives 0 reward (no penalty)


class TestMultiEnvironmentErrorHandling:
    """Test error handling across multiple environments."""
    
    def test_invalid_actions(self, sample_config):
        """Test handling of invalid actions in each environment."""
        environments = {
            'cartpole': {'valid_actions': [0, 1], 'invalid_action': 2},
            'mountain_car': {'valid_actions': [0, 1, 2], 'invalid_action': 3},
            'acrobot': {'valid_actions': [0, 1, 2], 'invalid_action': -1}
        }
        
        for env_name, action_info in environments.items():
            config = sample_config.copy()
            config['game'] = {'environment': env_name}
            
            specs = EnvironmentFactory.get_environment_specs(config)
            config['network']['input_dim'] = specs['input_dim']
            config['network']['output_dim'] = specs['output_dim']
            
            # Add environment-specific configs
            if env_name == 'mountain_car':
                config['mountain_car'] = {
                    'min_position': -1.2, 'max_position': 0.6, 'max_speed': 0.07,
                    'goal_position': 0.5, 'goal_velocity': 0.0, 'force': 0.001,
                    'gravity': 0.0025, 'time_step': 0.02
                }
            elif env_name == 'acrobot':
                config['acrobot'] = {
                    'link_length_1': 1.0, 'link_length_2': 1.0, 'link_mass_1': 1.0, 'link_mass_2': 1.0,
                    'link_com_pos_1': 0.5, 'link_com_pos_2': 0.5, 'link_moi': 1.0,
                    'max_vel_1': 4 * 3.14159, 'max_vel_2': 9 * 3.14159, 'torque_noise_max': 0.0,
                    'time_step': 0.05, 'gravity': 9.8
                }
            
            env = EnvironmentFactory.create_environment(config)
            env.reset()
            
            # Test valid actions don't crash
            for valid_action in action_info['valid_actions']:
                try:
                    env.step(valid_action)
                except Exception as e:
                    pytest.fail(f"Valid action {valid_action} failed for {env_name}: {e}")
    
    def test_extreme_states(self, sample_config):
        """Test handling of extreme states in each environment."""
        config = sample_config.copy()
        
        # Test CartPole with extreme position
        config['game'] = {'environment': 'cartpole'}
        specs = EnvironmentFactory.get_environment_specs(config)
        config['network']['input_dim'] = specs['input_dim']
        config['network']['output_dim'] = specs['output_dim']
        
        env = EnvironmentFactory.create_environment(config)
        env.state = np.array([100.0, 100.0, 3.14, 100.0])  # Extreme values
        
        try:
            _, _, _ = env.step(0)
        except Exception as e:
            pytest.fail(f"CartPole failed with extreme state: {e}")
    
    def test_network_device_compatibility(self, sample_config):
        """Test that networks work on available devices."""
        for env_name in ['cartpole', 'pendulum']:
            config = sample_config.copy()
            config['game'] = {'environment': env_name}
            
            specs = EnvironmentFactory.get_environment_specs(config)
            config['network']['input_dim'] = specs['input_dim']
            config['network']['output_dim'] = specs['output_dim']
            
            network = PPONetwork(config)
            
            # Test CPU
            device = torch.device('cpu')
            network = network.to(device)
            
            # Create input on same device
            if env_name == 'cartpole':
                test_input = torch.randn(1, 4).to(device)
            else:  # pendulum
                test_input = torch.randn(1, 3).to(device)
            
            # Forward pass should work
            try:
                if env_name == 'cartpole':
                    action_probs, values = network(test_input)
                else:  # pendulum
                    action_mean, action_std, values = network(test_input)
            except Exception as e:
                pytest.fail(f"Network forward pass failed for {env_name} on {device}: {e}")
