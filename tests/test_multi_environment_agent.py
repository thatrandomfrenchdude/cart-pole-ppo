"""
Tests for the multi-environment PPO agent module.
"""
import torch
import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.agent import PPOAgent


class TestPPOAgentDiscrete:
    """Test PPO agent with discrete action environments."""
    
    def test_cartpole_initialization(self, sample_config):
        """Test agent initialization for CartPole."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        agent = PPOAgent(config)
        
        # Check that components are initialized
        assert hasattr(agent, 'network')
        assert hasattr(agent, 'optimizer')
        assert isinstance(agent.optimizer, torch.optim.Adam)
        assert agent.continuous_action is False
        
        # Check hyperparameters
        ppo_config = config['ppo']
        assert agent.gamma == ppo_config['discount_factor']
        assert agent.eps_clip == ppo_config['clip_ratio']
        assert agent.k_epochs == ppo_config['update_epochs']
        
        # Check memory arrays are initialized empty
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.rewards) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.values) == 0
        assert len(agent.dones) == 0
    
    def test_mountain_car_select_action(self, sample_config):
        """Test action selection for MountainCar."""
        config = sample_config.copy()
        config['game'] = {'environment': 'mountain_car'}
        config['network'] = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 3}
        agent = PPOAgent(config)
        
        # Create sample state
        state = np.array([0.1, 0.05])
        
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Check return types for discrete action
        assert isinstance(action, int)
        assert action in [0, 1, 2]  # Should be valid action for MountainCar
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_acrobot_select_action(self, sample_config):
        """Test action selection for Acrobot."""
        config = sample_config.copy()
        config['game'] = {'environment': 'acrobot'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 3}
        agent = PPOAgent(config)
        
        # Create sample state
        state = np.array([0.1, 0.2, 0.05, 0.1])
        
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Check return types for discrete action
        assert isinstance(action, int)
        assert action in [0, 1, 2]  # Should be valid action for Acrobot
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestPPOAgentContinuous:
    """Test PPO agent with continuous action environments."""
    
    def test_pendulum_initialization(self, sample_config):
        """Test agent initialization for Pendulum."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Check continuous action flag
        assert agent.continuous_action is True
        
        # Check network has proper continuous action components
        assert hasattr(agent.network, 'actor_mean')
        assert hasattr(agent.network, 'actor_log_std')
    
    def test_pendulum_select_action(self, sample_config):
        """Test action selection for Pendulum."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Create sample state
        state = np.array([0.8, 0.6, 1.5])  # [cos(theta), sin(theta), theta_dot]
        
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Check return types for continuous action
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)  # Should be 1D array with one element
        assert -1.0 <= action[0] <= 1.0  # Should be clipped to valid range
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_continuous_action_clipping(self, sample_config):
        """Test that continuous actions are properly clipped."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Set network parameters to produce extreme actions for testing
        with torch.no_grad():
            agent.network.actor_mean.weight.fill_(10.0)  # Large weights
            agent.network.actor_mean.bias.fill_(5.0)     # Large bias
        
        state = np.array([1.0, 0.0, 0.0])
        action, _, _ = agent.select_action(state)
        
        # Action should still be clipped
        assert -1.0 <= action[0] <= 1.0


class TestPPOAgentTraining:
    """Test PPO agent training functionality."""
    
    def test_store_transition_discrete(self, sample_config):
        """Test storing transitions for discrete action environments."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        agent = PPOAgent(config)
        
        # Store a transition
        state = np.array([0.1, 0.2, 0.05, 0.1])
        action = 1
        reward = 1.0
        log_prob = -0.693
        value = 0.5
        done = False
        
        agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Check that transition was stored
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.rewards) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.values) == 1
        assert len(agent.dones) == 1
        
        # Check stored values
        assert np.array_equal(agent.states[0], state)
        assert agent.actions[0] == action
        assert agent.rewards[0] == reward
        assert agent.log_probs[0] == log_prob
        assert agent.values[0] == value
        assert agent.dones[0] == done
    
    def test_store_transition_continuous(self, sample_config):
        """Test storing transitions for continuous action environments."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Store a transition
        state = np.array([0.8, 0.6, 1.5])
        action = np.array([0.3])
        reward = -2.5
        log_prob = -1.2
        value = 0.1
        done = False
        
        agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Check that transition was stored
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert np.array_equal(agent.actions[0], action)
    
    def test_update_discrete_actions(self, sample_config):
        """Test PPO update with discrete actions."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        agent = PPOAgent(config)
        
        # Store some transitions
        for i in range(10):
            state = np.random.rand(4)
            action = np.random.randint(0, 2)
            reward = np.random.rand()
            log_prob = -np.random.rand()
            value = np.random.rand()
            done = i == 9  # Last one is done
            
            agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Update should not raise an error
        try:
            agent.update()
        except Exception as e:
            pytest.fail(f"Update failed with discrete actions: {e}")
        
        # Memory should be cleared after update
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
    
    def test_update_continuous_actions(self, sample_config):
        """Test PPO update with continuous actions."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Store some transitions
        for i in range(10):
            state = np.random.rand(3)
            action = np.random.rand(1) * 2 - 1  # Random action in [-1, 1]
            reward = -np.random.rand() * 10  # Negative rewards typical for Pendulum
            log_prob = -np.random.rand()
            value = np.random.rand()
            done = False  # Pendulum episodes don't naturally terminate
            
            agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Update should not raise an error
        try:
            agent.update()
        except Exception as e:
            pytest.fail(f"Update failed with continuous actions: {e}")
        
        # Memory should be cleared after update
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
    
    def test_clear_memory(self, sample_config):
        """Test clearing memory."""
        agent = PPOAgent(sample_config)
        
        # Add some data
        agent.states = [1, 2, 3]
        agent.actions = [1, 2, 3]
        agent.rewards = [1, 2, 3]
        agent.log_probs = [1, 2, 3]
        agent.values = [1, 2, 3]
        agent.dones = [True, False, True]
        
        # Clear memory
        agent.clear_memory()
        
        # Check all lists are empty
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.rewards) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.values) == 0
        assert len(agent.dones) == 0


class TestPPOAgentModelSaving:
    """Test model saving and loading functionality."""
    
    def test_save_and_load_model_discrete(self, sample_config, temp_dir):
        """Test saving and loading model for discrete action environments."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        agent = PPOAgent(config)
        
        # Create temp file path
        model_path = os.path.join(temp_dir, 'test_discrete_model.pth')
        
        # Save model
        training_state = {
            'episode': 100,
            'timestep': 5000,
            'reward_history': [1.0, 2.0, 3.0],
            'episode_rewards': [10.0, 20.0, 30.0]
        }
        agent.save_model(model_path, training_state)
        
        # Verify file exists
        assert os.path.exists(model_path)
        
        # Create new agent and load model
        new_agent = PPOAgent(config)
        loaded_state = new_agent.load_model(model_path)
        
        # Verify training state was loaded
        assert loaded_state['episode'] == 100
        assert loaded_state['timestep'] == 5000
        assert loaded_state['reward_history'] == [1.0, 2.0, 3.0]
        assert loaded_state['episode_rewards'] == [10.0, 20.0, 30.0]
    
    def test_save_and_load_model_continuous(self, sample_config, temp_dir):
        """Test saving and loading model for continuous action environments."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        agent = PPOAgent(config)
        
        # Create temp file path
        model_path = os.path.join(temp_dir, 'test_continuous_model.pth')
        
        # Save model
        agent.save_model(model_path)
        
        # Verify file exists
        assert os.path.exists(model_path)
        
        # Create new agent and load model
        new_agent = PPOAgent(config)
        loaded_state = new_agent.load_model(model_path)
        
        # Should not fail for continuous action agent
        assert loaded_state is not None
    
    def test_load_nonexistent_model(self, sample_config):
        """Test loading a nonexistent model."""
        agent = PPOAgent(sample_config)
        
        # Try to load nonexistent model
        loaded_state = agent.load_model('nonexistent_model.pth')
        
        # Should return None for nonexistent model
        assert loaded_state is None


class TestPPOAgentEnvironmentCompatibility:
    """Test agent compatibility with different environments."""
    
    @pytest.mark.parametrize("env_name,input_dim,output_dim,continuous", [
        ('cartpole', 4, 2, False),
        ('mountain_car', 2, 3, False),
        ('pendulum', 3, 1, True),
        ('acrobot', 4, 3, False),
    ])
    def test_environment_compatibility(self, sample_config, env_name, input_dim, output_dim, continuous):
        """Test agent works with all supported environments."""
        config = sample_config.copy()
        config['game'] = {'environment': env_name}
        config['network'] = {
            'input_dim': input_dim,
            'hidden_dim': 64,
            'output_dim': output_dim
        }
        
        # Create agent
        agent = PPOAgent(config)
        
        # Check continuous action flag
        assert agent.continuous_action == continuous
        
        # Test action selection
        state = np.random.rand(input_dim)
        action, log_prob, value = agent.select_action(state)
        
        # Verify action format
        if continuous:
            assert isinstance(action, np.ndarray)
            assert action.shape == (output_dim,)
            assert np.all(-1.0 <= action) and np.all(action <= 1.0)
        else:
            assert isinstance(action, int)
            assert 0 <= action < output_dim
        
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
