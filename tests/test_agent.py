"""
Tests for the PPO agent module.
"""
import torch
import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from agent import PPOAgent


class TestPPOAgent:
    
    def test_initialization(self, sample_config):
        """Test agent initialization with config."""
        agent = PPOAgent(sample_config)
        
        # Check that components are initialized
        assert hasattr(agent, 'network')
        assert hasattr(agent, 'optimizer')
        assert isinstance(agent.optimizer, torch.optim.Adam)
        
        # Check hyperparameters
        ppo_config = sample_config['ppo']
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
    
    def test_select_action(self, sample_config):
        """Test action selection."""
        agent = PPOAgent(sample_config)
        
        # Create sample state
        state = np.array([0.1, 0.2, 0.05, 0.1])
        
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Check return types
        assert isinstance(action, int)
        assert action in [0, 1]  # Should be valid action
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
        # Log prob should be negative (since it's a log of a probability)
        assert log_prob <= 0
    
    def test_store_transition(self, sample_config):
        """Test storing experience transitions."""
        agent = PPOAgent(sample_config)
        
        # Store a transition
        state = np.array([0.1, 0.2, 0.05, 0.1])
        action = 0
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
    
    def test_update_empty_memory(self, sample_config):
        """Test update with empty memory."""
        agent = PPOAgent(sample_config)
        
        # Should handle empty memory gracefully
        agent.update()  # Should not raise error
    
    def test_update_with_transitions(self, sample_config):
        """Test update with stored transitions."""
        agent = PPOAgent(sample_config)
        
        # Store some transitions
        for i in range(10):
            state = np.random.randn(4)
            action = i % 2
            reward = 1.0 if i < 8 else 0.0  # Terminal states get 0 reward
            log_prob = -np.random.rand()
            value = np.random.rand()
            done = (i == 9)
            
            agent.store_transition(state, action, reward, log_prob, value, done)
        
        # Update should work without error
        agent.update()
        
        # Memory should be cleared after update
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
        assert len(agent.rewards) == 0
    
    def test_save_model(self, sample_config, temp_dir):
        """Test model saving functionality."""
        agent = PPOAgent(sample_config)
        
        model_path = os.path.join(temp_dir, 'test_model.pth')
        training_state = {
            'episode': 10,
            'timestep': 1000,
            'reward_history': [1.0, 2.0, 3.0],
            'episode_rewards': [195.0, 200.0, 190.0]
        }
        
        # Save model
        agent.save_model(model_path, training_state)
        
        # Check file exists
        assert os.path.exists(model_path)
    
    def test_load_model_nonexistent(self, sample_config):
        """Test loading non-existent model."""
        agent = PPOAgent(sample_config)
        
        # Should return None for non-existent file
        result = agent.load_model('nonexistent_model.pth')
        assert result is None
    
    def test_load_model_existing(self, sample_config, temp_dir):
        """Test loading existing model."""
        agent1 = PPOAgent(sample_config)
        
        model_path = os.path.join(temp_dir, 'test_model.pth')
        training_state = {
            'episode': 15,
            'timestep': 1500,
            'reward_history': [1.0, 2.0, 3.0],
            'episode_rewards': [195.0, 200.0, 190.0]
        }
        
        # Save model
        agent1.save_model(model_path, training_state)
        
        # Create new agent and load
        agent2 = PPOAgent(sample_config)
        loaded_state = agent2.load_model(model_path)
        
        # Check loaded state
        assert loaded_state is not None
        assert loaded_state['episode'] == 15
        assert loaded_state['timestep'] == 1500
        assert loaded_state['reward_history'] == [1.0, 2.0, 3.0]
    
    def test_action_consistency(self, sample_config):
        """Test that action selection is consistent with same input."""
        agent = PPOAgent(sample_config)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        state = np.array([0.1, 0.2, 0.05, 0.1])
        
        # Get actions multiple times with same seed
        torch.manual_seed(42)
        action1, _, _ = agent.select_action(state)
        
        torch.manual_seed(42)
        action2, _, _ = agent.select_action(state)
        
        # Actions should be the same with same seed
        assert action1 == action2
    
    def test_network_training_mode(self, sample_config):
        """Test that network is in training mode during updates."""
        agent = PPOAgent(sample_config)
        
        # Store some transitions
        for i in range(5):
            state = np.random.randn(4)
            agent.store_transition(state, 0, 1.0, -0.693, 0.5, False)
        
        # Network should be in training mode by default
        assert agent.network.training
        
        # Update (should keep training mode)
        agent.update()
        assert agent.network.training
    
    def test_multiple_episodes_memory(self, sample_config):
        """Test memory handling across multiple episodes."""
        agent = PPOAgent(sample_config)
        
        # Simulate multiple episodes
        for episode in range(3):
            # Add transitions for this episode
            for step in range(5):
                state = np.random.randn(4)
                done = (step == 4)  # Last step of episode
                agent.store_transition(state, 0, 1.0, -0.693, 0.5, done)
        
        # Should have 15 transitions total
        assert len(agent.states) == 15
        
        # Update should clear all memory
        agent.update()
        assert len(agent.states) == 0
