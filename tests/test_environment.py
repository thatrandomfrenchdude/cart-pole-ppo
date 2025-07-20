"""
Tests for the CartPole environment module.
"""
import numpy as np
import pytest
import math
from environment import CartPoleEnv


class TestCartPoleEnv:
    
    def test_initialization(self, sample_config):
        """Test environment initialization with config."""
        env = CartPoleEnv(sample_config)
        
        # Check that attributes are set correctly
        assert env.gravity == sample_config['environment']['gravity']
        assert env.masscart == sample_config['environment']['cart_mass']
        assert env.masspole == sample_config['environment']['pole_mass']
        assert env.total_mass == env.masscart + env.masspole
        assert env.length == sample_config['environment']['pole_half_length']
        assert env.force_mag == sample_config['environment']['force_magnitude']
        assert env.tau == sample_config['environment']['time_step']
        
        # Check computed values
        expected_theta_threshold = sample_config['environment']['angle_threshold_degrees'] * 2 * math.pi / 360
        assert abs(env.theta_threshold_radians - expected_theta_threshold) < 1e-10
        assert env.x_threshold == sample_config['environment']['position_threshold']
    
    def test_reset(self, sample_config):
        """Test environment reset functionality."""
        env = CartPoleEnv(sample_config)
        
        # Reset multiple times and check state shape and range
        for _ in range(10):
            state = env.reset()
            assert isinstance(state, np.ndarray)
            assert state.shape == (4,)
            assert np.all(state >= -0.05)
            assert np.all(state <= 0.05)
    
    def test_step_left_action(self, sample_config):
        """Test step with left action (action=0)."""
        env = CartPoleEnv(sample_config)
        initial_state = env.reset()
        
        # Take left action
        next_state, reward, done = env.step(0)
        
        # Check return types and shapes
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (4,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        # State should have changed
        assert not np.array_equal(initial_state, next_state)
    
    def test_step_right_action(self, sample_config):
        """Test step with right action (action=1)."""
        env = CartPoleEnv(sample_config)
        initial_state = env.reset()
        
        # Take right action
        next_state, reward, done = env.step(1)
        
        # Check return types and shapes
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (4,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        # State should have changed
        assert not np.array_equal(initial_state, next_state)
    
    def test_episode_termination_conditions(self, sample_config):
        """Test that episode terminates under proper conditions."""
        env = CartPoleEnv(sample_config)
        
        # Test position boundary termination
        env.state = np.array([env.x_threshold + 0.1, 0, 0, 0])
        _, _, done = env.step(0)
        assert done
        
        # Test negative position boundary
        env.state = np.array([-env.x_threshold - 0.1, 0, 0, 0])
        _, _, done = env.step(0)
        assert done
        
        # Test angle boundary termination
        env.state = np.array([0, 0, env.theta_threshold_radians + 0.1, 0])
        _, _, done = env.step(0)
        assert done
        
        # Test negative angle boundary
        env.state = np.array([0, 0, -env.theta_threshold_radians - 0.1, 0])
        _, _, done = env.step(0)
        assert done
    
    def test_physics_simulation(self, sample_config):
        """Test that physics simulation behaves reasonably."""
        env = CartPoleEnv(sample_config)
        env.reset()
        
        # Start with small angle, should not immediately fail
        env.state = np.array([0, 0, 0.1, 0])  # Small positive angle, zero velocity
        
        # Apply right force (should counteract positive angle)
        next_state, reward, done = env.step(1)
        
        # Should receive reward for not failing
        assert reward == 1.0
        assert not done
        
        # Position and/or velocity should change due to applied force and physics
        # Check that at least one component changed significantly
        state_changed = (
            abs(next_state[0] - 0) > 1e-6 or  # x position changed
            abs(next_state[1] - 0) > 1e-6 or  # x velocity changed  
            abs(next_state[3] - 0) > 1e-6     # angular velocity changed
        )
        assert state_changed, f"Expected state to change but got: {next_state}"
    
    def test_reward_structure(self, sample_config):
        """Test reward structure."""
        env = CartPoleEnv(sample_config)
        env.reset()
        
        # Non-terminal states should give reward of 1
        for _ in range(5):
            _, reward, done = env.step(0)
            if not done:
                assert reward == 1.0
    
    def test_state_bounds_checking(self, sample_config):
        """Test that state bounds are properly checked."""
        env = CartPoleEnv(sample_config)
        env.reset()
        
        # Test edge cases near boundaries
        test_states = [
            [env.x_threshold - 0.01, 0, 0, 0],  # Near position boundary
            [-env.x_threshold + 0.01, 0, 0, 0],  # Near negative position boundary
            [0, 0, env.theta_threshold_radians - 0.01, 0],  # Near angle boundary
            [0, 0, -env.theta_threshold_radians + 0.01, 0],  # Near negative angle boundary
        ]
        
        for state in test_states:
            env.state = np.array(state)
            _, _, done = env.step(0)
            assert not done  # Should not be done yet
    
    def test_multiple_episodes(self, sample_config):
        """Test running multiple episodes."""
        env = CartPoleEnv(sample_config)
        
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 100:  # Prevent infinite loops
                action = np.random.choice([0, 1])
                state, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Should have accumulated some reward
            assert total_reward > 0
            assert steps > 0
