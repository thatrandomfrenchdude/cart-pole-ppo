import pytest
import numpy as np
import math
from src.environments import (
    EnvironmentFactory, 
    CartPoleEnv, 
    MountainCarEnv, 
    PendulumEnv, 
    AcrobotEnv
)


class TestEnvironmentFactory:
    """Test the environment factory class."""
    
    def test_create_cartpole_environment(self, sample_config):
        """Test creating CartPole environment."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        
        env = EnvironmentFactory.create_environment(config)
        assert isinstance(env, CartPoleEnv)
    
    def test_create_mountain_car_environment(self, sample_config):
        """Test creating MountainCar environment."""
        config = sample_config.copy()
        config['game'] = {'environment': 'mountain_car'}
        config['mountain_car'] = {
            'min_position': -1.2,
            'max_position': 0.6,
            'max_speed': 0.07,
            'goal_position': 0.5,
            'goal_velocity': 0.0,
            'force': 0.001,
            'gravity': 0.0025,
            'time_step': 0.02
        }
        
        env = EnvironmentFactory.create_environment(config)
        assert isinstance(env, MountainCarEnv)
    
    def test_create_pendulum_environment(self, sample_config):
        """Test creating Pendulum environment."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['pendulum'] = {
            'max_speed': 8.0,
            'max_torque': 2.0,
            'time_step': 0.05,
            'gravity': 10.0,
            'mass': 1.0,
            'length': 1.0
        }
        
        env = EnvironmentFactory.create_environment(config)
        assert isinstance(env, PendulumEnv)
    
    def test_create_acrobot_environment(self, sample_config):
        """Test creating Acrobot environment."""
        config = sample_config.copy()
        config['game'] = {'environment': 'acrobot'}
        config['acrobot'] = {
            'link_length_1': 1.0,
            'link_length_2': 1.0,
            'link_mass_1': 1.0,
            'link_mass_2': 1.0,
            'link_com_pos_1': 0.5,
            'link_com_pos_2': 0.5,
            'link_moi': 1.0,
            'max_vel_1': 4 * 3.14159,
            'max_vel_2': 9 * 3.14159,
            'torque_noise_max': 0.0,
            'time_step': 0.05,
            'gravity': 9.8
        }
        
        env = EnvironmentFactory.create_environment(config)
        assert isinstance(env, AcrobotEnv)
    
    def test_unknown_environment_raises_error(self, sample_config):
        """Test that unknown environment raises ValueError."""
        config = sample_config.copy()
        config['game'] = {'environment': 'unknown'}
        
        with pytest.raises(ValueError, match="Unknown environment: unknown"):
            EnvironmentFactory.create_environment(config)
    
    def test_get_environment_specs(self, sample_config):
        """Test getting environment specifications."""
        # CartPole
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        specs = EnvironmentFactory.get_environment_specs(config)
        assert specs == {'input_dim': 4, 'output_dim': 2}
        
        # MountainCar (now continuous)
        config['game'] = {'environment': 'mountain_car'}
        specs = EnvironmentFactory.get_environment_specs(config)
        assert specs == {'input_dim': 2, 'output_dim': 1}
        
        # Pendulum
        config['game'] = {'environment': 'pendulum'}
        specs = EnvironmentFactory.get_environment_specs(config)
        assert specs == {'input_dim': 3, 'output_dim': 1}
        
        # Acrobot
        config['game'] = {'environment': 'acrobot'}
        specs = EnvironmentFactory.get_environment_specs(config)
        assert specs == {'input_dim': 4, 'output_dim': 3}


class TestMountainCarEnv:
    """Test the MountainCar environment."""
    
    @pytest.fixture
    def mountain_car_config(self):
        """Sample configuration for MountainCar."""
        return {
            'mountain_car': {
                'min_position': -1.2,
                'max_position': 0.6,
                'max_speed': 0.07,
                'goal_position': 0.5,
                'goal_velocity': 0.0,
                'force': 0.001,
                'gravity': 0.0025,
                'time_step': 0.02
            }
        }
    
    def test_initialization(self, mountain_car_config):
        """Test MountainCar environment initialization."""
        env = MountainCarEnv(mountain_car_config)
        
        assert env.min_position == -1.2
        assert env.max_position == 0.6
        assert env.max_speed == 0.07
        assert env.goal_position == 0.5
        assert env.goal_velocity == 0.0
        assert env.force == 0.001
        assert env.gravity == 0.0025
        assert env.tau == 0.02
    
    def test_reset(self, mountain_car_config):
        """Test resetting the environment."""
        env = MountainCarEnv(mountain_car_config)
        state = env.reset()
        
        assert len(state) == 2
        assert -0.6 <= state[0] <= -0.4  # Position in valley
        assert state[1] == 0.0  # Zero velocity
    
    def test_step_actions(self, mountain_car_config):
        """Test stepping with different actions."""
        env = MountainCarEnv(mountain_car_config)
        env.reset()
        
        # Test continuous actions
        for action in [-1.0, 0.0, 1.0]:  # left, none, right
            initial_state = env.state.copy()
            next_state, reward, done = env.step(action)
            
            assert len(next_state) == 2
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            
            # State should change
            assert not np.array_equal(initial_state, next_state)
    
    def test_boundary_conditions(self, mountain_car_config):
        """Test boundary conditions."""
        env = MountainCarEnv(mountain_car_config)
        
        # Test left boundary
        env.state = np.array([env.min_position - 0.1, -0.05])
        next_state, _, _ = env.step(0)  # Push left
        assert next_state[0] == env.min_position
        assert next_state[1] == 0.0
        
        # Test right boundary (goal)
        env.state = np.array([env.max_position - 0.01, 0.05])
        next_state, reward, done = env.step(1.0)  # Push right with continuous action
        assert next_state[0] >= env.goal_position
        assert done  # Should reach goal
        assert reward == 99.9  # Goal reward (100) minus force penalty (0.1 * 1.0^2)
    
    def test_reward_structure(self, mountain_car_config):
        """Test reward structure."""
        env = MountainCarEnv(mountain_car_config)
        env.reset()
        
        # Normal step should give -0.1*a^2 reward (force penalty)
        _, reward, done = env.step(1.0)  # Max force
        if not done:
            assert reward == -0.1  # -0.1 * 1.0^2
        
        # Reaching goal should give 100 reward minus force penalty
        env.state = np.array([env.goal_position, 0.01])
        _, reward, done = env.step(0.5)  # Half force
        if done:
            assert reward == 99.975  # 100.0 - 0.1 * 0.5^2 = 100.0 - 0.025


class TestPendulumEnv:
    """Test the Pendulum environment."""
    
    @pytest.fixture
    def pendulum_config(self):
        """Sample configuration for Pendulum."""
        return {
            'pendulum': {
                'max_speed': 8.0,
                'max_torque': 2.0,
                'time_step': 0.05,
                'gravity': 10.0,
                'mass': 1.0,
                'length': 1.0
            }
        }
    
    def test_initialization(self, pendulum_config):
        """Test Pendulum environment initialization."""
        env = PendulumEnv(pendulum_config)
        
        assert env.max_speed == 8.0
        assert env.max_torque == 2.0
        assert env.tau == 0.05
        assert env.gravity == 10.0
        assert env.mass == 1.0
        assert env.length == 1.0
    
    def test_reset(self, pendulum_config):
        """Test resetting the environment."""
        env = PendulumEnv(pendulum_config)
        obs = env.reset()
        
        assert len(obs) == 3  # [cos(theta), sin(theta), theta_dot]
        assert -1.0 <= obs[0] <= 1.0  # cos(theta)
        assert -1.0 <= obs[1] <= 1.0  # sin(theta)
        assert -1.0 <= obs[2] <= 1.0  # theta_dot (normalized)
    
    def test_step_continuous_action(self, pendulum_config):
        """Test stepping with continuous actions."""
        env = PendulumEnv(pendulum_config)
        env.reset()
        
        # Test different action types
        test_actions = [0.0, 0.5, -0.5, 1.0, -1.0, [0.3], np.array([0.7])]
        
        for action in test_actions:
            initial_state = env.state.copy()
            obs, reward, done = env.step(action)
            
            assert len(obs) == 3
            assert isinstance(reward, (int, float))
            assert done is False  # Pendulum episodes don't terminate naturally
            
            # State should change
            assert not np.array_equal(initial_state, env.state)
    
    def test_action_clipping(self, pendulum_config):
        """Test that actions are properly clipped."""
        env = PendulumEnv(pendulum_config)
        env.reset()
        
        # Test extreme actions
        extreme_actions = [10.0, -10.0, [5.0], np.array([-3.0])]
        
        for action in extreme_actions:
            # Should not raise an error
            obs, reward, done = env.step(action)
            assert len(obs) == 3
    
    def test_angle_normalization(self, pendulum_config):
        """Test that angles are properly normalized."""
        env = PendulumEnv(pendulum_config)
        
        # Set extreme angle
        env.state = np.array([10 * np.pi, 0.0])
        obs = env._get_obs()
        
        # Should still be valid observation
        assert -1.0 <= obs[0] <= 1.0  # cos(theta)
        assert -1.0 <= obs[1] <= 1.0  # sin(theta)
    
    def test_reward_structure(self, pendulum_config):
        """Test reward structure (cost minimization)."""
        env = PendulumEnv(pendulum_config)
        env.reset()
        
        # Upright position should give better reward (less negative)
        env.state = np.array([0.0, 0.0])  # Upright
        obs_up, reward_up, _ = env.step(0.0)
        
        # Hanging down should give worse reward (more negative)
        env.state = np.array([np.pi, 0.0])  # Hanging down
        obs_down, reward_down, _ = env.step(0.0)
        
        assert reward_up > reward_down  # Less negative is better


class TestAcrobotEnv:
    """Test the Acrobot environment."""
    
    @pytest.fixture
    def acrobot_config(self):
        """Sample configuration for Acrobot."""
        return {
            'acrobot': {
                'link_length_1': 1.0,
                'link_length_2': 1.0,
                'link_mass_1': 1.0,
                'link_mass_2': 1.0,
                'link_com_pos_1': 0.5,
                'link_com_pos_2': 0.5,
                'link_moi': 1.0,
                'max_vel_1': 4 * 3.14159,
                'max_vel_2': 9 * 3.14159,
                'torque_noise_max': 0.0,
                'time_step': 0.05,
                'gravity': 9.8
            }
        }
    
    def test_initialization(self, acrobot_config):
        """Test Acrobot environment initialization."""
        env = AcrobotEnv(acrobot_config)
        
        assert env.LINK_LENGTH_1 == 1.0
        assert env.LINK_LENGTH_2 == 1.0
        assert env.LINK_MASS_1 == 1.0
        assert env.LINK_MASS_2 == 1.0
        assert env.tau == 0.05
        assert env.gravity == 9.8
        assert len(env.AVAIL_TORQUE) == 3
    
    def test_reset(self, acrobot_config):
        """Test resetting the environment."""
        env = AcrobotEnv(acrobot_config)
        state = env.reset()
        
        assert len(state) == 4  # [theta1, theta2, theta1_dot, theta2_dot]
        # First link should start near hanging down (pi + small perturbation)
        assert abs(state[0] - np.pi) < 0.2
    
    def test_step_discrete_actions(self, acrobot_config):
        """Test stepping with discrete actions."""
        env = AcrobotEnv(acrobot_config)
        env.reset()
        
        # Test all three discrete actions
        for action in [0, 1, 2]:  # -1, 0, +1 torque
            initial_state = env.state.copy()
            next_state, reward, done = env.step(action)
            
            assert len(next_state) == 4
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            
            # State should change
            assert not np.array_equal(initial_state, next_state)
    
    def test_velocity_clipping(self, acrobot_config):
        """Test that velocities are properly clipped."""
        env = AcrobotEnv(acrobot_config)
        
        # Set extreme velocities
        env.state = np.array([0, 0, 100, 100])  # Extreme velocities
        next_state, _, _ = env.step(1)  # No torque
        
        # Velocities should be clipped
        assert abs(next_state[2]) <= env.MAX_VEL_1
        assert abs(next_state[3]) <= env.MAX_VEL_2
    
    def test_angle_wrapping(self, acrobot_config):
        """Test that angles are properly wrapped."""
        env = AcrobotEnv(acrobot_config)
        
        # Set angles outside [-pi, pi]
        env.state = np.array([10*np.pi, -5*np.pi, 0, 0])
        next_state, _, _ = env.step(1)
        
        # Angles should be wrapped to [-pi, pi]
        assert -np.pi <= next_state[0] <= np.pi
        assert -np.pi <= next_state[1] <= np.pi
    
    def test_goal_reaching(self, acrobot_config):
        """Test goal reaching condition."""
        env = AcrobotEnv(acrobot_config)
        
        # Set end-effector at goal height
        # For this, both links need to be roughly upright
        env.state = np.array([0.0, 0.0, 0.0, 0.0])  # Both links upright
        next_state, reward, done = env.step(1)
        
        # Should reach goal (or be very close)
        if done:
            assert reward == 0.0  # Goal reached gives 0 reward (no more -1 penalty)
        else:
            assert reward == -1.0
    
    def test_reward_structure(self, acrobot_config):
        """Test reward structure."""
        env = AcrobotEnv(acrobot_config)
        env.reset()
        
        # Normal step should give -1 reward
        _, reward, done = env.step(1)
        if not done:
            assert reward == -1.0
        
        # Reaching goal should give 100 reward
        # (This is tested indirectly in test_goal_reaching)


class TestEnvironmentIntegration:
    """Integration tests for all environments."""
    
    def test_all_environments_run_episode(self, sample_config):
        """Test that all environments can run a complete episode."""
        environments = ['cartpole', 'mountain_car', 'pendulum', 'acrobot']
        
        for env_name in environments:
            config = sample_config.copy()
            config['game'] = {'environment': env_name}
            
            # Add specific config for each environment
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
                    'max_vel_1': 4 * 3.14159, 'max_vel_2': 9 * 3.14159, 'torque_noise_max': 0.0,
                    'time_step': 0.05, 'gravity': 9.8
                }
            
            env = EnvironmentFactory.create_environment(config)
            specs = EnvironmentFactory.get_environment_specs(config)
            
            # Run a short episode
            state = env.reset()
            for _ in range(10):
                if env_name == 'pendulum':
                    action = 0.5  # Continuous action
                else:
                    action = 0  # Discrete action
                
                next_state, reward, done = env.step(action)
                state = next_state
                
                if done:
                    break
            
            # Verify state dimensions match specs
            if env_name == 'pendulum':
                assert len(state) == specs['input_dim']  # Observation space
            else:
                assert len(state) == specs['input_dim']  # State space
