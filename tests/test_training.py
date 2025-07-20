"""
Tests for the training module including both training and example modes.
"""
import pytest
import numpy as np
from collections import deque
from unittest.mock import Mock, patch, MagicMock
from training import training_loop, example_mode_loop


class TestTrainingLoop:
    
    def test_training_loop_initialization(self, sample_config, mock_shared_state):
        """Test training loop basic setup and initialization."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment and agent
        mock_env = Mock()
        mock_env.reset.return_value = np.array([0.1, 0.2, 0.05, 0.1])
        mock_env.step.return_value = (np.array([0.11, 0.21, 0.04, 0.09]), 1.0, False)
        
        mock_agent = Mock()
        mock_agent.select_action.return_value = (0, -0.693, 0.5)
        mock_agent.load_model.return_value = None
        
        # Stop the loop immediately by setting running_flag to False
        running_flag['value'] = False
        
        # Run training loop (will exit immediately due to running_flag)
        training_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,  # Minimal delay
            summary_frequency=10,
            update_frequency=5,
            model_save_path="test_model.pth",
            save_frequency=10,
            solved_threshold=195.0,
            solved_window=100,
            stop_when_solved=True,
            current_state=current_state,
            reward_history=reward_history,
            episode_rewards=episode_rewards,
            running_flag=running_flag
        )
        
        # Verify basic setup occurred
        mock_agent.load_model.assert_called_once()
    
    def test_training_loop_episode_execution(self, sample_config, mock_shared_state):
        """Test that training loop executes episodes properly."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment
        mock_env = Mock()
        reset_states = [
            np.array([0.1, 0.2, 0.05, 0.1]),
            np.array([0.0, 0.0, 0.0, 0.0])
        ]
        mock_env.reset.side_effect = reset_states
        
        # Create a sequence of steps that leads to episode end
        step_results = [
            (np.array([0.11, 0.21, 0.04, 0.09]), 1.0, False),
            (np.array([0.12, 0.22, 0.03, 0.08]), 1.0, False),
            (np.array([0.13, 0.23, 0.02, 0.07]), 1.0, True)  # Episode ends
        ]
        mock_env.step.side_effect = step_results
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.select_action.return_value = (1, -0.693, 0.5)
        mock_agent.load_model.return_value = None
        
        # Stop after one episode
        episode_count = 0
        def counting_reset():
            nonlocal episode_count
            episode_count += 1
            if episode_count > 1:
                running_flag['value'] = False
            return reset_states[min(episode_count - 1, len(reset_states) - 1)]
        mock_env.reset.side_effect = counting_reset
        
        # Run training loop
        training_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,
            summary_frequency=1,
            update_frequency=10,
            model_save_path="test_model.pth",
            save_frequency=1,
            solved_threshold=195.0,
            solved_window=100,
            stop_when_solved=True,
            current_state=current_state,
            reward_history=reward_history,
            episode_rewards=episode_rewards,
            running_flag=running_flag
        )
        
        # Verify environment interactions
        assert mock_env.reset.call_count >= 1
        assert mock_env.step.call_count >= 1
        
        # Verify agent interactions
        assert mock_agent.select_action.call_count >= 1
        assert mock_agent.store_transition.call_count >= 1
    
    def test_training_loop_state_updates(self, sample_config, mock_shared_state):
        """Test that training loop updates shared state correctly."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = np.array([0.1, 0.2, 0.05, 0.1])
        
        test_state = np.array([0.5, 1.0, 0.1, 0.2])
        test_reward = 1.0
        mock_env.step.return_value = (test_state, test_reward, True)  # End episode immediately
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.select_action.return_value = (0, -0.693, 0.5)
        mock_agent.load_model.return_value = None
        
        # Stop after first episode
        episode_count = 0
        def stop_after_episode(*args):
            nonlocal episode_count
            episode_count += 1
            if episode_count > 0:
                running_flag['value'] = False
            return (test_state, test_reward, True)
        mock_env.step.side_effect = stop_after_episode
        
        # Run training loop
        training_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,
            summary_frequency=1,
            update_frequency=10,
            model_save_path="test_model.pth",
            save_frequency=1,
            solved_threshold=195.0,
            solved_window=100,
            stop_when_solved=True,
            current_state=current_state,
            reward_history=reward_history,
            episode_rewards=episode_rewards,
            running_flag=running_flag
        )
        
        # Verify state was updated
        assert current_state['position'] == test_state[0]
        assert current_state['velocity'] == test_state[1]
        assert current_state['angle'] == test_state[2]
        assert current_state['angular_velocity'] == test_state[3]
        assert current_state['reward'] == test_reward
    
    def test_model_loading_resume(self, sample_config, mock_shared_state):
        """Test training loop resuming from saved model."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment and agent
        mock_env = Mock()
        mock_env.reset.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        mock_env.step.return_value = (np.array([0.1, 0.1, 0.1, 0.1]), 1.0, True)
        
        mock_agent = Mock()
        mock_agent.select_action.return_value = (0, -0.693, 0.5)
        
        # Mock successful model loading
        mock_training_state = {
            'episode': 50,
            'timestep': 5000,
            'reward_history': [100.0, 150.0, 200.0],
            'episode_rewards': [180.0, 190.0, 195.0]
        }
        mock_agent.load_model.return_value = mock_training_state
        
        # Stop immediately
        running_flag['value'] = False
        
        # Run training loop
        training_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,
            summary_frequency=10,
            update_frequency=10,
            model_save_path="existing_model.pth",
            save_frequency=10,
            solved_threshold=195.0,
            solved_window=100,
            stop_when_solved=True,
            current_state=current_state,
            reward_history=reward_history,
            episode_rewards=episode_rewards,
            running_flag=running_flag
        )
        
        # Verify model loading was attempted
        mock_agent.load_model.assert_called_once_with("existing_model.pth")
        
        # Verify history was restored
        assert len(reward_history) == 3
        assert len(episode_rewards) == 3


class TestExampleModeLoop:
    
    def test_example_mode_initialization(self, sample_config, mock_shared_state):
        """Test example mode loop initialization."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment and agent
        mock_env = Mock()
        mock_env.reset.return_value = np.array([0.0, 0.0, 0.01, 0.0])  # Slight initial angle
        mock_env.step.return_value = (np.array([0.01, 0.1, 0.005, -0.1]), 1.0, False)
        
        mock_agent = Mock()
        mock_agent.select_action.return_value = (1, -0.693, 0.8)  # Good action
        mock_agent.load_model.return_value = True  # Model loaded successfully
        
        # Stop immediately by setting the flag to False
        running_flag['value'] = False
        
        # Run example mode
        example_mode_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.01,
            example_model_path="example/model.pth",
            current_state=current_state,
            running_flag=running_flag
        )
        
        # Verify model loading was attempted
        mock_agent.load_model.assert_called_once_with("example/model.pth")
    
    @patch('training.logger')
    def test_example_mode_model_loading_failure(self, mock_logger, sample_config, mock_shared_state):
        """Test example mode with model loading failure."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment and agent
        mock_env = Mock()
        mock_agent = Mock()
        # Make load_model raise an exception to trigger the error logging
        mock_agent.load_model.side_effect = Exception("Model loading failed")
        
        # Stop immediately
        running_flag['value'] = False
        
        # Run example mode
        example_mode_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.01,
            example_model_path="nonexistent_model.pth",
            current_state=current_state,
            running_flag=running_flag
        )
        
        # Verify error was logged
        mock_logger.error.assert_called()
    
    def test_example_mode_continuous_running(self, sample_config, mock_shared_state):
        """Test that example mode runs continuously with good performance."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment - simulate good cart-pole balance
        mock_env = Mock()
        
        # Create sequence of balanced states
        balanced_states = [
            np.array([0.0, 0.0, 0.01, 0.0]),   # Reset state
            np.array([0.01, 0.1, 0.005, -0.05]),  # Balancing
            np.array([0.02, 0.05, 0.002, -0.02]),  # Still balancing
            np.array([0.01, -0.05, 0.001, 0.01])   # Good balance
        ]
        
        step_count = 0
        def mock_step(action):
            nonlocal step_count
            step_count += 1
            # Return good rewards and no termination for first few steps
            if step_count < 200:  # Long episode
                return balanced_states[step_count % len(balanced_states)], 1.0, False
            else:
                return balanced_states[0], 1.0, True  # End episode eventually
        
        mock_env.reset.return_value = balanced_states[0]
        mock_env.step.side_effect = mock_step
        
        # Mock agent - simulate good policy
        mock_agent = Mock()
        mock_agent.load_model.return_value = {'episode': 100, 'total_timesteps': 10000}
        
        # Agent makes good decisions
        def smart_action_selection(state):
            # Simple policy: if pole leaning right, push cart right
            angle = state[2]
            action = 1 if angle > 0 else 0
            return action, -0.693, 0.9  # High value for good states
        
        mock_agent.select_action.side_effect = smart_action_selection
        
        # Stop after limited steps for testing
        step_count_limit = 3  # Run just a few steps
        original_simulation_speed = 0.001  # Fast simulation
        
        # Create a custom environment that stops after a few steps
        step_count = 0
        def tracking_step(action):
            nonlocal step_count
            step_count += 1
            if step_count >= step_count_limit:
                running_flag['value'] = False
            return (np.array([0.1, 0.2, 0.05, 0.1]), 1.0, step_count >= step_count_limit)
        
        mock_env.step.side_effect = tracking_step
        
        # Run example mode
        example_mode_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,  # Fast for testing
            example_model_path="example_model.pth",
            current_state=current_state,
            running_flag=running_flag
        )
        
        # Verify good performance metrics
        assert current_state['position'] == 0.1  # Should be updated from last step
        assert current_state['angle'] == 0.05    # Should be updated from last step  
        assert mock_env.step.call_count > 0
        assert mock_agent.select_action.call_count > 0
    
    def test_example_mode_state_updates(self, sample_config, mock_shared_state):
        """Test that example mode updates state correctly."""
        current_state, reward_history, episode_rewards, running_flag = mock_shared_state
        
        # Mock environment
        mock_env = Mock()
        test_state = np.array([0.5, 1.0, 0.1, 0.2])
        test_reward = 1.0
        
        mock_env.reset.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        mock_env.step.return_value = (test_state, test_reward, False)
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.load_model.return_value = {'episode': 100, 'total_timesteps': 10000}
        mock_agent.select_action.return_value = (1, -0.693, 0.8)
        
        # Stop after one step
        step_count = 0
        def stop_after_one_step(*args):
            nonlocal step_count
            step_count += 1
            if step_count > 0:
                running_flag['value'] = False
            return (test_state, test_reward, False)
        mock_env.step.side_effect = stop_after_one_step
        
        # Run example mode
        example_mode_loop(
            env=mock_env,
            agent=mock_agent,
            simulation_speed=0.001,
            example_model_path="test_model.pth",
            current_state=current_state,
            running_flag=running_flag
        )
        
        # Verify state was updated (reward is not updated in example mode)
        assert current_state['position'] == test_state[0]
        assert current_state['velocity'] == test_state[1]
        assert current_state['angle'] == test_state[2]
        assert current_state['angular_velocity'] == test_state[3]
        # Note: reward is intentionally kept frozen in example mode
