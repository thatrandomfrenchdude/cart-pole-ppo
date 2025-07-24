"""
Tests for the main module including example mode and training mode initialization.
"""
import threading
import pytest
import os
from collections import deque
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMain:
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    def test_main_example_mode(self, mock_load_config, mock_setup_logging, 
                              mock_env_factory, mock_ppo_agent, 
                              mock_thread, mock_create_app):
        """Test main function in example mode."""
        # Configure mocks
        example_config = {
            'game': {'environment': 'cartpole'},
            'network': {'input_dim': 4, 'output_dim': 2},
            'training': {
                'example_mode': True,
                'example_model_paths': {
                    'cartpole': 'example/model.pth',
                    'mountain_car': 'example/mountain_car_model.pth',
                    'pendulum': 'example/pendulum_model.pth',
                    'acrobot': 'example/acrobot_model.pth'
                },
                'simulation_speed': 0.01,
                'reward_history_length': 100,
                'episode_history_length': 50
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200
            },
            'server': {
                'host': '127.0.0.1',
                'port': 8080,
                'debug': False
            }
        }
        mock_load_config.return_value = example_config
        mock_setup_logging.return_value = ('test.log', 'overwrite mode')
        
        # Mock environment factory
        mock_env_factory.get_environment_specs.return_value = {'input_dim': 4, 'output_dim': 2}
        mock_env_factory.create_environment.return_value = Mock()
        
        # Mock Flask app
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_app.run.side_effect = KeyboardInterrupt()  # Simulate user stopping
        
        # Mock thread
        mock_training_thread = Mock()
        mock_thread.return_value = mock_training_thread
        
        # Import and run main
        from main import main
        
        # Should handle KeyboardInterrupt gracefully
        main()
        
        # Verify example mode initialization
        mock_load_config.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_env_factory.create_environment.assert_called_once()
        mock_ppo_agent.assert_called_once()
        
        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_training_thread.start.assert_called_once()
        
        # Verify Flask app was created and run
        mock_create_app.assert_called_once()
        mock_app.run.assert_called_once_with(
            host='127.0.0.1', 
            port=8080, 
            debug=False, 
            threaded=True
        )
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    @patch('main.os.path.exists')
    def test_main_training_mode_new_model(self, mock_exists, mock_load_config, 
                                         mock_setup_logging, mock_env_factory, 
                                         mock_ppo_agent, mock_thread, mock_create_app):
        """Test main function in training mode with new model."""
        # Configure mocks
        training_config = {
            'game': {'environment': 'cartpole'},
            'network': {'input_dim': 4, 'output_dim': 2},
            'training': {
                'example_mode': False,
                'model_save_paths': {
                    'cartpole': 'models/new_model.pth',
                    'mountain_car': 'models/mountain_car_model.pth',
                    'pendulum': 'models/pendulum_model.pth',
                    'acrobot': 'models/acrobot_model.pth'
                },
                'save_frequency': 50,
                'reward_history_length': 1000,
                'episode_history_length': 100,
                'solved_reward_thresholds': {
                    'cartpole': 195.0,
                    'mountain_car': -110.0,
                    'pendulum': -200.0,
                    'acrobot': -100.0
                },
                'solved_episodes_window': 100,
                'stop_when_solved': True,
                'simulation_speed': 0.01
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200,
                'learning_rate': 0.0003
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': False
            }
        }
        mock_load_config.return_value = training_config
        mock_setup_logging.return_value = ('training.log', 'overwrite mode')
        mock_exists.return_value = False  # New model (doesn't exist)
        
        # Mock environment factory
        mock_env_factory.get_environment_specs.return_value = {'input_dim': 4, 'output_dim': 2}
        mock_env_factory.create_environment.return_value = Mock()
        
        # Mock Flask app
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_app.run.side_effect = KeyboardInterrupt()  # Simulate user stopping
        
        # Mock thread
        mock_training_thread = Mock()
        mock_thread.return_value = mock_training_thread
        
        # Import and run main
        from main import main
        
        main()
        
        # Verify training mode initialization
        mock_exists.assert_called_once_with('models/new_model.pth')
        mock_setup_logging.assert_called_once_with(training_config, append_mode=False)
        
        # Verify environment and agent creation
        mock_env_factory.create_environment.assert_called_once_with(training_config)
        mock_ppo_agent.assert_called_once_with(training_config)
        
        # Verify training thread was created with correct arguments
        mock_thread.assert_called_once()
        call_args = mock_thread.call_args
        assert call_args[1]['target'] is not None  # training_loop function
        assert call_args[1]['daemon'] is False
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    @patch('main.os.path.exists')
    def test_main_training_mode_resume_model(self, mock_exists, mock_load_config, 
                                           mock_setup_logging, mock_env_factory, 
                                           mock_ppo_agent, mock_thread, mock_create_app):
        """Test main function in training mode resuming existing model."""
        # Configure mocks
        training_config = {
            'game': {'environment': 'cartpole'},
            'training': {
                'example_mode': False,
                'model_save_paths': {
                    'cartpole': 'models/existing_model.pth'
                },
                'save_frequency': 50,
                'reward_history_length': 1000,
                'episode_history_length': 100,
                'solved_reward_thresholds': {
                    'cartpole': 195.0
                },
                'solved_episodes_window': 100,
                'stop_when_solved': True,
                'simulation_speed': 0.01
            },
            'network': {
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 2
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200,
                'learning_rate': 0.0003
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': False
            }
        }
        mock_load_config.return_value = training_config
        mock_setup_logging.return_value = ('training.log', 'append mode')
        mock_exists.return_value = True  # Existing model
        
        # Mock Flask app
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_app.run.side_effect = KeyboardInterrupt()
        
        # Mock thread
        mock_training_thread = Mock()
        mock_thread.return_value = mock_training_thread
        
        # Import and run main
        from main import main
        
        main()
        
        # Verify append mode logging for resuming
        mock_setup_logging.assert_called_once_with(training_config, append_mode=True)
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    def test_shared_state_initialization_example_mode(self, mock_load_config, 
                                                    mock_setup_logging, mock_env_factory, 
                                                    mock_ppo_agent, mock_thread, mock_create_app):
        """Test shared state initialization in example mode."""
        example_config = {
            'game': {'environment': 'cartpole'},
            'training': {
                'example_mode': True,
                'example_model_paths': {
                    'cartpole': 'example/model.pth'
                },
                'simulation_speed': 0.01,
                'reward_history_length': 100,
                'episode_history_length': 50
            },
            'network': {
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 2
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200
            },
            'server': {'host': '127.0.0.1', 'port': 8080, 'debug': False}
        }
        mock_load_config.return_value = example_config
        mock_setup_logging.return_value = ('test.log', 'overwrite mode')
        
        # Mock Flask app to capture shared state
        captured_current_state = None
        captured_reward_history = None
        captured_episode_rewards = None
        
        def capture_create_app_args(current_state, reward_history, episode_rewards):
            nonlocal captured_current_state, captured_reward_history, captured_episode_rewards
            captured_current_state = current_state
            captured_reward_history = reward_history
            captured_episode_rewards = episode_rewards
            mock_app = Mock()
            mock_app.run.side_effect = KeyboardInterrupt()
            return mock_app
        
        mock_create_app.side_effect = capture_create_app_args
        
        # Import and run main
        from main import main
        
        main()
        
        # Verify example mode shared state initialization
        assert captured_current_state is not None
        assert captured_current_state['reward'] == 1.0  # Frozen example reward
        assert captured_current_state['episode'] == 999  # Frozen example episode
        
        # Verify reward history populated with example data
        assert len(captured_reward_history) == 100
        assert all(r == 195.0 for r in captured_reward_history)
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    def test_exception_handling(self, mock_load_config, mock_setup_logging, 
                               mock_env_factory, mock_ppo_agent, mock_thread, mock_create_app):
        """Test exception handling in main function."""
        training_config = {
            'game': {'environment': 'cartpole'},
            'training': {
                'example_mode': False, 
                'model_save_paths': {
                    'cartpole': 'test.pth'
                },
                'reward_history_length': 100,
                'episode_history_length': 50,
                'simulation_speed': 0.01,
                'save_frequency': 10,
                'solved_reward_thresholds': {
                    'cartpole': 195.0
                },
                'solved_episodes_window': 100,
                'stop_when_solved': True
            },
            'network': {
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 2
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200,
                'learning_rate': 0.0003
            },
            'server': {'host': '127.0.0.1', 'port': 8080, 'debug': False}
        }
        mock_load_config.return_value = training_config
        mock_setup_logging.return_value = ('test.log', 'overwrite mode')
        
        # Mock Flask app to raise an exception
        mock_app = Mock()
        mock_app.run.side_effect = Exception("Test exception")
        mock_create_app.return_value = mock_app
        
        # Mock thread
        mock_training_thread = Mock()
        mock_thread.return_value = mock_training_thread
        
        # Import and run main
        from main import main
        
        # Should handle exception gracefully without raising
        main()
        
        # Verify cleanup occurred
        mock_training_thread.join.assert_called_once_with(timeout=5)
    
    @patch('main.create_app')
    @patch('main.threading.Thread')
    @patch('main.PPOAgent')
    @patch('main.EnvironmentFactory')
    @patch('main.setup_logging')
    @patch('main.load_config')
    def test_running_flag_shared_correctly(self, mock_load_config, mock_setup_logging, 
                                         mock_env_factory, mock_ppo_agent, 
                                         mock_thread, mock_create_app):
        """Test that running flag is properly shared between threads."""
        example_config = {
            'game': {'environment': 'cartpole'},
            'training': {
                'example_mode': True,
                'example_model_paths': {
                    'cartpole': 'example/model.pth'
                },
                'simulation_speed': 0.01,
                'reward_history_length': 100,
                'episode_history_length': 50
            },
            'network': {
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 2
            },
            'logging': {
                'episode_summary_frequency': 10
            },
            'ppo': {
                'update_frequency': 200
            },
            'server': {'host': '127.0.0.1', 'port': 8080, 'debug': False}
        }
        mock_load_config.return_value = example_config
        mock_setup_logging.return_value = ('test.log', 'overwrite mode')
        
        # Capture the running flag passed to thread
        captured_running_flag = None
        
        def capture_thread_args(*args, **kwargs):
            nonlocal captured_running_flag
            if 'args' in kwargs:
                # Find running_flag in the arguments (it's the second to last one, before config)
                captured_running_flag = kwargs['args'][-2]
            mock_thread = Mock()
            return mock_thread
        
        mock_thread.side_effect = capture_thread_args
        
        # Mock Flask app
        mock_app = Mock()
        mock_app.run.side_effect = KeyboardInterrupt()
        mock_create_app.return_value = mock_app
        
        # Import and run main
        from main import main
        
        main()
        
        # Verify running flag is a dict with 'value' key
        assert captured_running_flag is not None
        assert isinstance(captured_running_flag, dict)
        assert 'value' in captured_running_flag
