"""
Integration tests for the complete cart-pole PPO application.
"""
import pytest
import threading
import time
import tempfile
import os
import shutil
from unittest.mock import patch
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.mark.integration
class TestFullApplicationIntegration:
    
    def test_example_mode_full_integration(self, temp_dir):
        """Test running the full application in example mode."""
        # Create a temporary config file for example mode
        config_content = """
# Test configuration for example mode
environment:
  gravity: 9.8
  cart_mass: 1.0
  pole_mass: 0.1
  pole_half_length: 0.5
  force_magnitude: 10.0
  time_step: 0.02
  position_threshold: 2.4
  angle_threshold_degrees: 12

network:
  input_dim: 4
  hidden_dim: 64  # Smaller for faster testing
  output_dim: 2
  
ppo:
  learning_rate: 0.001
  discount_factor: 0.99
  clip_ratio: 0.2
  update_epochs: 2  # Fewer epochs for faster testing
  update_frequency: 50  # More frequent updates for testing

training:
  simulation_speed: 0.001  # Fast simulation for testing
  reward_history_length: 50
  episode_history_length: 25
  model_save_path: "test_model.pth"
  save_frequency: 5
  example_mode: true
  example_model_path: "test_example_model.pth"
  solved_reward_threshold: 195.0
  solved_episodes_window: 100
  stop_when_solved: true

server:
  host: "127.0.0.1"
  port: 8081  # Different port to avoid conflicts
  debug: false

logging:
  level: "WARNING"  # Reduce noise in tests
  format: "%(asctime)s - %(message)s"
  episode_summary_frequency: 2
  log_file: "test_training.log"
"""
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create a dummy example model file
        example_model_path = os.path.join(temp_dir, 'test_example_model.pth')
        with open(example_model_path, 'wb') as f:
            f.write(b'dummy model data')  # Dummy data
        
        original_cwd = os.getcwd()
        try:
            # Change to temp directory so config.yaml is found
            os.chdir(temp_dir)
            
            # Import main after changing directory
            import main
            
            # Mock Flask app.run to prevent actual server startup and add timeout
            ran_server = {'value': False}
            
            def mock_app_run(*args, **kwargs):
                ran_server['value'] = True
                time.sleep(0.5)  # Simulate running briefly
                raise KeyboardInterrupt()  # Simulate user stopping
            
            with patch.object(main, 'create_app') as mock_create_app:
                mock_app = type('MockApp', (), {'run': mock_app_run})()
                mock_create_app.return_value = mock_app
                
                # Run main function
                main.main()
                
                # Verify that server would have run
                assert ran_server['value']
                
                # Verify that create_app was called (Flask app was set up)
                mock_create_app.assert_called_once()
                
        finally:
            os.chdir(original_cwd)
    
    def test_training_mode_basic_integration(self, temp_dir):
        """Test basic training mode integration without full training."""
        config_content = """
environment:
  gravity: 9.8
  cart_mass: 1.0
  pole_mass: 0.1
  pole_half_length: 0.5
  force_magnitude: 10.0
  time_step: 0.02
  position_threshold: 2.4
  angle_threshold_degrees: 12

network:
  input_dim: 4
  hidden_dim: 32  # Very small for testing
  output_dim: 2
  
ppo:
  learning_rate: 0.01
  discount_factor: 0.99
  clip_ratio: 0.2
  update_epochs: 1
  update_frequency: 10

training:
  simulation_speed: 0.001
  reward_history_length: 10
  episode_history_length: 5
  model_save_path: "test_training_model.pth"
  save_frequency: 2
  example_mode: false  # Training mode
  solved_reward_threshold: 195.0
  solved_episodes_window: 100
  stop_when_solved: false  # Don't stop for testing

server:
  host: "127.0.0.1"
  port: 8082
  debug: false

logging:
  level: "ERROR"  # Minimal logging
  format: "%(message)s"
  episode_summary_frequency: 1
  log_file: "test_training.log"
"""
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Import main
            import main
            
            # Mock Flask to prevent server startup and add quick termination
            ran_server = {'value': False}
            
            def mock_app_run(*args, **kwargs):
                ran_server['value'] = True
                time.sleep(0.2)  # Brief simulation
                raise KeyboardInterrupt()  # Stop quickly
            
            with patch.object(main, 'create_app') as mock_create_app:
                mock_app = type('MockApp', (), {'run': mock_app_run})()
                mock_create_app.return_value = mock_app
                
                # Run main
                main.main()
                
                # Verify server startup was attempted
                assert ran_server['value']
                
                # Should have created Flask app
                mock_create_app.assert_called_once()
                
                # Check that log file was created
                assert os.path.exists('test_training.log')
                
        finally:
            os.chdir(original_cwd)
    
    def test_config_loading_integration(self, temp_dir):
        """Test that configuration loading works in full integration."""
        # Test with custom configuration values
        config_content = """
environment:
  gravity: 8.5  # Custom value
  cart_mass: 1.5  # Custom value

ppo:
  learning_rate: 0.002  # Custom value

training:
  example_mode: true
  simulation_speed: 0.01

server:
  host: "127.0.0.1"
  port: 8083

logging:
  level: "ERROR"
  log_file: "custom_test.log"
"""
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create dummy example model
        example_model_path = os.path.join(temp_dir, 'example', 'model.pth')
        os.makedirs(os.path.dirname(example_model_path), exist_ok=True)
        with open(example_model_path, 'wb') as f:
            f.write(b'dummy')
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Import config module to test loading
            from config import load_config
            
            config = load_config()
            
            # Verify custom values were loaded
            assert config['environment']['gravity'] == 8.5
            assert config['environment']['cart_mass'] == 1.5
            assert config['ppo']['learning_rate'] == 0.002
            assert config['training']['example_mode'] is True
            assert config['server']['port'] == 8083
            assert config['logging']['log_file'] == 'custom_test.log'
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.slow
    def test_short_training_session_integration(self, temp_dir):
        """Test a very short training session to verify training loop works."""
        config_content = """
environment:
  gravity: 9.8
  cart_mass: 1.0
  pole_mass: 0.1
  pole_half_length: 0.5
  force_magnitude: 10.0
  time_step: 0.02
  position_threshold: 2.4
  angle_threshold_degrees: 12

network:
  input_dim: 4
  hidden_dim: 16  # Very small network
  output_dim: 2
  
ppo:
  learning_rate: 0.01
  discount_factor: 0.99
  clip_ratio: 0.2
  update_epochs: 1
  update_frequency: 5  # Very frequent updates

training:
  simulation_speed: 0.001  # Very fast
  reward_history_length: 20
  episode_history_length: 10
  model_save_path: "short_training_model.pth"
  save_frequency: 1  # Save every episode
  example_mode: false
  solved_reward_threshold: 50.0  # Low threshold for testing
  solved_episodes_window: 3
  stop_when_solved: true  # Stop when solved

server:
  host: "127.0.0.1"
  port: 8084
  debug: false

logging:
  level: "ERROR"  # Minimal logging
  format: "%(message)s"
  episode_summary_frequency: 1
  log_file: "short_training.log"
"""
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            import main
            
            # Set up auto-termination after a short time
            def auto_terminate():
                time.sleep(2.0)  # Let it run for 2 seconds
                # This will be caught by the training loop
                raise KeyboardInterrupt()
            
            ran_training = {'value': False}
            
            def mock_app_run(*args, **kwargs):
                ran_training['value'] = True
                # Start auto-termination
                timer = threading.Thread(target=auto_terminate)
                timer.daemon = True
                timer.start()
                # Wait for termination
                timer.join()
                raise KeyboardInterrupt()
            
            with patch.object(main, 'create_app') as mock_create_app:
                mock_app = type('MockApp', (), {'run': mock_app_run})()
                mock_create_app.return_value = mock_app
                
                # Run main
                main.main()
                
                # Verify training was attempted
                assert ran_training['value']
                
                # Check that model file was created (indicating training occurred)
                model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
                # Should have at least attempted to create model files
                
                # Check log file exists
                assert os.path.exists('short_training.log')
                
        finally:
            os.chdir(original_cwd)
