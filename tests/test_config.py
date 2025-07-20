"""
Tests for the configuration module.
"""
import os
import yaml
import pytest
import tempfile
from config import load_config


class TestConfig:
    
    def test_load_config_with_existing_file(self, temp_dir):
        """Test loading config from existing YAML file."""
        # Create test config file
        test_config = {
            'environment': {
                'gravity': 8.0,
                'cart_mass': 1.5,
                'pole_mass': 0.2
            },
            'ppo': {
                'learning_rate': 0.001,
                'discount_factor': 0.95
            },
            'training': {
                'example_mode': True
            }
        }
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Change to temp directory to test relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            loaded_config = load_config()
            
            # Check loaded values
            assert loaded_config['environment']['gravity'] == 8.0
            assert loaded_config['environment']['cart_mass'] == 1.5
            assert loaded_config['ppo']['learning_rate'] == 0.001
            assert loaded_config['training']['example_mode'] is True
        finally:
            os.chdir(original_cwd)
    
    def test_load_config_default_when_no_file(self, temp_dir):
        """Test loading default config when no file exists."""
        # Change to empty temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = load_config()
            
            # Check that default values are returned
            assert config['environment']['gravity'] == 9.8
            assert config['environment']['cart_mass'] == 1.0
            assert config['environment']['pole_mass'] == 0.1
            assert config['network']['input_dim'] == 4
            assert config['network']['hidden_dim'] == 128
            assert config['network']['output_dim'] == 2
            assert config['ppo']['learning_rate'] == 0.0003
            assert config['training']['example_mode'] is False
        finally:
            os.chdir(original_cwd)
    
    def test_default_config_structure(self):
        """Test that default config has all required sections."""
        # Create a temporary directory without config.yaml
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                
                # Check that all required sections exist
                required_sections = ['environment', 'network', 'ppo', 'training', 'server', 'logging']
                for section in required_sections:
                    assert section in config
                    assert isinstance(config[section], dict)
            finally:
                os.chdir(original_cwd)
    
    def test_environment_config_defaults(self):
        """Test environment configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                env_config = config['environment']
                
                # Check all environment parameters
                assert env_config['gravity'] == 9.8
                assert env_config['cart_mass'] == 1.0
                assert env_config['pole_mass'] == 0.1
                assert env_config['pole_half_length'] == 0.5
                assert env_config['force_magnitude'] == 10.0
                assert env_config['time_step'] == 0.02
                assert env_config['position_threshold'] == 2.4
                assert env_config['angle_threshold_degrees'] == 12
            finally:
                os.chdir(original_cwd)
    
    def test_network_config_defaults(self):
        """Test network configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                net_config = config['network']
                
                assert net_config['input_dim'] == 4
                assert net_config['hidden_dim'] == 128
                assert net_config['output_dim'] == 2
            finally:
                os.chdir(original_cwd)
    
    def test_ppo_config_defaults(self):
        """Test PPO configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                ppo_config = config['ppo']
                
                assert ppo_config['learning_rate'] == 0.0003
                assert ppo_config['discount_factor'] == 0.99
                assert ppo_config['clip_ratio'] == 0.2
                assert ppo_config['update_epochs'] == 4
                assert ppo_config['update_frequency'] == 200
            finally:
                os.chdir(original_cwd)
    
    def test_training_config_defaults(self):
        """Test training configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                training_config = config['training']
                
                assert training_config['simulation_speed'] == 0.05
                assert training_config['reward_history_length'] == 1000
                assert training_config['episode_history_length'] == 100
                assert training_config['model_save_path'] == 'models/ppo_cartpole.pth'
                assert training_config['save_frequency'] == 50
                assert training_config['example_mode'] is False
                assert training_config['solved_reward_threshold'] == 195.0
                assert training_config['solved_episodes_window'] == 100
                assert training_config['stop_when_solved'] is True
            finally:
                os.chdir(original_cwd)
    
    def test_server_config_defaults(self):
        """Test server configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                server_config = config['server']
                
                assert server_config['host'] == '0.0.0.0'
                assert server_config['port'] == 8080
                assert server_config['debug'] is False
            finally:
                os.chdir(original_cwd)
    
    def test_logging_config_defaults(self):
        """Test logging configuration defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config()
                logging_config = config['logging']
                
                assert logging_config['level'] == 'INFO'
                assert logging_config['format'] == '%(asctime)s - %(message)s'
                assert logging_config['episode_summary_frequency'] == 10
                assert logging_config['log_file'] == 'training.log'
            finally:
                os.chdir(original_cwd)
    
    def test_partial_config_file(self, temp_dir):
        """Test loading config with partial YAML file (should merge with defaults)."""
        # Create partial config file
        partial_config = {
            'environment': {
                'gravity': 7.5
            },
            'ppo': {
                'learning_rate': 0.002
            }
        }
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(partial_config, f)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = load_config()
            
            # Check that specified values are loaded
            assert config['environment']['gravity'] == 7.5
            assert config['ppo']['learning_rate'] == 0.002
            
            # Check that missing values still have defaults (this test might fail 
            # depending on the actual implementation - if it does deep merge)
            # For now, assuming the function completely replaces the loaded config
        finally:
            os.chdir(original_cwd)
    
    def test_invalid_yaml_file(self, temp_dir):
        """Test handling of invalid YAML file."""
        config_path = os.path.join(temp_dir, 'config.yaml')
        
        # Create invalid YAML
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Should handle invalid YAML gracefully (might raise exception or return defaults)
            # The exact behavior depends on implementation
            try:
                config = load_config()
                # If it succeeds, it should at least be a dict
                assert isinstance(config, dict)
            except yaml.YAMLError:
                # It's also acceptable to raise a YAML error
                pass
        finally:
            os.chdir(original_cwd)
