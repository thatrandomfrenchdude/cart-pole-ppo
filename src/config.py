import yaml
import os


def load_config():
    """Load configuration from YAML file or return defaults."""
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration if file doesn't exist
        return {
            'environment': {
                'gravity': 9.8, 'cart_mass': 1.0, 'pole_mass': 0.1, 'pole_half_length': 0.5,
                'force_magnitude': 10.0, 'time_step': 0.02, 'position_threshold': 2.4, 'angle_threshold_degrees': 12
            },
            'network': {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 2},
            'ppo': {'learning_rate': 0.0003, 'discount_factor': 0.99, 'clip_ratio': 0.2, 'update_epochs': 4, 'update_frequency': 200},
            'training': {'simulation_speed': 0.05, 'reward_history_length': 1000, 'episode_history_length': 100, 'model_save_path': 'models/ppo_cartpole.pth', 'save_frequency': 50, 'solved_reward_threshold': 195.0, 'solved_episodes_window': 100, 'stop_when_solved': True, 'example_mode': False, 'example_model_path': 'example/model.pth'},
            'server': {'host': '0.0.0.0', 'port': 8080, 'debug': False},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(message)s', 'episode_summary_frequency': 10, 'log_file': 'training.log'}
        }
