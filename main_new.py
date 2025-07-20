import threading
import logging
import os
from collections import deque

from src.config import load_config
from src.utils import setup_logging
from src.environment import CartPoleEnv
from src.agent import PPOAgent
from src.training import training_loop, example_mode_loop
from src.web_server import create_app


# Initialize logger with default settings, will be reconfigured in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config()
    
    # Check for example mode
    example_mode = config['training'].get('example_mode', False)
    
    # Global variables for Flask endpoints and shared state
    current_state = {"position": 0, "velocity": 0, "angle": 0, "angular_velocity": 0, "reward": 0, "episode": 0, "timestep": 0}
    reward_history = deque(maxlen=1000)  # Default maxlen, will be updated in main
    episode_rewards = deque(maxlen=100)  # Default maxlen, will be updated in main
    running_flag = {'value': True}  # Use dict so it's mutable and shared
    
    if example_mode:
        example_model_path = config['training'].get('example_model_path', 'example/model.pth')
        
        # Set up logging for example mode (always overwrite)
        log_file, log_mode_desc = setup_logging(config, append_mode=False)
        
        logger.info("="*60)
        logger.info("🎬 PPO Cart-Pole Example Mode")
        logger.info("="*60)
        logger.info("Configuration loaded successfully")
        logger.info(f"Example model path: {example_model_path}")
        logger.info(f"Logging to file: {log_file} ({log_mode_desc})")
        
        # Initialize environment and agent for example mode
        env = CartPoleEnv(config)
        agent = PPOAgent(config)
        
        # Initialize current_state with frozen example values for display
        current_state.update({
            "reward": 1.0,  # Frozen - typical reward for successful steps
            "episode": 999,  # Frozen - indicates example mode
        })
        
        # Initialize reward history with example data (frozen for display)
        example_rewards = [195.0] * 100  # Show that this model is already solved
        reward_history.extend(example_rewards)
        episode_rewards.extend(example_rewards)
        
        # Start example mode in a separate thread
        training_thread = threading.Thread(
            target=example_mode_loop,
            args=(env, agent, config['training']['simulation_speed'], example_model_path, current_state, running_flag),
            daemon=False
        )
        training_thread.start()
        
        logger.info("Starting example demonstration and Flask server...")
        logger.info(f"Access the visualization at http://{config['server']['host']}:{config['server']['port']}")
        logger.info("="*60)
        
    else:
        # Original training mode
        # Check if we're resuming from an existing model
        model_save_path = config['training']['model_save_path']
        model_exists = os.path.exists(model_save_path)
        
        # Update global deque sizes with config settings
        reward_history.clear()
        reward_history = deque(reward_history, maxlen=config['training']['reward_history_length'])
        episode_rewards.clear()
        episode_rewards = deque(episode_rewards, maxlen=config['training']['episode_history_length'])
        
        # Set up logging - append if resuming, overwrite if starting fresh
        log_file, log_mode_desc = setup_logging(config, append_mode=model_exists)
        
        # Add session separator if appending to existing log
        if model_exists:
            logger.info("")  # Empty line for separation
            logger.info("="*60)
            logger.info("PPO Cart-Pole Training Session RESUMED")
            logger.info("="*60)
        else:
            logger.info("="*60)
            logger.info("PPO Cart-Pole Training Session Started")
            logger.info("="*60)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Logging to file: {log_file} ({log_mode_desc})")
        logger.info(f"Training configuration:")
        logger.info(f"  - Episodes between model saves: {config['training']['save_frequency']}")
        logger.info(f"  - Episodes between summaries: {config['logging']['episode_summary_frequency']}")
        logger.info(f"  - Model save path: {config['training']['model_save_path']}")
        logger.info(f"  - Learning rate: {config['ppo']['learning_rate']}")
        logger.info(f"  - Update frequency: {config['ppo']['update_frequency']} steps")
        logger.info(f"  - Solved threshold: {config['training']['solved_reward_threshold']} average reward")
        logger.info(f"  - Solved window: {config['training']['solved_episodes_window']} episodes")
        logger.info(f"  - Stop when solved: {config['training']['stop_when_solved']}")
        
        # Initialize environment and agent
        env = CartPoleEnv(config)
        agent = PPOAgent(config)
        
        # Model configuration (model_save_path already defined above)
        save_frequency = config['training']['save_frequency']
        logger.info(f"Model save configuration: path='{model_save_path}', frequency={save_frequency}")
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=training_loop, 
            args=(env, agent, config['training']['simulation_speed'], config['logging']['episode_summary_frequency'], config['ppo']['update_frequency'], model_save_path, save_frequency, config['training']['solved_reward_threshold'], config['training']['solved_episodes_window'], config['training']['stop_when_solved'], current_state, reward_history, episode_rewards, running_flag),
            daemon=False  # Don't make it a daemon so it can run properly
        )
        training_thread.start()
        
        logger.info("Starting PPO Cart-Pole training and Flask server...")
        logger.info(f"Access the visualization at http://{config['server']['host']}:{config['server']['port']}")
        logger.info("="*60)
    
    # Create Flask app with shared state
    app = create_app(current_state, reward_history, episode_rewards)
    
    try:
        app.run(host=config['server']['host'], port=config['server']['port'], debug=config['server']['debug'], threaded=True)
    except KeyboardInterrupt:
        if example_mode:
            logger.info("Example demonstration interrupted by user")
        else:
            logger.info("Training interrupted by user")
        running_flag['value'] = False  # Signal the thread to stop
        training_thread.join(timeout=5)  # Wait for thread to finish
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        running_flag['value'] = False
        training_thread.join(timeout=5)
    finally:
        running_flag['value'] = False
        logger.info("="*60)
        if example_mode:
            logger.info("PPO Cart-Pole Example Mode Ended")
        else:
            logger.info("PPO Cart-Pole Training Session Ended")
        logger.info("="*60)


if __name__ == "__main__":
    main()
