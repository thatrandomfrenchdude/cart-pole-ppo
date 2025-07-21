import time
import logging
import numpy as np
import os

try:
    from .model_loader import ExampleModeAgent
except ImportError:
    from model_loader import ExampleModeAgent


logger = logging.getLogger(__name__)


def training_loop(env, agent, simulation_speed, summary_frequency, update_frequency, model_save_path, save_frequency, solved_threshold, solved_window, stop_when_solved, current_state, reward_history, episode_rewards, running_flag):
    """
    Main training loop for PPO agent.
    
    Args:
        env: CartPole environment
        agent: PPO agent
        simulation_speed: Sleep time between steps
        summary_frequency: Episodes between summary logs
        update_frequency: Steps between PPO updates
        model_save_path: Path to save model checkpoints
        save_frequency: Episodes between model saves
        solved_threshold: Average reward threshold to consider problem solved
        solved_window: Number of consecutive episodes to average for solving criteria
        stop_when_solved: Whether to stop training when the problem is solved
        current_state: Shared state dict for web interface
        reward_history: Shared reward history deque
        episode_rewards: Shared episode rewards deque
        running_flag: Shared running flag dict
    """
    # Debug logging for parameters
    logger.info(f"Training loop started with parameters:")
    logger.info(f"  - Model save path: {model_save_path}")
    logger.info(f"  - Save frequency: {save_frequency} episodes")
    logger.info(f"  - Summary frequency: {summary_frequency} episodes")
    logger.info(f"  - Update frequency: {update_frequency} steps")
    logger.info(f"  - Solved threshold: {solved_threshold} average reward")
    logger.info(f"  - Solved window: {solved_window} episodes")
    logger.info(f"  - Stop when solved: {stop_when_solved}")
    
    # Initialize training state
    episode = 0
    timestep = 0
    update_timestep = update_frequency
    
    # Try to load existing training state
    try:
        training_state = agent.load_model(model_save_path)
        if training_state and training_state['episode'] > 0:
            episode = training_state['episode']
            timestep = training_state['timestep']
            
            # Restore reward histories if available
            if training_state['reward_history']:
                reward_history.extend(training_state['reward_history'])
            if training_state['episode_rewards']:
                episode_rewards.extend(training_state['episode_rewards'])
                
            logger.info(f"Resuming training from episode {episode + 1}, timestep {timestep}")
        else:
            logger.info("Starting fresh training session")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Starting fresh training session")
    
    def save_training_state():
        """Helper function to save current training state."""
        training_state = {
            'episode': episode,
            'timestep': timestep,
            'reward_history': list(reward_history),
            'episode_rewards': list(episode_rewards)
        }
        agent.save_model(model_save_path, training_state)
    
    try:
        while running_flag['value']:
            state = env.reset()
            episode_reward = 0
            done = False
            
            logger.info(f"Starting Episode {episode + 1}")
            
            while not done and running_flag['value']:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, log_prob, value, done)
                
                # Update current state for visualization
                current_state.update({
                    "position": float(next_state[0]),
                    "velocity": float(next_state[1]),
                    "angle": float(next_state[2]),
                    "angular_velocity": float(next_state[3]),
                    "reward": float(reward),
                    "episode": episode + 1,  # Display current episode (1-indexed)
                    "timestep": timestep
                })
                
                episode_reward += reward
                state = next_state
                timestep += 1
                
                # Log current state (reduce frequency for performance)
                if timestep % 10 == 0 or done:
                    logger.info(f"Step {timestep}: Pos={next_state[0]:.3f}, Angle={next_state[2]:.3f}, Reward={reward}")
                
                # Update policy
                if timestep % update_timestep == 0:
                    agent.update()
                    logger.info(f"PPO update completed at timestep {timestep}")
                
                time.sleep(simulation_speed)  # Control simulation speed
            
            if not running_flag['value']:  # Training was interrupted
                break
                
            reward_history.append(episode_reward)
            episode_rewards.append(episode_reward)
            
            logger.info(f"Episode {episode + 1} finished with reward: {episode_reward}")
            logger.info(f"Average reward (last 10 episodes): {np.mean(list(episode_rewards)[-10:]):.2f}")
            
            episode += 1
            
            # Check if the problem is solved
            if stop_when_solved and len(episode_rewards) >= solved_window:
                recent_avg = np.mean(list(episode_rewards)[-solved_window:])
                if recent_avg >= solved_threshold:
                    logger.info("="*60)
                    logger.info("🎉 CART-POLE PROBLEM SOLVED! 🎉")
                    logger.info(f"Average reward over last {solved_window} episodes: {recent_avg:.2f}")
                    logger.info(f"Threshold: {solved_threshold}")
                    logger.info(f"Total episodes completed: {episode}")
                    logger.info(f"Total timesteps: {timestep}")
                    logger.info("="*60)
                    
                    # Save the final solved model
                    logger.info("Saving solved model...")
                    save_training_state()
                    
                    # Set running to False to stop the training loop
                    running_flag['value'] = False
                    break
            
            # More frequent saves early in training, less frequent later
            should_save = False
            if episode <= 10:  # First 10 episodes - save every episode
                should_save = True
            elif episode <= 50:  # Episodes 11-50 - save every 5 episodes
                should_save = (episode % 5 == 0)
            else:  # After episode 50 - save every save_frequency episodes
                should_save = (episode % save_frequency == 0)
            
            if should_save:
                logger.info(f"Auto-saving at episode {episode}")
                save_training_state()
            
            if episode % summary_frequency == 0:
                logger.info(f"Completed {episode} episodes. Recent average: {np.mean(list(episode_rewards)[-10:]):.2f}")
        
    except Exception as e:
        logger.error(f"Error in training loop: {e}")
        logger.info("Saving training state before exit...")
        save_training_state()
        raise
    finally:
        # Always save final model when training loop ends
        logger.info("Training loop ending. Saving final state...")
        save_training_state()


def example_mode_loop(env, agent, simulation_speed, example_model_path, current_state, running_flag, config=None):
    """
    Example mode loop that runs a pre-trained model for demonstration.
    Supports .pth, .pt (TorchScript), and .onnx formats.
    Training metrics and episode rewards are frozen, but live state is updated.
    
    Args:
        env: CartPole environment
        agent: PPO agent (not used in example mode, kept for compatibility)
        simulation_speed: Sleep time between steps
        example_model_path: Path to the pre-trained model (.pth, .pt, or .onnx)
        current_state: Shared state dict for web interface
        running_flag: Shared running flag dict
        config: Configuration dict (required for .pth format)
    """
    logger.info("="*60)
    logger.info("🎬 EXAMPLE MODE - Running Pre-trained Model")
    logger.info("="*60)
    logger.info(f"Loading model from: {example_model_path}")
    
    # Check if model file exists
    if not os.path.exists(example_model_path):
        logger.error(f"Model file not found: {example_model_path}")
        logger.error("Cannot run example mode without a valid model")
        return
    
    # Load the pre-trained model using multi-format loader
    try:
        example_agent = ExampleModeAgent(example_model_path, config)
        
        format_info = example_agent.get_format_info()
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Format: {format_info['format'].upper()}")
        logger.info(f"   Supports value estimation: {format_info['supports_value_estimation']}")
        
        # Get training state if available (only for .pth format)
        training_state = example_agent.get_training_state()
        if training_state and training_state['episode'] > 0:
            logger.info(f"   Model was trained for {training_state['episode']} episodes")
        else:
            logger.info("   No training history available (deployment format)")
            
    except Exception as e:
        logger.error(f"Failed to load example model: {e}")
        logger.error("Cannot run example mode without a valid model")
        return
    
    logger.info("Starting example demonstration...")
    logger.info("Press Ctrl+C to stop the demonstration")
    
    # Initialize demonstration state
    episode = 1
    total_timestep = 0
    
    try:
        while running_flag['value']:
            state = env.reset()
            episode_reward = 0
            done = False
            timestep_in_episode = 0
            
            logger.info(f"Starting demonstration episode {episode} (Format: {format_info['format'].upper()})")
            
            while not done and running_flag['value']:
                # Use the trained model to select action (no exploration)
                action, log_prob, value = example_agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                # Update current state for live visualization (keep these dynamic)
                current_state.update({
                    "position": float(next_state[0]),
                    "velocity": float(next_state[1]),
                    "angle": float(next_state[2]),
                    "angular_velocity": float(next_state[3]),
                    "timestep": total_timestep
                })
                # Note: reward, episode are kept frozen from initialization
                
                episode_reward += reward
                state = next_state
                timestep_in_episode += 1
                total_timestep += 1
                
                # Log current state less frequently for cleaner output
                if timestep_in_episode % 20 == 0 or done:
                    value_str = f", Value={value:.3f}" if format_info['supports_value_estimation'] else ""
                    logger.info(f"Episode {episode}, Step {timestep_in_episode}: Pos={next_state[0]:.3f}, Angle={next_state[2]:.3f}, Reward={reward}{value_str}")
                
                time.sleep(simulation_speed)
            
            if not running_flag['value']:
                break
                
            logger.info(f"Demonstration episode {episode} finished with reward: {episode_reward}")
            episode += 1
            
            # Brief pause between episodes
            time.sleep(1.0)
        
    except Exception as e:
        logger.error(f"Error in example mode: {e}")
        raise
    finally:
        logger.info("Example mode demonstration ended")
        logger.info("="*60)
