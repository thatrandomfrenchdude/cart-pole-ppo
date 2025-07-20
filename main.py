import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from flask import Flask, jsonify
import threading
import time
from collections import deque
import logging
import math
import yaml
import os

# Load configuration
def load_config():
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
            'training': {'simulation_speed': 0.05, 'reward_history_length': 1000, 'episode_history_length': 100, 'model_save_path': 'models/ppo_cartpole.pth', 'save_frequency': 50},
            'server': {'host': '0.0.0.0', 'port': 8080, 'debug': False},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(message)s', 'episode_summary_frequency': 10, 'log_file': 'training.log'}
        }

def setup_logging(config):
    """Set up logging to both console and file with overwrite mode."""
    log_file = config['logging']['log_file']
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging to both console and file (overwrite mode)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode='w')  # File output (overwrite mode)
        ],
        force=True
    )
    
    return log_file

# Global variables for Flask endpoints
current_state = {"position": 0, "velocity": 0, "angle": 0, "angular_velocity": 0, "reward": 0}
reward_history = deque(maxlen=1000)  # Default maxlen, will be updated in main
episode_rewards = deque(maxlen=100)  # Default maxlen, will be updated in main
running = True

# Initialize logger with default settings, will be reconfigured in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
class CartPoleEnv:
    def __init__(self, config):
        env_config = config['environment']
        self.gravity = env_config['gravity']
        self.masscart = env_config['cart_mass']
        self.masspole = env_config['pole_mass']
        self.total_mass = self.masspole + self.masscart
        self.length = env_config['pole_half_length']  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = env_config['force_magnitude']
        self.tau = env_config['time_step']  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = env_config['angle_threshold_degrees'] * 2 * math.pi / 360
        self.x_threshold = env_config['position_threshold']
        
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        reward = 1.0 if not done else 0.0
        return self.state, reward, done

class PPONetwork(nn.Module):
    def __init__(self, config):
        super(PPONetwork, self).__init__()
        net_config = config['network']
        input_dim = net_config['input_dim']
        hidden_dim = net_config['hidden_dim']
        output_dim = net_config['output_dim']
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.shared(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, config):
        ppo_config = config['ppo']
        self.network = PPONetwork(config)
        self.optimizer = optim.Adam(self.network.parameters(), lr=ppo_config['learning_rate'])
        self.gamma = ppo_config['discount_factor']
        self.eps_clip = ppo_config['clip_ratio']
        self.k_epochs = ppo_config['update_epochs']
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.network(state)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        if len(self.states) == 0:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Convert lists to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        
        advantages = discounted_rewards - old_values
        
        # PPO update
        for _ in range(self.k_epochs):
            action_probs, values = self.network(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(values.squeeze(), discounted_rewards)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.clear_memory()
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save_model(self, filepath):
        """Save the model state dict and optimizer state."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory:  # Only create directory if directory path is not empty
                os.makedirs(directory, exist_ok=True)
            
            checkpoint = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'gamma': self.gamma,
                'eps_clip': self.eps_clip,
                'k_epochs': self.k_epochs
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Model successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {str(e)}")
            # Try to save to current directory as fallback
            fallback_path = "ppo_cartpole_backup.pth"
            try:
                torch.save(checkpoint, fallback_path)
                logger.info(f"Model saved to fallback location: {fallback_path}")
            except Exception as e2:
                logger.error(f"Failed to save to fallback location: {str(e2)}")
    
    def load_model(self, filepath):
        """Load the model state dict and optimizer state."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location='cpu')
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load hyperparameters if they exist in checkpoint
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.eps_clip = checkpoint.get('eps_clip', self.eps_clip)
            self.k_epochs = checkpoint.get('k_epochs', self.k_epochs)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        else:
            logger.info(f"No existing model found at {filepath}. Starting training from scratch.")
            return False

# Global variables for Flask endpoints
current_state = {"position": 0, "velocity": 0, "angle": 0, "angular_velocity": 0, "reward": 0}
reward_history = deque(maxlen=1000)  # Default maxlen, will be updated in main
episode_rewards = deque(maxlen=100)  # Default maxlen, will be updated in main
running = True

# Initialize logger with default settings, will be reconfigured in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def training_loop(env, agent, simulation_speed, summary_frequency, update_frequency, model_save_path, save_frequency):
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
    """
    global current_state, current_reward, episode_rewards, reward_history
    
    # Debug logging for parameters
    logger.info(f"Training loop started with parameters:")
    logger.info(f"  - Model save path: {model_save_path}")
    logger.info(f"  - Save frequency: {save_frequency} episodes")
    logger.info(f"  - Summary frequency: {summary_frequency} episodes")
    logger.info(f"  - Update frequency: {update_frequency} steps")
    
    episode = 0
    update_timestep = update_frequency
    timestep = 0
    
    while running:
        state = env.reset()
        episode_reward = 0
        done = False
        
        logger.info(f"Starting Episode {episode + 1}")
        
        while not done and running:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            # Update current state for visualization
            current_state = {
                "position": float(next_state[0]),
                "velocity": float(next_state[1]),
                "angle": float(next_state[2]),
                "angular_velocity": float(next_state[3]),
                "reward": float(reward)
            }
            
            episode_reward += reward
            state = next_state
            timestep += 1
            
            # Log current state
            logger.info(f"Step {timestep}: Pos={next_state[0]:.3f}, Angle={next_state[2]:.3f}, Reward={reward}")
            
            # Update policy
            if timestep % update_timestep == 0:
                agent.update()
                logger.info(f"PPO update completed at timestep {timestep}")
            
            time.sleep(simulation_speed)  # Control simulation speed
        
        reward_history.append(episode_reward)
        episode_rewards.append(episode_reward)
        
        logger.info(f"Episode {episode + 1} finished with reward: {episode_reward}")
        logger.info(f"Average reward (last 10 episodes): {np.mean(list(episode_rewards)[-10:]):.2f}")
        
        episode += 1
        
        # Save model periodically
        if episode % save_frequency == 0:
            logger.info(f"Reached save frequency trigger: Episode {episode}, Save frequency: {save_frequency}")
            agent.save_model(model_save_path)
        
        if episode % summary_frequency == 0:
            logger.info(f"Completed {episode} episodes. Recent average: {np.mean(list(episode_rewards)[-10:]):.2f}")
    
    # Save final model when training loop ends
    logger.info("Training loop ended. Saving final model...")
    agent.save_model(model_save_path)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/styles.css')
def styles():
    with open('styles.css', 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/css'}

@app.route('/visualization.js')
def visualization():
    with open('visualization.js', 'r') as f:
        return f.read(), 200, {'Content-Type': 'application/javascript'}

@app.route('/state')
def get_state():
    return jsonify(current_state)

@app.route('/history')
def get_history():
    return jsonify({
        'rewards': list(reward_history),
        'avg_reward': float(np.mean(list(episode_rewards)) if episode_rewards else 0)
    })

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Update global deque sizes with config settings
    reward_history.clear()
    reward_history = deque(reward_history, maxlen=config['training']['reward_history_length'])
    episode_rewards.clear()
    episode_rewards = deque(episode_rewards, maxlen=config['training']['episode_history_length'])
    
    # Reconfigure logging with config settings
    log_file = setup_logging(config)
    
    logger.info("="*60)
    logger.info("PPO Cart-Pole Training Session Started")
    logger.info("="*60)
    logger.info("Configuration loaded successfully")
    logger.info(f"Logging to file: {log_file} (overwrite mode)")
    logger.info(f"Training configuration:")
    logger.info(f"  - Episodes between model saves: {config['training']['save_frequency']}")
    logger.info(f"  - Episodes between summaries: {config['logging']['episode_summary_frequency']}")
    logger.info(f"  - Model save path: {config['training']['model_save_path']}")
    logger.info(f"  - Learning rate: {config['ppo']['learning_rate']}")
    logger.info(f"  - Update frequency: {config['ppo']['update_frequency']} steps")
    
    # Initialize environment and agent
    env = CartPoleEnv(config)
    agent = PPOAgent(config)
    
    # Try to load existing model if it exists
    model_save_path = config['training']['model_save_path']
    save_frequency = config['training']['save_frequency']
    logger.info(f"Model save configuration: path='{model_save_path}', frequency={save_frequency}")
    
    if agent.load_model(model_save_path):
        logger.info("Resuming training from saved model")
    else:
        logger.info("Starting training with new model")
    
    # Start training in a separate thread
    training_thread = threading.Thread(
        target=training_loop, 
        args=(env, agent, config['training']['simulation_speed'], config['logging']['episode_summary_frequency'], config['ppo']['update_frequency'], model_save_path, save_frequency),
        daemon=True
    )
    training_thread.start()
    
    logger.info("Starting PPO Cart-Pole training and Flask server...")
    logger.info(f"Access the visualization at http://{config['server']['host']}:{config['server']['port']}")
    logger.info("="*60)
    
    try:
        app.run(host=config['server']['host'], port=config['server']['port'], debug=config['server']['debug'])
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        logger.info("="*60)
        logger.info("PPO Cart-Pole Training Session Ended")
        logger.info("="*60)
