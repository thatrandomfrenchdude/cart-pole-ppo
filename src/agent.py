import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import logging

try:
    from .network import PPONetwork
except ImportError:
    from network import PPONetwork


logger = logging.getLogger(__name__)


class PPOAgent:
    def __init__(self, config):
        ppo_config = config['ppo']
        self.network = PPONetwork(config)
        self.optimizer = optim.Adam(self.network.parameters(), lr=ppo_config['learning_rate'])
        self.gamma = ppo_config['discount_factor']
        self.eps_clip = ppo_config['clip_ratio']
        self.k_epochs = ppo_config['update_epochs']
        
        # Determine if this is a continuous action environment
        game_type = config['game']['environment'].lower()
        self.continuous_action = (game_type in ['pendulum', 'mountain_car'])
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if self.continuous_action:
            action_mean, action_std, value = self.network(state)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # Clip action to [-1, 1] range
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
            return action_clipped.detach().numpy().flatten(), log_prob.item(), value.item()
        else:
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
        if len(discounted_rewards) > 1 and discounted_rewards.std() > 1e-8:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        # If only one reward or no variance, don't normalize
        
        # Convert lists to tensors efficiently
        states = torch.FloatTensor(np.array(self.states))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        old_values = torch.FloatTensor(np.array(self.values))
        
        if self.continuous_action:
            actions = torch.FloatTensor(np.array(self.actions))
        else:
            actions = torch.LongTensor(np.array(self.actions))
        
        advantages = discounted_rewards - old_values
        
        # PPO update
        for _ in range(self.k_epochs):
            if self.continuous_action:
                action_mean, action_std, values = self.network(states)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
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
    
    def save_model(self, filepath, training_state=None):
        """Save the model state dict, optimizer state, and training progress."""
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
            
            # Add training state if provided
            if training_state:
                checkpoint.update(training_state)
            
            torch.save(checkpoint, filepath)
            logger.info(f"Model successfully saved to {filepath}")
            if training_state:
                logger.info(f"Training state saved - Episode: {training_state.get('episode', 'N/A')}, Timestep: {training_state.get('timestep', 'N/A')}")
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
        """Load the model state dict, optimizer state, and training progress."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load hyperparameters if they exist in checkpoint
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.eps_clip = checkpoint.get('eps_clip', self.eps_clip)
            self.k_epochs = checkpoint.get('k_epochs', self.k_epochs)
            
            logger.info(f"Model loaded from {filepath}")
            
            # Return training state if it exists
            training_state = {
                'episode': checkpoint.get('episode', 0),
                'timestep': checkpoint.get('timestep', 0),
                'reward_history': checkpoint.get('reward_history', []),
                'episode_rewards': checkpoint.get('episode_rewards', [])
            }
            
            if training_state['episode'] > 0:
                logger.info(f"Training state restored - Episode: {training_state['episode']}, Timestep: {training_state['timestep']}")
            
            return training_state
        else:
            logger.info(f"No existing model found at {filepath}. Starting training from scratch.")
            return None
