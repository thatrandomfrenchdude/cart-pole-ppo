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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple Cart-Pole Environment Implementation
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
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
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(PPONetwork, self).__init__()
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
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.network = PPONetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
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

# Global variables for Flask endpoints
current_state = {"position": 0, "velocity": 0, "angle": 0, "angular_velocity": 0, "reward": 0}
reward_history = deque(maxlen=1000)
episode_rewards = deque(maxlen=100)
running = True

def training_loop():
    global current_state, reward_history, episode_rewards, running
    
    env = CartPoleEnv()
    agent = PPOAgent()
    
    episode = 0
    update_timestep = 200
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
            
            time.sleep(0.05)  # Control simulation speed
        
        reward_history.append(episode_reward)
        episode_rewards.append(episode_reward)
        
        logger.info(f"Episode {episode + 1} finished with reward: {episode_reward}")
        logger.info(f"Average reward (last 10 episodes): {np.mean(list(episode_rewards)[-10:]):.2f}")
        
        episode += 1
        
        if episode % 10 == 0:
            logger.info(f"Completed {episode} episodes. Recent average: {np.mean(list(episode_rewards)[-10:]):.2f}")

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
    # Start training in a separate thread
    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()
    
    logger.info("Starting PPO Cart-Pole training and Flask server...")
    logger.info("Access the visualization at http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
