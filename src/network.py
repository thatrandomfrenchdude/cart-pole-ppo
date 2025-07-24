import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPONetwork(nn.Module):
    def __init__(self, config):
        super(PPONetwork, self).__init__()
        net_config = config['network']
        input_dim = net_config['input_dim']
        hidden_dim = net_config['hidden_dim']
        output_dim = net_config['output_dim']
        
        # Determine if this is a continuous action environment
        game_type = config['game']['environment'].lower()
        self.continuous_action = (game_type in ['pendulum', 'mountain_car'])
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if self.continuous_action:
            # For continuous actions, output mean and log_std
            self.actor_mean = nn.Linear(hidden_dim, output_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            # For discrete actions, output action logits
            self.actor = nn.Linear(hidden_dim, output_dim)
        
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.shared(x)
        state_value = self.critic(x)
        
        if self.continuous_action:
            action_mean = self.actor_mean(x)
            action_std = torch.exp(self.actor_log_std)
            return action_mean, action_std, state_value
        else:
            action_logits = self.actor(x)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, state_value
