import torch
import torch.nn as nn
import torch.nn.functional as F


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
