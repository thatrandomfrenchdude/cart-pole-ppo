"""
Tests for the multi-environment neural network architecture.
"""
import torch
import numpy as np
import pytest
from src.network import PPONetwork


class TestPPONetworkDiscrete:
    """Test PPO network with discrete action environments."""
    
    def test_cartpole_network_initialization(self, sample_config):
        """Test network initialization for CartPole."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 2}
        
        network = PPONetwork(config)
        
        # Check network structure
        assert hasattr(network, 'shared')
        assert hasattr(network, 'actor')
        assert hasattr(network, 'critic')
        assert not hasattr(network, 'actor_mean')  # Discrete action network
        assert not hasattr(network, 'actor_log_std')
        assert network.continuous_action is False
    
    def test_mountain_car_network_forward(self, sample_config):
        """Test forward pass for MountainCar network."""
        config = sample_config.copy()
        config['game'] = {'environment': 'mountain_car'}
        config['network'] = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 3}
        
        network = PPONetwork(config)
        
        # Create batch of states
        batch_size = 5
        states = torch.randn(batch_size, 2)
        
        # Forward pass
        action_probs, values = network(states)
        
        # Check output shapes
        assert action_probs.shape == (batch_size, 3)
        assert values.shape == (batch_size, 1)
        
        # Check action probabilities sum to 1
        prob_sums = torch.sum(action_probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)
        
        # Check probabilities are non-negative
        assert torch.all(action_probs >= 0)
    
    def test_acrobot_network_dimensions(self, sample_config):
        """Test network dimensions for Acrobot."""
        config = sample_config.copy()
        config['game'] = {'environment': 'acrobot'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 3}
        
        network = PPONetwork(config)
        
        # Check layer dimensions
        shared_layers = list(network.shared.children())
        assert shared_layers[0].in_features == 4     # First linear layer input
        assert shared_layers[0].out_features == 128  # First linear layer output
        assert shared_layers[2].in_features == 128   # Second linear layer input
        assert shared_layers[2].out_features == 128  # Second linear layer output
        
        assert network.actor.in_features == 128
        assert network.actor.out_features == 3
        assert network.critic.in_features == 128
        assert network.critic.out_features == 1


class TestPPONetworkContinuous:
    """Test PPO network with continuous action environments."""
    
    def test_pendulum_network_initialization(self, sample_config):
        """Test network initialization for Pendulum."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        
        network = PPONetwork(config)
        
        # Check network structure for continuous actions
        assert hasattr(network, 'shared')
        assert hasattr(network, 'actor_mean')
        assert hasattr(network, 'actor_log_std')
        assert hasattr(network, 'critic')
        assert not hasattr(network, 'actor')  # No discrete actor
        assert network.continuous_action is True
    
    def test_pendulum_network_forward(self, sample_config):
        """Test forward pass for Pendulum network."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        
        network = PPONetwork(config)
        
        # Create batch of states
        batch_size = 5
        states = torch.randn(batch_size, 3)
        
        # Forward pass
        action_mean, action_std, values = network(states)
        
        # Check output shapes
        assert action_mean.shape == (batch_size, 1)
        assert action_std.shape == (1,)  # Shared log_std parameter
        assert values.shape == (batch_size, 1)
        
        # Check action std is positive
        assert torch.all(action_std > 0)
    
    def test_action_std_parameter(self, sample_config):
        """Test that action_log_std is a learnable parameter."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        
        network = PPONetwork(config)
        
        # Check that actor_log_std is a parameter
        assert isinstance(network.actor_log_std, torch.nn.Parameter)
        assert network.actor_log_std.requires_grad is True
        
        # Check initial value (should be zeros -> std = 1.0)
        initial_std = torch.exp(network.actor_log_std)
        expected_std = torch.ones_like(initial_std)
        assert torch.allclose(initial_std, expected_std)


class TestPPONetworkGradients:
    """Test gradient flow through the network."""
    
    def test_discrete_network_gradients(self, sample_config):
        """Test gradient flow for discrete action networks."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 2}
        
        network = PPONetwork(config)
        
        # Create dummy input and target
        states = torch.randn(10, 4, requires_grad=True)
        target_values = torch.randn(10, 1)
        
        # Forward pass
        action_probs, values = network(states)
        
        # Compute dummy loss
        value_loss = torch.nn.MSELoss()(values, target_values)
        action_loss = -torch.mean(torch.log(action_probs[:, 0]))
        total_loss = value_loss + action_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_continuous_network_gradients(self, sample_config):
        """Test gradient flow for continuous action networks."""
        config = sample_config.copy()
        config['game'] = {'environment': 'pendulum'}
        config['network'] = {'input_dim': 3, 'hidden_dim': 64, 'output_dim': 1}
        
        network = PPONetwork(config)
        
        # Create dummy input and targets
        states = torch.randn(10, 3, requires_grad=True)
        target_values = torch.randn(10, 1)
        target_actions = torch.randn(10, 1)
        
        # Forward pass
        action_mean, action_std, values = network(states)
        
        # Compute dummy loss that includes both mean and std
        value_loss = torch.nn.MSELoss()(values, target_values)
        action_loss = torch.nn.MSELoss()(action_mean, target_actions)
        # Include std in loss to ensure gradients flow to log_std parameter
        std_loss = torch.nn.MSELoss()(action_std, torch.ones_like(action_std))
        total_loss = value_loss + action_loss + std_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
        
        # Check that log_std parameter has gradients
        assert network.actor_log_std.grad is not None


class TestPPONetworkEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unknown_environment_type(self, sample_config):
        """Test network creation with unknown environment type."""
        config = sample_config.copy()
        config['game'] = {'environment': 'unknown_env'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 2}
        
        # Should default to discrete action network
        network = PPONetwork(config)
        assert network.continuous_action is False
        assert hasattr(network, 'actor')
        assert not hasattr(network, 'actor_mean')
    
    def test_zero_dimensional_input(self, sample_config):
        """Test network behavior with edge case dimensions."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        config['network'] = {'input_dim': 1, 'hidden_dim': 2, 'output_dim': 1}
        
        network = PPONetwork(config)
        
        # Should still work with minimal dimensions
        states = torch.randn(1, 1)
        action_probs, values = network(states)
        
        assert action_probs.shape == (1, 1)
        assert values.shape == (1, 1)
    
    def test_large_batch_size(self, sample_config):
        """Test network with large batch size."""
        config = sample_config.copy()
        config['game'] = {'environment': 'cartpole'}
        config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 2}
        
        network = PPONetwork(config)
        
        # Large batch
        batch_size = 1000
        states = torch.randn(batch_size, 4)
        
        action_probs, values = network(states)
        
        assert action_probs.shape == (batch_size, 2)
        assert values.shape == (batch_size, 1)
        assert not torch.isnan(action_probs).any()
        assert not torch.isnan(values).any()


class TestPPONetworkParameterCount:
    """Test parameter counts for different network configurations."""
    
    def test_parameter_count_scaling(self, sample_config):
        """Test that parameter count scales appropriately with network size."""
        base_config = sample_config.copy()
        base_config['game'] = {'environment': 'cartpole'}
        
        # Small network
        small_config = base_config.copy()
        small_config['network'] = {'input_dim': 4, 'hidden_dim': 32, 'output_dim': 2}
        small_net = PPONetwork(small_config)
        small_params = sum(p.numel() for p in small_net.parameters())
        
        # Large network
        large_config = base_config.copy()
        large_config['network'] = {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 2}
        large_net = PPONetwork(large_config)
        large_params = sum(p.numel() for p in large_net.parameters())
        
        # Large network should have more parameters
        assert large_params > small_params
    
    def test_continuous_vs_discrete_parameters(self, sample_config):
        """Test parameter count difference between continuous and discrete networks."""
        # Discrete network
        discrete_config = sample_config.copy()
        discrete_config['game'] = {'environment': 'cartpole'}
        discrete_config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 2}
        discrete_net = PPONetwork(discrete_config)
        discrete_params = sum(p.numel() for p in discrete_net.parameters())
        
        # Continuous network (same dimensions)
        continuous_config = sample_config.copy()
        continuous_config['game'] = {'environment': 'pendulum'}
        continuous_config['network'] = {'input_dim': 4, 'hidden_dim': 64, 'output_dim': 2}
        continuous_net = PPONetwork(continuous_config)
        continuous_params = sum(p.numel() for p in continuous_net.parameters())
        
        # Continuous network has actor_log_std parameter but no discrete actor bias
        # Parameter counts should be similar but not identical
        assert abs(continuous_params - discrete_params) < discrete_params * 0.1  # Within 10%
