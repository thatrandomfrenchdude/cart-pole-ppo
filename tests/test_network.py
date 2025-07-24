"""
Tests for the PPO neural network module.
"""
import torch
import pytest
from network import PPONetwork


class TestPPONetwork:
    
    def test_initialization(self, sample_config):
        """Test network initialization with config."""
        network = PPONetwork(sample_config)
        
        # Check that network components exist
        assert hasattr(network, 'shared')
        assert hasattr(network, 'actor')
        assert hasattr(network, 'critic')
        
        # Check layer dimensions
        net_config = sample_config['network']
        assert network.actor.in_features == net_config['hidden_dim']
        assert network.actor.out_features == net_config['output_dim']
        assert network.critic.in_features == net_config['hidden_dim']
        assert network.critic.out_features == 1
    
    def test_forward_pass_shape(self, sample_config):
        """Test forward pass returns correct shapes."""
        network = PPONetwork(sample_config)
        
        # Create sample input
        batch_size = 5
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        action_probs, state_value = network(x)
        
        # Check output shapes
        expected_output_dim = sample_config['network']['output_dim']
        assert action_probs.shape == (batch_size, expected_output_dim)
        assert state_value.shape == (batch_size, 1)
    
    def test_forward_pass_single_input(self, sample_config):
        """Test forward pass with single input."""
        network = PPONetwork(sample_config)
        
        # Single input
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(1, input_dim)
        
        action_probs, state_value = network(x)
        
        # Check shapes
        output_dim = sample_config['network']['output_dim']
        assert action_probs.shape == (1, output_dim)
        assert state_value.shape == (1, 1)
    
    def test_action_probabilities_valid(self, sample_config):
        """Test that action probabilities are valid."""
        network = PPONetwork(sample_config)
        
        # Create sample input
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(3, input_dim)
        
        action_probs, _ = network(x)
        
        # Check that probabilities sum to 1 (approximately)
        prob_sums = torch.sum(action_probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones(3), atol=1e-6)
        
        # Check that all probabilities are non-negative
        assert torch.all(action_probs >= 0)
    
    def test_state_value_range(self, sample_config):
        """Test that state values are reasonable."""
        network = PPONetwork(sample_config)
        
        # Create sample input
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(10, input_dim)
        
        _, state_values = network(x)
        
        # State values should be finite
        assert torch.all(torch.isfinite(state_values))
    
    def test_gradient_flow(self, sample_config):
        """Test that gradients flow through the network."""
        network = PPONetwork(sample_config)
        
        # Create sample input and target
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(2, input_dim)
        
        action_probs, state_values = network(x)
        
        # Create dummy loss
        loss = torch.sum(action_probs) + torch.sum(state_values)
        loss.backward()
        
        # Check that gradients exist
        for param in network.parameters():
            assert param.grad is not None
    
    def test_network_parameters(self, sample_config):
        """Test network parameter count and types."""
        network = PPONetwork(sample_config)
        
        # Check that parameters exist
        params = list(network.parameters())
        assert len(params) > 0
        
        # All parameters should be torch tensors
        for param in params:
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad
    
    def test_shared_layers(self, sample_config):
        """Test shared layer structure."""
        network = PPONetwork(sample_config)
        
        # Shared layers should be a Sequential module
        assert isinstance(network.shared, torch.nn.Sequential)
        
        # Check layer types in shared network
        layers = list(network.shared.children())
        assert len(layers) == 4  # Linear, ReLU, Linear, ReLU
        assert isinstance(layers[0], torch.nn.Linear)
        assert isinstance(layers[1], torch.nn.ReLU)
        assert isinstance(layers[2], torch.nn.Linear)
        assert isinstance(layers[3], torch.nn.ReLU)
    
    def test_different_config_dimensions(self):
        """Test network with different configuration dimensions."""
        config = {
            'game': {'environment': 'acrobot'},
            'network': {
                'input_dim': 6,
                'hidden_dim': 64,
                'output_dim': 3
            }
        }
        
        network = PPONetwork(config)
        
        # Test forward pass with new dimensions
        x = torch.randn(1, 6)
        action_probs, state_value = network(x)
        
        assert action_probs.shape == (1, 3)
        assert state_value.shape == (1, 1)
    
    def test_eval_mode(self, sample_config):
        """Test network in evaluation mode."""
        network = PPONetwork(sample_config)
        
        # Switch to eval mode
        network.eval()
        
        input_dim = sample_config['network']['input_dim']
        x = torch.randn(1, input_dim)
        
        with torch.no_grad():
            action_probs, state_value = network(x)
            
            # Should still produce valid outputs
            assert action_probs.shape[1] == sample_config['network']['output_dim']
            assert state_value.shape == (1, 1)
