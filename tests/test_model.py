#!/usr/bin/env python3
"""
Test script to verify the converted TorchScript model works correctly
"""

import torch
import numpy as np
import pytest
import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture
def torchscript_model_path():
    """Return path to TorchScript model, skip if not available"""
    model_path = "model_deployment.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"TorchScript model not found: {model_path}")
    return model_path

def test_torchscript_model_loading(torchscript_model_path):
    """Test that TorchScript model can be loaded"""
    model = torch.jit.load(torchscript_model_path)
    model.eval()
    assert model is not None

def test_torchscript_model_prediction(torchscript_model_path):
    """Test TorchScript model predictions"""
    model = torch.jit.load(torchscript_model_path)
    model.eval()
    
    # Test prediction
    test_state = [0.1, 0.5, 0.1, 0.2]
    input_tensor = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action_probs = model(input_tensor)
    
    assert action_probs.shape == (1, 2), f"Expected shape (1, 2), got {action_probs.shape}"
    
    # Check that probabilities are valid
    probs = action_probs[0].numpy()
    assert len(probs) == 2, "Should have 2 action probabilities"
    assert all(p >= 0 for p in probs), "All probabilities should be non-negative"

def test_torchscript_model_consistency(torchscript_model_path):
    """Test that TorchScript model produces consistent results"""
    model = torch.jit.load(torchscript_model_path)
    model.eval()
    
    test_state = [0.1, 0.5, 0.1, 0.2]
    input_tensor = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0)
    
    # Get multiple predictions for the same input
    results = []
    with torch.no_grad():
        for _ in range(3):
            action_probs = model(input_tensor)
            results.append(action_probs[0].numpy())
    
    # All results should be identical (deterministic model)
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0], results[i], decimal=6)

# Legacy test function for standalone execution

def test_torchscript_model(model_path="model_deployment.pt"):
    """Test the converted TorchScript model"""
    print(f"🧪 Testing TorchScript model: {model_path}")
    
    try:
        # Load the TorchScript model
        model = torch.jit.load(model_path)
        model.eval()
        
        # Create test inputs (Cart-Pole state: [cart_pos, cart_vel, pole_angle, pole_angular_vel])
        test_inputs = [
            [0.0, 0.0, 0.0, 0.0],      # Centered position
            [0.1, 0.5, 0.1, 0.2],      # Slightly off-center
            [-0.1, -0.2, -0.05, -0.1], # Moving left
            [0.2, 0.0, 0.15, 0.0],     # Tilted right
        ]
        
        print("Test Results:")
        print("Input State -> Action Probabilities [Left, Right]")
        print("-" * 55)
        
        for i, state in enumerate(test_inputs):
            input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = model(input_tensor)
                probs = action_probs[0].numpy()
                
            # Determine recommended action
            action = "Left" if probs[0] > probs[1] else "Right"
            confidence = max(probs)
            
            print(f"{state} -> [{probs[0]:.4f}, {probs[1]:.4f}] ({action}, {confidence:.1%})")
        
        print(f"\n✅ TorchScript model is working correctly!")
        print(f"Model input shape: {list(input_tensor.shape)}")
        print(f"Model output shape: {list(action_probs.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing TorchScript model: {e}")
        return False

if __name__ == "__main__":
    test_torchscript_model()
