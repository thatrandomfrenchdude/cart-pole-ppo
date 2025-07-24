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
    model_path = "example/model.pt"
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
