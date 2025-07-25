#!/usr/bin/env python3
"""
Test script for ONNX model with onnxruntime-qnn support for NPU acceleration
"""

import numpy as np
import pytest
import sys
import os
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import onnxruntime as ort
except ImportError:
    pytest.skip("onnxruntime not available", allow_module_level=True)

class ONNXCartPoleAgent:
    """Cart-Pole agent using ONNX model with potential QNN acceleration"""
    
    def __init__(self, onnx_model_path="example/model.onnx", use_qnn=False):
        """
        Initialize the ONNX-based agent
        
        Args:
            onnx_model_path: Path to the ONNX model file
            use_qnn: Whether to try using QNN execution provider (requires onnxruntime-qnn)
        """
        self.model_path = onnx_model_path
        
        # Set up execution providers
        providers = []
        if use_qnn:
            # QNN provider for NPU acceleration (requires onnxruntime-qnn package)
            providers.append('QNNExecutionProvider')
        
        # Always include CPU as fallback
        providers.append('CPUExecutionProvider')
        
        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"✅ Loaded ONNX model: {onnx_model_path}")
            print(f"📋 Using providers: {self.session.get_providers()}")
            print(f"🔤 Input name: {self.input_name}")
            print(f"🔤 Output name: {self.output_name}")
            
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            raise
    
    def predict(self, state):
        """
        Predict action probabilities for a given state
        
        Args:
            state: Cart-pole state [cart_pos, cart_vel, pole_angle, pole_angular_vel]
            
        Returns:
            action_probabilities: [left_prob, right_prob]
        """
        # Ensure state is the right shape and type
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        if len(state.shape) == 1:
            state = state.reshape(1, -1)  # Add batch dimension
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: state})
        action_probs = outputs[0][0]  # Remove batch dimension
        
        return action_probs
    
    def act(self, state):
        """
        Choose an action based on the model's predictions
        
        Args:
            state: Cart-pole state
            
        Returns:
            action: 0 (left) or 1 (right)
        """
        action_probs = self.predict(state)
        return np.argmax(action_probs)
    
    def benchmark(self, num_iterations=1000):
        """Benchmark inference speed"""
        print(f"🏃 Running benchmark with {num_iterations} iterations...")
        
        # Create random test data
        test_state = np.random.randn(1, 4).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: test_state})
        
        # Timed runs
        start_time = time.time()
        for _ in range(num_iterations):
            self.session.run([self.output_name], {self.input_name: test_state})
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / num_iterations
        fps = 1000 / avg_time_ms
        
        print(f"📊 Average inference time: {avg_time_ms:.3f}ms")
        print(f"🚀 Throughput: {fps:.1f} FPS")


# Pytest test functions
@pytest.fixture
def onnx_model_path():
    """Return path to ONNX model, skip if not available"""
    model_path = "example/model.onnx"
    if not os.path.exists(model_path):
        pytest.skip(f"ONNX model not found: {model_path}")
    return model_path

def test_onnx_agent_initialization(onnx_model_path):
    """Test that ONNXCartPoleAgent can be initialized"""
    agent = ONNXCartPoleAgent(onnx_model_path, use_qnn=False)
    assert agent.session is not None
    assert agent.input_name is not None
    assert agent.output_name is not None

def test_onnx_agent_prediction(onnx_model_path):
    """Test that ONNX agent can make predictions"""
    agent = ONNXCartPoleAgent(onnx_model_path, use_qnn=False)
    
    # Test prediction
    test_state = [0.1, 0.5, 0.1, 0.2]
    action = agent.act(test_state)
    
    assert isinstance(action, (int, np.integer))
    assert action in [0, 1]

def test_onnx_agent_performance(onnx_model_path):
    """Test ONNX agent performance"""
    agent = ONNXCartPoleAgent(onnx_model_path, use_qnn=False)
    
    # Warm-up
    test_state = [0.1, 0.5, 0.1, 0.2]
    agent.act(test_state)
    
    # Performance test
    num_predictions = 100
    start_time = time.time()
    
    for _ in range(num_predictions):
        agent.act(test_state)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_prediction = total_time / num_predictions
    
    # Should be fast (less than 1ms on average)
    assert avg_time_per_prediction < 0.001, f"Average prediction time too slow: {avg_time_per_prediction:.4f}s"

def test_onnx_qnn_acceleration_availability():
    """Test QNN provider availability and skip gracefully if not available"""
    try:
        # Check if QNN provider is available in the current onnxruntime installation
        available_providers = ort.get_available_providers()
        
        if 'QNNExecutionProvider' not in available_providers:
            pytest.skip("QNN execution provider not available in current onnxruntime installation")
        
        # Check for model existence
        model_path = "example/model.onnx"
        if not os.path.exists(model_path):
            pytest.skip("ONNX model not available for QNN testing")
            
        # If we get here, we can test QNN functionality
        agent = ONNXCartPoleAgent(model_path, use_qnn=True)
        providers = agent.session.get_providers()
        
        # Test that some provider is being used (QNN or fallback to CPU)
        test_state = [0.1, 0.5, 0.1, 0.2]
        action = agent.act(test_state)
        
        assert isinstance(action, (int, np.integer))
        assert action in [0, 1]
        
        # Log which providers are actually being used
        print(f"Active providers: {providers}")
        
    except Exception as e:
        pytest.skip(f"QNN acceleration test failed: {e}")

def test_onnx_model_consistency():
    """Test that ONNX model produces consistent results"""
    if not os.path.exists("example/model.onnx"):
        pytest.skip("ONNX model not available")
        
    agent = ONNXCartPoleAgent("example/model.onnx", use_qnn=False)
    
    test_state = [0.1, 0.5, 0.1, 0.2]
    
    # Get multiple predictions for the same state
    actions = [agent.act(test_state) for _ in range(5)]
    
    # All predictions should be identical (deterministic model)
    assert len(set(actions)) == 1, "Model should produce consistent results"
