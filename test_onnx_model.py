#!/usr/bin/env python3
"""
Example script showing how to use the converted ONNX model with onnxruntime-qnn
for NPU acceleration on Snapdragon X Elite
"""

import numpy as np
import onnxruntime as ort
import time

class ONNXCartPoleAgent:
    """Cart-Pole agent using ONNX model with potential QNN acceleration"""
    
    def __init__(self, onnx_model_path="model_deployment.onnx", use_qnn=False):
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


def test_onnx_model():
    """Test the ONNX model functionality"""
    print("🧪 Testing ONNX Cart-Pole Agent")
    print("=" * 50)
    
    try:
        # Try with QNN first (will fall back to CPU if not available)
        print("🔧 Attempting to load with QNN provider...")
        agent = ONNXCartPoleAgent(use_qnn=True)
        
        # Test with some example states
        test_states = [
            [0.0, 0.0, 0.0, 0.0],      # Centered
            [0.1, 0.5, 0.1, 0.2],      # Moving right
            [-0.1, -0.2, -0.05, -0.1], # Moving left
            [0.2, 0.0, 0.15, 0.0],     # Tilted right
        ]
        
        print("\n🎯 Testing predictions:")
        print("State -> Action Probs -> Recommended Action")
        print("-" * 55)
        
        for state in test_states:
            probs = agent.predict(state)
            action = agent.act(state)
            action_name = "Left" if action == 0 else "Right"
            confidence = probs[action] * 100
            
            print(f"{state} -> [{probs[0]:.3f}, {probs[1]:.3f}] -> {action_name} ({confidence:.1f}%)")
        
        # Run benchmark
        print(f"\n{'-'*50}")
        agent.benchmark(100)  # Reduced iterations for quicker test
        
        return True
        
    except FileNotFoundError:
        print("❌ ONNX model file not found. Please run the conversion script first.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def usage_example():
    """Show how to integrate into a game loop"""
    print(f"\n{'-'*50}")
    print("💡 Usage Example in Game Loop:")
    print("""
# Initialize agent
agent = ONNXCartPoleAgent("model_deployment.onnx", use_qnn=True)

# In your game loop:
while game_running:
    # Get current state from environment
    state = env.get_state()  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    
    # Get action from model
    action = agent.act(state)  # 0=left, 1=right
    
    # Apply action to environment
    env.step(action)
    
    # Update display, etc.
    ...
""")


if __name__ == "__main__":
    success = test_onnx_model()
    if success:
        usage_example()
    else:
        print("\n💡 To generate the ONNX model, run:")
        print("python src/aihub_conversion.py")
