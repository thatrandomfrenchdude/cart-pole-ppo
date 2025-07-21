#!/usr/bin/env python3
"""
Compare performance and usage between different model formats:
- Original .pth checkpoint (PyTorch training format)
- TorchScript .pt (PyTorch deployment format) 
- ONNX .onnx (Cross-platform deployment format)
"""

import time
import torch
import numpy as np
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from network import PPONetwork

def load_original_model():
    """Load the original PyTorch model from checkpoint"""
    print("📦 Loading original PyTorch model (.pth)...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load('example/model.pth', map_location='cpu')
    
    # Create and load model
    model = PPONetwork(config)
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    
    def predict(state):
        """Predict using original model (returns both action_probs and value)"""
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value = model(input_tensor)
        return action_probs[0].numpy(), state_value[0].item()
    
    return predict

def load_torchscript_model():
    """Load TorchScript model"""
    print("📦 Loading TorchScript model (.pt)...")
    
    model = torch.jit.load('model_deployment.pt')
    model.eval()
    
    def predict(state):
        """Predict using TorchScript model (action_probs only)"""
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = model(input_tensor)
        return action_probs[0].numpy()
    
    return predict

def load_onnx_model():
    """Load ONNX model"""
    print("📦 Loading ONNX model (.onnx)...")
    
    try:
        import onnxruntime as ort
        
        # Try QNN provider first, fall back to CPU
        providers = []
        try:
            # This will only work if onnxruntime-qnn is installed
            providers.append('QNNExecutionProvider')
        except:
            pass
        providers.append('CPUExecutionProvider')
        
        session = ort.InferenceSession('model_deployment.onnx', providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"   Using providers: {session.get_providers()}")
        
        def predict(state):
            """Predict using ONNX model"""
            input_data = np.array(state, dtype=np.float32).reshape(1, -1)
            outputs = session.run([output_name], {input_name: input_data})
            return outputs[0][0]
        
        return predict
        
    except ImportError:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
        return None

def benchmark_model(predict_func, name, num_runs=1000):
    """Benchmark a prediction function"""
    print(f"\n🏃 Benchmarking {name} ({num_runs} runs)...")
    
    # Create test data
    test_states = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.5, 0.1, 0.2],
        [-0.1, -0.2, -0.05, -0.1],
        [0.2, 0.0, 0.15, 0.0],
    ]
    
    # Warm up
    for state in test_states[:2]:
        predict_func(state)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        for state in test_states:
            result = predict_func(state)
    end_time = time.time()
    
    total_predictions = num_runs * len(test_states)
    avg_time_ms = (end_time - start_time) * 1000 / total_predictions
    fps = 1000 / avg_time_ms
    
    print(f"   Average inference: {avg_time_ms:.3f}ms")
    print(f"   Throughput: {fps:.0f} predictions/second")
    
    return avg_time_ms, fps

def compare_predictions():
    """Compare prediction outputs between different formats"""
    print(f"\n🔍 Comparing Predictions")
    print("=" * 60)
    
    # Test states
    test_states = [
        [0.0, 0.0, 0.0, 0.0],      # Balanced
        [0.1, 0.5, 0.1, 0.2],      # Leaning right, moving right
        [-0.1, -0.2, -0.05, -0.1], # Leaning left, moving left
        [0.2, 0.0, 0.15, 0.0],     # Far right, tilted right
    ]
    
    # Load models
    original_predict = load_original_model()
    torchscript_predict = load_torchscript_model()
    onnx_predict = load_onnx_model()
    
    print(f"\nState -> [Left_Prob, Right_Prob] -> Action")
    print("-" * 60)
    
    for i, state in enumerate(test_states):
        print(f"\nTest {i+1}: {state}")
        
        # Original model (includes state value)
        action_probs, state_value = original_predict(state)
        action = "Left" if action_probs[0] > action_probs[1] else "Right"
        print(f"  Original:    [{action_probs[0]:.3f}, {action_probs[1]:.3f}] -> {action} (value: {state_value:.3f})")
        
        # TorchScript model
        action_probs = torchscript_predict(state)
        action = "Left" if action_probs[0] > action_probs[1] else "Right"
        print(f"  TorchScript: [{action_probs[0]:.3f}, {action_probs[1]:.3f}] -> {action}")
        
        # ONNX model
        if onnx_predict:
            action_probs = onnx_predict(state)
            action = "Left" if action_probs[0] > action_probs[1] else "Right"
            print(f"  ONNX:        [{action_probs[0]:.3f}, {action_probs[1]:.3f}] -> {action}")

def main():
    """Main comparison function"""
    print("🚀 PPO Cart-Pole Model Format Comparison")
    print("=" * 60)
    
    # Check which files exist
    files_status = {
        'Original (.pth)': os.path.exists('example/model.pth'),
        'TorchScript (.pt)': os.path.exists('example/model.pt'),
        'ONNX (.onnx)': os.path.exists('example/model.onnx'),
    }
    
    print("📂 Available Model Files:")
    for format_name, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {format_name}")
    
    if not all(files_status.values()):
        print(f"\n💡 To generate missing models, run: python convert_model.py")
        if not files_status['ONNX (.onnx)']:
            print("   For ONNX support, install: pip install onnx onnxruntime")
    
    # Compare predictions if all models available
    if files_status['Original (.pth)'] and files_status['TorchScript (.pt)']:
        compare_predictions()
    
    # Benchmark performance
    print(f"\n🏃 Performance Benchmark")
    print("=" * 60)
    
    results = {}
    
    if files_status['Original (.pth)']:
        original_predict = load_original_model()
        avg_time, fps = benchmark_model(original_predict, "Original PyTorch", 500)
        results['Original'] = {'time': avg_time, 'fps': fps}
    
    if files_status['TorchScript (.pt)']:
        torchscript_predict = load_torchscript_model()
        avg_time, fps = benchmark_model(torchscript_predict, "TorchScript", 500)
        results['TorchScript'] = {'time': avg_time, 'fps': fps}
    
    if files_status['ONNX (.onnx)']:
        onnx_predict = load_onnx_model()
        if onnx_predict:
            avg_time, fps = benchmark_model(onnx_predict, "ONNX", 500)
            results['ONNX'] = {'time': avg_time, 'fps': fps}
    
    # Summary
    if results:
        print(f"\n📊 Performance Summary")
        print("=" * 60)
        print(f"{'Format':<12} {'Avg Time (ms)':<15} {'Throughput (FPS)':<20}")
        print("-" * 60)
        
        for format_name, metrics in results.items():
            print(f"{format_name:<12} {metrics['time']:<15.3f} {metrics['fps']:<20.0f}")
        
        # Find fastest
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]['time']:.3f}ms avg)")
    
    print(f"\n💡 Usage Recommendations:")
    print("   • For development/testing: Any format works")
    print("   • For PyTorch deployment: Use TorchScript (.pt)")  
    print("   • For NPU acceleration: Use ONNX (.onnx) with onnxruntime-qnn")
    print("   • For cross-platform: Use ONNX (.onnx)")

if __name__ == "__main__":
    main()
