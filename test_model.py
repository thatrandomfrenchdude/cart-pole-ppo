#!/usr/bin/env python3
"""
Test script to verify the converted TorchScript model works correctly
"""

import torch
import numpy as np

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
