#!/usr/bin/env python3
"""
Test script to validate all model formats work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.model_loader import ExampleModeAgent
from src.config import load_config

def test_model_format(model_path):
    """Test a specific model format"""
    print(f"\n🧪 Testing: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return False
    
    try:
        config = load_config()
        agent = ExampleModeAgent(model_path, config)
        
        format_info = agent.get_format_info()
        print(f"✅ Format: {format_info['format'].upper()}")
        print(f"   Supports value estimation: {format_info['supports_value_estimation']}")
        
        # Test a prediction
        test_state = [0.1, 0.5, 0.1, 0.2]
        action, log_prob, value = agent.select_action(test_state)
        print(f"   Test prediction: action={action}, log_prob={log_prob:.3f}, value={value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return False

def main():
    print("🚀 Multi-Format Model Validation")
    print("=" * 50)
    
    # Test all formats
    formats_to_test = [
        "example/model.pth",
        "example/model.pt", 
        "example/model.onnx"
    ]
    
    results = {}
    for model_path in formats_to_test:
        results[model_path] = test_model_format(model_path)
    
    # Summary
    print(f"\n📊 Test Results:")
    print("=" * 50)
    
    success_count = 0
    for model_path, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {model_path}")
        if success:
            success_count += 1
    
    print(f"\n🎯 Summary: {success_count}/{len(formats_to_test)} formats working")
    
    if success_count == len(formats_to_test):
        print("🎉 All model formats are working correctly!")
        return True
    else:
        print("⚠️ Some model formats have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
