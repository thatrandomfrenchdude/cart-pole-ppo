#!/usr/bin/env python3
"""
Test script to validate all model formats work correctly
"""

import sys
import os
import pytest

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model_loader import ExampleModeAgent
from src.config import load_config

def validate_model_format(model_path):
    """Validate a specific model format"""
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
        return False
    
    try:
        config = load_config()
        agent = ExampleModeAgent(model_path, config)
        
        format_info = agent.get_format_info()
          # Test a prediction
        test_state = [0.1, 0.5, 0.1, 0.2]
        action, log_prob, value = agent.select_action(test_state)
        
        # Validate the results - handle different numeric types
        import numpy as np
        
        # Action can be int, numpy int, or tensor - convert to int for validation
        action_int = int(action)
        assert action_int in [0, 1], f"Action should be 0 or 1, got {action_int}"
        
        # Log prob and value can be float, numpy float, or tensor
        log_prob_float = float(log_prob)
        value_float = float(value)
        
        assert isinstance(log_prob_float, float), "Log probability should be convertible to float"
        assert isinstance(value_float, float), "Value should be convertible to float"
        
        return True, format_info
        
    except Exception as e:
        pytest.fail(f"Error loading {model_path}: {e}")
        return False, None

@pytest.mark.parametrize("model_path", [
    "example/model.pth",
    "example/model.pt", 
    "example/model.onnx"
])
def test_model_formats(model_path):
    """Test that each model format works correctly"""
    success, format_info = validate_model_format(model_path)
    assert success, f"Model format {model_path} should work correctly"
    assert format_info is not None, f"Should get format info for {model_path}"

def test_all_formats_consistency():
    """Test that all model formats produce consistent results"""
    config = load_config()
    test_state = [0.1, 0.5, 0.1, 0.2]
    
    formats_to_test = [
        "example/model.pth",
        "example/model.pt", 
        "example/model.onnx"
    ]
    
    results = {}
    for model_path in formats_to_test:
        if os.path.exists(model_path):
            try:
                agent = ExampleModeAgent(model_path, config)
                action, log_prob, value = agent.select_action(test_state)
                # Convert to standard types for comparison
                results[model_path] = {
                    "action": int(action), 
                    "log_prob": float(log_prob)
                }
            except Exception as e:
                pytest.fail(f"Failed to test {model_path}: {e}")
    
    if len(results) >= 2:
        # Check that actions are consistent across formats
        actions = [r["action"] for r in results.values()]
        assert len(set(actions)) <= 1, "All formats should produce the same action"

# Legacy main function for standalone execution
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
        try:
            success, format_info = validate_model_format(model_path)
            results[model_path] = success
            if success:
                print(f"\n🧪 Testing: {model_path}")
                print(f"✅ Format: {format_info['format'].upper()}")
                print(f"   Supports value estimation: {format_info['supports_value_estimation']}")
        except Exception as e:
            print(f"\n🧪 Testing: {model_path}")
            print(f"❌ Error: {e}")
            results[model_path] = False
    
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
