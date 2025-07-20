#!/usr/bin/env python3
"""
Simple test script to verify all modules are working correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from src.config import load_config
        print("✅ src.config imported successfully")
        
        from src.utils import setup_logging
        print("✅ src.utils imported successfully")
        
        from src.environment import CartPoleEnv
        print("✅ src.environment imported successfully")
        
        from src.network import PPONetwork
        print("✅ src.network imported successfully")
        
        from src.agent import PPOAgent
        print("✅ src.agent imported successfully")
        
        from src.training import training_loop, example_mode_loop
        print("✅ src.training imported successfully")
        
        from src.web_server import create_app
        print("✅ src.web_server imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    try:
        from src.config import load_config
        config = load_config()
        
        required_sections = ['environment', 'network', 'ppo', 'training', 'server', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        print("✅ Configuration loaded and validated successfully")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_environment():
    """Test environment creation."""
    print("\nTesting environment creation...")
    try:
        from src.config import load_config
        from src.environment import CartPoleEnv
        
        config = load_config()
        env = CartPoleEnv(config)
        
        # Test reset
        state = env.reset()
        if len(state) != 4:
            raise ValueError(f"Expected state length 4, got {len(state)}")
        
        # Test step
        next_state, reward, done = env.step(0)
        if len(next_state) != 4:
            raise ValueError(f"Expected next_state length 4, got {len(next_state)}")
        
        print("✅ Environment created and tested successfully")
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_agent():
    """Test agent creation."""
    print("\nTesting agent creation...")
    try:
        from src.config import load_config
        from src.agent import PPOAgent
        
        config = load_config()
        agent = PPOAgent(config)
        
        # Test action selection with dummy state
        import numpy as np
        dummy_state = np.array([0.0, 0.0, 0.0, 0.0])
        action, log_prob, value = agent.select_action(dummy_state)
        
        if not isinstance(action, int) or action not in [0, 1]:
            raise ValueError(f"Expected action to be 0 or 1, got {action}")
        
        print("✅ Agent created and tested successfully")
        return True
    except Exception as e:
        print(f"❌ Agent error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("CART-POLE PPO MODULE TESTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_environment,
        test_agent
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("The modular reorganization was successful!")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total} passed)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
