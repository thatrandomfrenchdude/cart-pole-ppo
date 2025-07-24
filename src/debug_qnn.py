#!/usr/bin/env python3
"""
Debug script to test QNN setup and diagnose issues
"""

import os
import sys
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qnn_setup():
    """Test QNN setup step by step"""
    print("🔍 QNN Setup Diagnostics")
    print("=" * 60)
    
    # Test 1: Check ONNX Runtime installation
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime version: {ort.__version__}")
    except ImportError as e:
        print(f"❌ ONNX Runtime not available: {e}")
        return
    
    # Test 2: Check available providers
    available_providers = ort.get_available_providers()
    print(f"📦 Available providers: {available_providers}")
    
    if 'QNNExecutionProvider' not in available_providers:
        print("❌ QNNExecutionProvider not available")
        print("💡 Make sure onnxruntime-qnn is installed: pip install onnxruntime-qnn")
        return
    else:
        print("✅ QNNExecutionProvider is available")
    
    # Test 3: Check QNN SDK paths
    qnn_paths = [
        "C:\\Users\\debeu\\Tools\\qairt\\2.36.0.250627\\lib\\aarch64-windows-msvc",
        "C:\\Users\\debeu\\Tools\\qairt\\2.36.0.250627\\lib\\arm64x-windows-msvc", 
        "C:\\Users\\debeu\\Tools\\qairt\\2.36.0.250627\\lib\\x86_64-windows-msvc",
        "C:\\Users\\debeu\\Tools\\qairt\\2.36.0.250627\\lib"
    ]
    
    valid_paths = []
    for path in qnn_paths:
        if os.path.exists(path):
            htp_dll = os.path.join(path, "QnnHtp.dll")
            cpu_dll = os.path.join(path, "QnnCpu.dll")
            if os.path.exists(htp_dll):
                print(f"✅ Found QNN HTP backend: {path}")
                valid_paths.append((path, "HTP"))
            elif os.path.exists(cpu_dll):
                print(f"✅ Found QNN CPU backend: {path}")
                valid_paths.append((path, "CPU"))
            else:
                print(f"⚠️ Path exists but no QNN DLLs found: {path}")
        else:
            print(f"❌ Path not found: {path}")
    
    if not valid_paths:
        print("❌ No valid QNN backend paths found")
        return
    
    # Test 4: Try different provider configurations
    for path, backend_type in valid_paths:
        print(f"\n🧪 Testing {backend_type} backend at: {path}")
        
        # Test without options
        try:
            session = ort.InferenceSession('example/model.onnx', providers=['QNNExecutionProvider', 'CPUExecutionProvider'])
            actual_providers = session.get_providers()
            print(f"   No options - Providers: {actual_providers}")
            if 'QNNExecutionProvider' in actual_providers:
                print(f"   ✅ {backend_type} backend working (no options)")
                return True
            else:
                print(f"   ❌ {backend_type} backend failed (no options)")
        except Exception as e:
            print(f"   ❌ {backend_type} backend error (no options): {e}")
        
        # Test with backend_path option
        try:
            qnn_options = {'backend_path': path}
            providers = [('QNNExecutionProvider', qnn_options), 'CPUExecutionProvider']
            session = ort.InferenceSession('example/model.onnx', providers=providers)
            actual_providers = session.get_providers()
            print(f"   With path - Providers: {actual_providers}")
            if 'QNNExecutionProvider' in actual_providers:
                print(f"   ✅ {backend_type} backend working (with path)")
                return True
            else:
                print(f"   ❌ {backend_type} backend failed (with path)")
        except Exception as e:
            print(f"   ❌ {backend_type} backend error (with path): {e}")
    
    print("\n❌ All QNN backend tests failed")
    return False

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists('example/model.onnx'):
        print("❌ example/model.onnx not found")
        sys.exit(1)
    
    success = test_qnn_setup()
    if success:
        print("\n🎉 QNN setup successful!")
    else:
        print("\n💡 Suggestions:")
        print("   1. Verify QNN AI Runtime SDK is properly installed")
        print("   2. Make sure all QNN DLLs are in your PATH")
        print("   3. Check if your hardware supports QNN acceleration")
        print("   4. Try running as administrator")
