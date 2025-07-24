#!/usr/bin/env python3
"""
Debug script to test QNN setup and diagnose issues
"""

import os
import sys
import platform
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_windows_arm64_snapdragon():
    """Check if running on Windows ARM64 with Snapdragon X Elite chip"""
    # Check if Windows
    if platform.system() != 'Windows':
        return False
    
    # Check if ARM64
    machine = platform.machine().lower()
    if machine not in ['arm64', 'aarch64']:
        return False
    
    # Check for Snapdragon X Elite processor
    try:
        # Try to get processor info from WMI
        import subprocess
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'name', '/format:list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        cpu_info = result.stdout.lower()
        # Check for Snapdragon X Elite indicators
        snapdragon_indicators = ['snapdragon', 'x elite', 'x1e']
        return any(indicator in cpu_info for indicator in snapdragon_indicators)
    except:
        # Fallback: if we can't determine, assume it's Snapdragon if it's Windows ARM64
        return True

def check_platform_compatibility():
    """Check if the current platform supports QNN"""
    system = platform.system()
    machine = platform.machine().lower()
    
    print("🔍 Platform Compatibility Check")
    print("=" * 60)
    print(f"Operating System: {system}")
    print(f"Architecture: {machine}")
    
    if system == "Darwin":  # macOS
        print("❌ QNN is not available on macOS")
        print("💡 QNN (Qualcomm Neural Network) SDK only supports Windows ARM64 with Snapdragon processors")
        return False
    elif system == "Linux":
        print("❌ QNN is not available on Linux")
        print("💡 QNN (Qualcomm Neural Network) SDK only supports Windows ARM64 with Snapdragon processors")
        return False
    elif system == "Windows":
        if machine in ['amd64', 'x86_64', 'i386', 'x86']:
            print("❌ QNN is not available on x86/x64 Windows")
            print("💡 QNN (Qualcomm Neural Network) SDK requires ARM64 architecture with Snapdragon processors")
            return False
        elif machine in ['arm64', 'aarch64']:
            if is_windows_arm64_snapdragon():
                print("✅ Platform compatible: Windows ARM64 with Snapdragon processor detected")
                return True
            else:
                print("❌ QNN requires Snapdragon X Elite processor")
                print("💡 QNN (Qualcomm Neural Network) SDK is optimized for Snapdragon X Elite chips")
                return False
        else:
            print(f"❌ Unknown Windows architecture: {machine}")
            return False
    else:
        print(f"❌ QNN is not available on {system}")
        print("💡 QNN (Qualcomm Neural Network) SDK only supports Windows ARM64 with Snapdragon processors")
        return False

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
    # First check platform compatibility
    if not check_platform_compatibility():
        print("\n🚫 QNN debugging skipped due to platform incompatibility")
        sys.exit(0)
    
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