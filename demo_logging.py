#!/usr/bin/env python3
"""
Demo script showing the intelligent logging behavior of the PPO Cart-Pole implementation.

This script demonstrates:
1. Fresh start creates a new log file (overwrite mode)
2. Resume appends to existing log file (append mode)  
3. Complete training history is preserved across interruptions
"""

import os
import subprocess
import signal
import time

def main():
    print("🎯 PPO Cart-Pole Intelligent Logging Demo")
    print("=" * 50)
    
    # Clean slate
    if os.path.exists('models/ppo_cartpole.pth'):
        os.remove('models/ppo_cartpole.pth')
        print("📁 Removed existing model")
    
    if os.path.exists('training.log'):
        os.remove('training.log')
        print("📄 Removed existing log")
    
    print("\n1️⃣ FRESH START TEST")
    print("   - No existing model")
    print("   - Should create new log file (overwrite mode)")
    
    # Run fresh training
    proc = subprocess.Popen(['python', 'main.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    proc.send_signal(signal.SIGINT)
    proc.wait()
    
    # Check results
    model_exists = os.path.exists('models/ppo_cartpole.pth')
    log_exists = os.path.exists('training.log')
    
    print(f"   ✅ Model created: {model_exists}")
    print(f"   ✅ Log created: {log_exists}")
    
    if log_exists:
        with open('training.log', 'r') as f:
            content = f.read()
            fresh_start = "PPO Cart-Pole Training Session Started" in content
            overwrite_mode = "overwrite mode (fresh start)" in content
            print(f"   ✅ Fresh start logged: {fresh_start}")
            print(f"   ✅ Overwrite mode logged: {overwrite_mode}")
    
    print("\n2️⃣ RESUME TEST")
    print("   - Existing model found")
    print("   - Should append to existing log file")
    
    # Run resume training
    proc2 = subprocess.Popen(['python', 'main.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    proc2.send_signal(signal.SIGINT)
    proc2.wait()
    
    # Check results
    if log_exists:
        with open('training.log', 'r') as f:
            content = f.read()
            resumed = "PPO Cart-Pole Training Session RESUMED" in content
            append_mode = "append mode (resuming training)" in content
            resuming_training = "Resuming training from episode" in content
            
            print(f"   ✅ Resume session logged: {resumed}")
            print(f"   ✅ Append mode logged: {append_mode}")
            print(f"   ✅ Training continuation logged: {resuming_training}")
            
            # Count sessions
            session_count = content.count("Training Session")
            print(f"   ✅ Total sessions in log: {session_count}")
            
            # Show log size
            lines = len(content.strip().split('\n'))
            print(f"   📊 Log file size: {lines} lines")
    
    print("\n🎉 SUMMARY")
    print("   - Fresh start: Creates new log file")
    print("   - Resume: Appends to existing log file")
    print("   - Complete training history preserved")
    print("   - Clear session boundaries for navigation")

if __name__ == "__main__":
    main()
