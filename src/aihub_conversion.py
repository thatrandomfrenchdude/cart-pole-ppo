#!/usr/bin/env python3
"""
Enhanced AI Hub conversion script that creates multiple model formats:
- TorchScript (.pt) for PyTorch deployment
- ONNX (.onnx) for onnxruntime-qnn NPU acceleration
- Optionally compiles for Snapdragon X Elite via AI Hub
"""

import torch
import torch.onnx
import yaml
import os
import sys
import qai_hub as hub

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.network import PPONetwork

# Create a deployment-friendly wrapper
class PPODeploymentNetwork(torch.nn.Module):
    def __init__(self, ppo_network):
        super(PPODeploymentNetwork, self).__init__()
        self.ppo_network = ppo_network
    
    def forward(self, x):
        action_probs, _ = self.ppo_network(x)
        return action_probs

def load_config(path='config.yaml'):
    """Load configuration from YAML file"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {'network': {'input_dim': 4, 'hidden_dim': 128, 'output_dim': 2}}

def prepare_model(pth_path, config):
    """Load and prepare the PPO model for deployment"""
    print(f"📦 Loading model from: {pth_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    
    # Load the original PPO network
    ppo_model = PPONetwork(config)
    ppo_model.load_state_dict(checkpoint['network_state_dict'])
    ppo_model.eval()
    
    # Create deployment wrapper (only action predictions)
    model = PPODeploymentNetwork(ppo_model)
    model.eval()
    
    # Create example input
    example_input = torch.rand(1, config['network']['input_dim'])
    
    # Test the model
    with torch.no_grad():
        output = model(example_input)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {list(example_input.shape)}")
        print(f"   Output shape: {list(output.shape)}")
        print(f"   Sample prediction: {output[0].numpy()}")
    
    return model, example_input

def export_torchscript(model, example_input, path='model_deployment.pt'):
    """Export model to TorchScript format"""
    try:
        print(f"📄 Exporting to TorchScript: {path}")
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, path)
        
        # Verify the exported model works
        loaded_model = torch.jit.load(path)
        with torch.no_grad():
            test_output = loaded_model(example_input)
        
        print(f"✅ TorchScript export successful: {path}")
        return path
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        return None

def export_onnx(model, example_input, path='model_deployment.onnx'):
    """Export model to ONNX format for onnxruntime-qnn"""
    try:
        print(f"📄 Exporting to ONNX: {path}")
        torch.onnx.export(
            model,
            example_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model works
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        test_output = session.run(['output'], {'input': example_input.numpy()})
        
        print(f"✅ ONNX export successful: {path}")
        return path
    except ImportError:
        print("❌ ONNX export requires 'onnx' and 'onnxruntime' packages")
        print("   Install with: pip install onnx onnxruntime")
        return None
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def compile_for_snapdragon(model_path, input_dim, skip_on_error=True):
    """Compile model for Snapdragon X Elite (optional)"""
    print("🔧 Compiling for Snapdragon X Elite...")
    try:
        device = hub.Device("Snapdragon X Elite CRD")
        compile_job = hub.submit_compile_job(
            model=model_path,
            device=device,
            options="--target_runtime qnn_context_binary",
            input_specs={"input": (1, input_dim)}
        )
        
        print("⏳ Waiting for compilation (this may take several minutes)...")
        compile_job.wait()
        print("✔️ Compilation finished")
        
        compiled_model = compile_job.get_target_model()
        
        # Try to save compiled model locally
        try:
            compiled_path = "model_compiled_qnn.bin"
            compiled_model.seek(0)  # Reset stream position
            with open(compiled_path, 'wb') as f:
                f.write(compiled_model.read())
            print(f"💾 Saved compiled QNN model: {compiled_path}")
            return compiled_path
        except Exception as e:
            print(f"⚠️ Could not save compiled model locally: {e}")
            return "AI Hub Remote"
            
    except Exception as e:
        if skip_on_error:
            print(f"⚠️ Snapdragon compilation failed (skipping): {e}")
            print("   This requires valid AI Hub credentials")
            return None
        else:
            raise e

def main():
    """Main conversion pipeline"""
    print("🚀 PPO Cart-Pole Model Conversion Pipeline")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"✅ Config loaded: {config['network']}")
    
    # Prepare model
    model, example_input = prepare_model("example/model.pth", config)
    
    # Export to different formats
    results = {}
    
    # TorchScript export
    pt_path = export_torchscript(model, example_input)
    if pt_path:
        results['torchscript'] = pt_path
    
    # ONNX export
    onnx_path = export_onnx(model, example_input)
    if onnx_path:
        results['onnx'] = onnx_path
    
    # Optional: Compile for Snapdragon (may take time/require credentials)
    print("\n" + "-" * 40)
    compile_choice = input("Compile for Snapdragon X Elite? This requires AI Hub credentials and may take several minutes. (y/N): ").lower()
    
    if compile_choice in ['y', 'yes']:
        if pt_path:
            qnn_path = compile_for_snapdragon(pt_path, config['network']['input_dim'])
            if qnn_path:
                results['qnn'] = qnn_path
        else:
            print("⚠️ Cannot compile - TorchScript export failed")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Conversion Pipeline Complete!")
    print(f"{'='*60}")
    
    if results:
        print("📋 Generated Files:")
        for format_name, path in results.items():
            print(f"  • {format_name.upper()}: {path}")
        
        print(f"\n💡 Usage Instructions:")
        
        if 'onnx' in results:
            print(f"🔸 For NPU acceleration with onnxruntime-qnn:")
            print(f"   pip install onnxruntime-qnn")
            print(f"   python test_onnx_model.py")
        
        if 'torchscript' in results:
            print(f"🔸 For PyTorch deployment:")
            print(f"   python test_model.py")
        
        if 'qnn' in results:
            print(f"🔸 For direct QNN runtime (advanced):")
            print(f"   Use {results['qnn']} with Qualcomm QNN SDK")
    
    else:
        print("❌ No models were successfully exported")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)