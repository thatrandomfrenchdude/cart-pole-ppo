# Model Deployment Guide

This guide explains how to use the converted PPO Cart-Pole model in different formats for optimal performance on Snapdragon X Elite.

## 🗂️ Generated Model Files

After running the conversion script, you'll have:

- **`model_deployment.pt`** - TorchScript format (PyTorch deployment)
- **`model_deployment.onnx`** - ONNX format (cross-platform, NPU-ready)
- **`model_compiled_qnn.bin`** - Compiled QNN binary (if AI Hub compilation succeeded)

## 🚀 NPU Acceleration with onnxruntime-qnn

For best performance on Snapdragon X Elite, use the ONNX model with onnxruntime-qnn:

### Installation

```bash
# Install onnxruntime-qnn for NPU support
pip install onnxruntime-qnn --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-qnn/pypi/simple/

# Alternative: Download from Qualcomm Developer Portal
# https://developer.qualcomm.com/software/qualcomm-ai-engine-direct
```

### Usage Example

```python
import numpy as np
import onnxruntime as ort

# Create session with QNN provider for NPU acceleration
providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model_deployment.onnx', providers=providers)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
def predict_action(state):
    """
    Predict action for Cart-Pole game
    
    Args:
        state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    
    Returns:
        action: 0 (move left) or 1 (move right)
    """
    # Prepare input
    input_data = np.array(state, dtype=np.float32).reshape(1, -1)
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    action_probs = outputs[0][0]
    
    # Return action with highest probability
    return np.argmax(action_probs)

# Example usage
game_state = [0.1, 0.5, 0.1, 0.2]  # Example Cart-Pole state
action = predict_action(game_state)
print(f"Recommended action: {'Left' if action == 0 else 'Right'}")
```

### Testing

Run the provided test script:

```bash
python test_onnx_model.py
```

## 🐍 PyTorch Deployment

For PyTorch-based applications, use the TorchScript model:

```python
import torch

# Load the TorchScript model
model = torch.jit.load('model_deployment.pt')
model.eval()

def predict_action(state):
    """Predict action using TorchScript model"""
    input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action_probs = model(input_tensor)
    
    return torch.argmax(action_probs).item()

# Test
game_state = [0.1, 0.5, 0.1, 0.2]
action = predict_action(game_state)
```

## 🎯 Performance Comparison

Here's what you can expect:

| Format | Hardware | Avg Inference Time | Use Case |
|--------|----------|-------------------|----------|
| ONNX + QNN | Snapdragon X Elite NPU | ~0.1ms | Production deployment |
| ONNX + CPU | Any CPU | ~0.03ms | Development/testing |
| TorchScript | Any CPU/GPU | ~0.05ms | PyTorch ecosystem |

## 🔧 Integration Examples

### Game Loop Integration

```python
import onnxruntime as ort
import numpy as np

class CartPoleAI:
    def __init__(self, model_path='model_deployment.onnx'):
        # Initialize with NPU acceleration if available
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def get_action(self, observation):
        """Get action for current game state"""
        input_data = np.array(observation, dtype=np.float32).reshape(1, -1)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return np.argmax(outputs[0][0])

# In your game loop:
ai_player = CartPoleAI()

while game_running:
    # Get current state: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    current_state = environment.get_state()
    
    # Get AI action
    action = ai_player.get_action(current_state)
    
    # Apply action (0=left, 1=right)
    environment.step(action)
    
    # Update display
    render_game()
```

### Web API Deployment

```python
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# Load model once at startup
session = ort.InferenceSession('model_deployment.onnx', 
                              providers=['QNNExecutionProvider', 'CPUExecutionProvider'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    state = data['state']  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    
    input_data = np.array(state, dtype=np.float32).reshape(1, -1)
    outputs = session.run(['output'], {'input': input_data})
    action = int(np.argmax(outputs[0][0]))
    confidence = float(outputs[0][0][action])
    
    return jsonify({
        'action': action,
        'confidence': confidence,
        'action_name': 'Left' if action == 0 else 'Right'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## ⚡ Performance Tips

1. **Use ONNX + QNN** for best performance on Snapdragon X Elite
2. **Batch processing** if you have multiple states to process
3. **Warm-up runs** - first few inferences may be slower
4. **Memory management** - reuse input arrays when possible

## 🔍 Troubleshooting

### QNN Provider Not Available
If you see "QNNExecutionProvider not available":
- Install `onnxruntime-qnn` package
- Ensure you're running on Snapdragon X Elite hardware
- Check Qualcomm QNN SDK installation

### Model Loading Errors
- Verify model file paths are correct
- Check that input shapes match expected format [1, 4]
- Ensure numpy arrays are float32 type

### Performance Issues
- Use QNN provider for NPU acceleration
- Avoid creating new sessions for each prediction
- Consider model quantization for further optimization

## 📚 Additional Resources

- [Qualcomm AI Engine Direct Documentation](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct)
- [ONNX Runtime QNN Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Snapdragon X Elite AI Performance Guide](https://developer.qualcomm.com/hardware/snapdragon-x-elite)
