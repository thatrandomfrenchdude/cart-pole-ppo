"""
Multi-format model loader for PPO Cart-Pole models.
Supports .pth, .pt (TorchScript), and .onnx formats.
"""

import torch
import numpy as np
import os
import logging

try:
    from .network import PPONetwork
except ImportError:
    from network import PPONetwork

logger = logging.getLogger(__name__)


class ModelLoader:
    """Unified loader for different model formats"""
    
    @staticmethod
    def detect_format(model_path):
        """Detect the format of a model file"""
        if not os.path.exists(model_path):
            return None
            
        ext = os.path.splitext(model_path)[1].lower()
        if ext == '.pth':
            return 'pth'
        elif ext == '.pt':
            return 'torchscript'
        elif ext == '.onnx':
            return 'onnx'
        else:
            return None
    
    @staticmethod
    def load_pth_model(model_path, config):
        """Load original PyTorch checkpoint (.pth)"""
        logger.info(f"Loading PyTorch checkpoint: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = PPONetwork(config)
        model.load_state_dict(checkpoint['network_state_dict'])
        model.eval()
        
        def predict(state):
            """Predict using original model (returns both action_probs and value)"""
            input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs, state_value = model(input_tensor)
            return action_probs, state_value
        
        return predict, checkpoint
    
    @staticmethod
    def load_torchscript_model(model_path):
        """Load TorchScript model (.pt)"""
        logger.info(f"Loading TorchScript model: {model_path}")
        
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        
        def predict(state):
            """Predict using TorchScript model (action_probs only)"""
            input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_probs = model(input_tensor)
            return action_probs, None  # No value estimation in deployment model
        
        return predict, None
    
    @staticmethod
    def load_onnx_model(model_path):
        """Load ONNX model (.onnx)"""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("ONNX Runtime not available. Install with: pip install onnxruntime")
            return None, None
        
        logger.info(f"Loading ONNX model: {model_path}")
        
        # Try QNN provider first, fall back to CPU
        providers = []
        try:
            # Check if QNN provider is available
            available_providers = ort.get_available_providers()
            if 'QNNExecutionProvider' in available_providers:
                providers.append('QNNExecutionProvider')
                logger.info("QNN provider available for NPU acceleration")
        except:
            pass
        providers.append('CPUExecutionProvider')
        
        try:
            session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            logger.info(f"Using providers: {session.get_providers()}")
            
            def predict(state):
                """Predict using ONNX model"""
                input_data = np.array(state, dtype=np.float32).reshape(1, -1)
                outputs = session.run([output_name], {input_name: input_data})
                action_probs = torch.tensor(outputs[0])
                return action_probs, None  # No value estimation in deployment model
            
            return predict, None
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return None, None
    
    @staticmethod
    def load_model_auto(model_path, config=None):
        """Automatically detect and load model based on file extension"""
        format_type = ModelLoader.detect_format(model_path)
        
        if format_type is None:
            logger.error(f"Unsupported or missing model file: {model_path}")
            return None, None, None
        
        logger.info(f"Detected format: {format_type}")
        
        if format_type == 'pth':
            if config is None:
                logger.error("Config required for .pth format")
                return None, None, None
            predict_func, checkpoint = ModelLoader.load_pth_model(model_path, config)
            return predict_func, checkpoint, format_type
        
        elif format_type == 'torchscript':
            predict_func, checkpoint = ModelLoader.load_torchscript_model(model_path)
            return predict_func, checkpoint, format_type
        
        elif format_type == 'onnx':
            predict_func, checkpoint = ModelLoader.load_onnx_model(model_path)
            return predict_func, checkpoint, format_type
        
        else:
            logger.error(f"Unknown format: {format_type}")
            return None, None, None


class ExampleModeAgent:
    """Agent wrapper for example mode that supports multiple model formats"""
    
    def __init__(self, model_path, config=None):
        self.model_path = model_path
        self.config = config
        self.predict_func = None
        self.checkpoint = None
        self.format_type = None
        
        # Load the model
        self.predict_func, self.checkpoint, self.format_type = ModelLoader.load_model_auto(
            model_path, config
        )
        
        if self.predict_func is None:
            raise ValueError(f"Failed to load model from: {model_path}")
    
    def select_action(self, state):
        """Select action using the loaded model"""
        try:
            action_probs, state_value = self.predict_func(state)
            
            # Convert to numpy if needed
            if isinstance(action_probs, torch.Tensor):
                action_probs_np = action_probs[0].numpy()
            else:
                action_probs_np = action_probs
            
            # Select action with highest probability
            action = np.argmax(action_probs_np)
            
            # For compatibility with PPO agent interface
            log_prob = np.log(action_probs_np[action] + 1e-8)  # Add epsilon for numerical stability
            value = state_value[0].item() if state_value is not None else 0.0
            
            return action, log_prob, value
            
        except Exception as e:
            logger.error(f"Error in select_action: {e}")
            # Fallback to random action
            action = np.random.randint(0, 2)
            return action, 0.0, 0.0
    
    def get_training_state(self):
        """Get training state if available (only for .pth format)"""
        if self.format_type == 'pth' and self.checkpoint:
            return {
                'episode': self.checkpoint.get('episode', 0),
                'timestep': self.checkpoint.get('timestep', 0),
                'reward_history': self.checkpoint.get('reward_history', []),
                'episode_rewards': self.checkpoint.get('episode_rewards', [])
            }
        return None
    
    def get_format_info(self):
        """Get information about the loaded model format"""
        return {
            'path': self.model_path,
            'format': self.format_type,
            'supports_value_estimation': self.format_type == 'pth'
        }
