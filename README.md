# Proxmial Policy Optimization Visualized with Cart-Pole

<p align="center">
   <img src="assets/cartpole.gif" alt="Cart-Pole PPO Visualization">
</p>

This project demonstrates a complete reinforcement learning pipeline for the cart-pole balancing problem using Proximal Policy Optimization (PPO). It includes a custom cart-pole environment, a PPO agent implemented in PyTorch, and a web-based visualization interface to observe training progress in real-time.

## Table of Contents
- [Quick Start](#quick-start)
- [Model Formats](#model-formats)
- [Researcher Notes](#researcher-notes)
   - [PPO Algorithm Details](#ppo-algorithm-details)
   - [Customization with the Configuration File](#customization-with-the-configuration-file)
- [Developer Notes](#developer-notes)
   - [Technical Implementation](#technical-implementation)
   - [Project Layout](#project-layout)
   - [Testing](#testing)
- [License](#license)

## Quick Start
By default, the application runs a pretrained model in **example mode**. You'll see a real-time cart-pole simulation with live metrics demonstrating "perfect" cart-pole balancing behavior without requiring training time.

**To enable model training**, flip the `example_mode` setting under `training` to `false`:
```yaml
training:
  # some other settings...
  example_mode: false
  # some other settings...
```

### Prerequisites
Requires either Python 3.8+ with dependencies installed or Docker and Docker Compose. Use the appropriate method in the run instructions below.

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/thatrandomfrenchdude/cart-pole-ppo.git
   cd cart-pole-ppo
   ```
2. Configure the environment:
   ```bash
   # python
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

### Run the Tests
Run the tests to ensure everything is working correctly. The tests cover unit tests, integration tests, and performance tests.
1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

2. Run the tests:
   ```bash
   # On Linux/macOS
   chmod +x scripts/run_tests.sh
   ./scripts/run_tests.sh all
   
   # On Windows (PowerShell)
   .\scripts\run_tests.ps1 all
   ```

### Run the Application
1. Run the application server:
   ```bash
   # python
   python main.py

   # docker
   docker-compose up
   ```

2. Open your web browser and go to: `http://localhost:8080`

## Model Formats
The application supports multiple model formats for flexibility:
- **PyTorch**: `.pth` files
- **TorchScript**: `.pt` files
- **ONNX**: `.onnx` files

For training, the default model format is PyTorch and cannot be changed. In example mode, the model can be in any of the supported formats. The application will automatically detect the format based on the file extension.

The names of the models for training and example mode can be configured in the `config.yaml` file under the `training` section:
```yaml
training:
  model_save_path: "models/ppo_cartpole.pth"  # .pth only for training, any file name is accepted
  example_model_path: "example/model.pth"     # .pth, .pt, or .onnx supported
```

### Example Models
Examples of all three model formats can be found in the `example` directory:
```example/
├── model.pth        # PyTorch model
├── model.pt         # TorchScript model
└── model.onnx       # ONNX model
```
You can compare the performance of these models by activating the virtual environment and running the `compare_models.py` script:
```bash
# Activate the virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Run the comparison script
python src/compare_models.py # assumes the working directory is the project root
```

### A Note on Model Conversion
The `aihub_conversion.py` script is provided in the `src` directory to convert .pth models to .pt and .onnx using Qualcomm AI Hub. This provides NPU acceleration for the ONNX model when running on a Snapdragon X device. You can also convert the models on your own if you prefer.

## Researcher Notes

### PPO Algorithm Details

The implementation includes:
- **Policy Network**: Outputs action probabilities
- **Value Network**: Estimates state values for advantage calculation
- **Shared Feature Extraction**: Common layers for both policy and value functions
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation**: Improves learning stability

#### Training Process

1. **Environment Reset**: Start new episode with random initial state
2. **Action Selection**: Policy network chooses actions based on current state
3. **Experience Collection**: Store states, actions, rewards, and probabilities
4. **Policy Updates**: Every 200 steps, update the policy using PPO loss
5. **Performance Tracking**: Log episode rewards and training statistics

#### Performance Expectations

- Cart-Pole is considered "solved" when achieving an average reward of 195+ over 100 consecutive episodes
- Training typically shows improvement within the first few episodes
- Complete learning usually occurs within 100-500 episodes depending on initialization

### Customization with the Configuration File
The [`config.yaml`](config.yaml) file allows you to customize all aspects of training. There are sections for environment parameters, neural network architecture, PPO hyperparameters, training settings, server configuration, and logging.

Below are the environment and ppo subsets of the config:

```yaml
# Cart-Pole PPO Configuration

# Environment Physics Parameters
environment:
  gravity: 9.8                          # Gravitational acceleration (m/s^2)
  cart_mass: 1.0                        # Mass of the cart (kg)
  pole_mass: 0.1                        # Mass of the pole (kg)
  pole_half_length: 0.5                 # Half length of the pole (m)
  force_magnitude: 10.0                 # Magnitude of force applied to cart (N)
  time_step: 0.02                       # Time between state updates (seconds)
  position_threshold: 2.4               # Cart position limit (m)
  angle_threshold_degrees: 12           # Pole angle limit (degrees)

# PPO Algorithm Parameters
ppo:
  learning_rate: 0.0003                 # Learning rate for optimizer
  discount_factor: 0.99                 # Gamma - reward discount factor
  clip_ratio: 0.2                       # Epsilon - PPO clipping ratio
  update_epochs: 4                      # Number of epochs per PPO update
  update_frequency: 200                 # Steps between PPO updates
```

## Developer Notes

### Technical Implementation

- The implementation uses PyTorch for neural networks
- Flask serves both the API and static files
- The training runs in a separate thread to avoid blocking the web server
- State normalization and advantage estimation improve learning stability
- The visualization polls the backend every 100ms for smooth animation
- Modular architecture allows for easy testing and maintenance

### Project Layout

The project is organized as follows:
```
cart-pole-ppo/
├── scripts/                      # Utility scripts
│   ├── run_tests.sh              # Script to run all tests (Linux/macOS)
│   ├── run_tests.bat             # Script to run all tests (Windows)
│   └── run_tests.ps1             # Script to run all tests (Windows PowerShell)
├── src/
│   ├── visualization/            # Frontend files (HTML, CSS, JS)
│   │   ├── index.html            # Main HTML page
│   │   ├── styles.css            # CSS styles
│   │   └── visualization.js      # JavaScript for visualization and API calls
│   ├── agent.py                  # PPO agent implementation
│   ├── aihub_conversion.py       # Model conversion script for Qualcomm AI Hub
│   ├── config.py                 # Configuration file for hyperparameters and settings
│   ├── environment.py            # Custom cart-pole environment
│   ├── model_loader.py           # Model loading and inference logic
│   ├── network.py                # Neural network architecture
│   ├── training.py               # Training loop and experience collection
│   ├── utils.py                  # Utility functions (logging, config loading)
│   └── web_server.py             # Flask server and API endpoints
├── tests/                        # Test files
│   ├── conftest.py               # Pytest fixtures and configuration
│   ├── test_agent.py             # Tests for PPO agent
│   ├── test_config.py            # Tests for configuration loading
│   ├── test_environment.py       # Tests for custom environment
│   ├── test_formats.py           # Tests for model formats and conversions
│   ├── test_integration.py       # Integration tests for end-to-end functionality
│   ├── test_main.py              # Tests for main entry point
│   ├── test_model.py             # Tests for model loading and inference
│   ├── test_network.py           # Tests for neural network architecture
│   ├── test_onnx_model.py        # Tests for ONNX model compatibility
│   ├── test_training.py          # Tests for training loop
│   ├── test_utils.py             # Tests for utility functions
│   └── test_web_server.py        # Tests for web server and API endpoints
├── config.yaml                   # Configuration file for hyperparameters and settings
├── docker-compose.yml            # Docker Compose configuration
├── main.py                       # Main entry point to start training and server
├── requirements-test.txt         # Test dependencies
└── requirements.txt              # Python dependencies
```

### Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Or use the test runner scripts:
# Linux/macOS
./scripts/run_tests.sh all

# Windows PowerShell
.\scripts\run_tests.ps1 all

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run only unit tests (fast)
pytest -m "not slow and not integration"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_environment.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.