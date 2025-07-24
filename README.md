# Proximal Policy Optimization Visualized with Cart-Pole

<p align="center">
   <img src="assets/cartpole.gif" alt="Cart-Pole PPO Visualization">
</p>

This project demonstrates a complete reinforcement learning pipeline for the cart-pole balancing problem using Proximal Policy Optimization (PPO). It includes a custom cart-pole environment, a PPO agent implemented in PyTorch, and a web-based visualization interface to observe training progress in real-time.

The application allows you to run a pre-trained model in example mode, or train a new model from scratch. It supports multiple model formats including PyTorch, TorchScript, and ONNX, making it flexible for different deployment scenarios. I've also built in a conversion script to convert PyTorch models to ONNX format for use with Qualcomm AI Hub, enabling NPU acceleration on Snapdragon devices.

## Table of Contents
1. [Quick Start](#quick-start)
   - [Prerequisites](#prerequisites)
   - [Setup](#setup)
   - [Run the Application](#run-the-application)
   - [Change Modes](#change-modes)
2. [Model Formats](#model-formats)
3. [Custom Configurations](#custom-configurations)
4. [Proximal Policy Optimization at a Glance](#proximal-policy-optimization-at-a-glance)
5. [Developer Notes](#developer-notes)
   - [Technical Implementation](#technical-implementation)
   - [Project Layout](#project-layout)
   - [Testing](#testing)
6. [License](#license)

## Quick Start
By default, the application runs a pretrained model in **example mode**. You'll see a real-time cart-pole simulation with live metrics demonstrating "perfect" cart-pole balancing behavior without requiring training.

### Prerequisites
Requires either Python 3.8+ with dependencies installed or Docker and Docker Compose.

For NPU acceleration on Snapdragon devices, you must also install and configure the [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/thatrandomfrenchdude/cart-pole-ppo.git
   cd cart-pole-ppo
   ```
2. Configure the environment:
   ```bash
   # a. create a virtual environment
   python -m venv venv

   # b. activate the virtual environment
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows

   # c. install base dependencies
   pip install -r requirements.txt

   # d. install testing dependencies
   pip install -r requirements-test.txt

   # e. OPTIONAL: install QNN dependencies for Snapdragon machines
   pip install onnxruntime-qnn
   ```

3. Run the tests to ensure everything is working correctly:
   ```bash
   # On Linux/macOS
   chmod +x scripts/run_tests.sh
   ./scripts/run_tests.sh all
   
   # On Windows
   .\scripts\run_tests.ps1 all # PowerShell
   .\scripts\run_tests.bat all # Command Prompt
   ```
   Note: I have observed that some of the integrations tests don't work on Windows due to permission issues.

### Run the Application
1. Run the application server:
   ```bash
   # python
   python main.py

   # docker
   docker-compose up
   ```

2. Open your web browser and go to: `http://localhost:8080`

### Change Modes
To enable model training, flip the `example_mode` setting under `training` to `false`:
```yaml

training:
  # some other settings...
  example_mode: false
  # some other settings...
```

## Model Formats

The application supports **PyTorch** (`.pth`), **TorchScript** (`.pt`), and **ONNX** (`.onnx`) model formats for flexibility. The default model format for training is PyTorch and cannot be changed. In example mode, the application will automatically detect the format based on the file extension.

### Model Naming
The model names for both modes can be configured in the `config.yaml` file under the `training` section:
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

### Performance Comparison
You can compare the performance of these models using the `compare_models.py` script. It provides a quick evaluation of each model's inference speed in relation to the others.

Run it as follows:
```bash
# Activate the virtual environment
source venv/bin/activate # Mac/Linux
venv\Scripts\activate    # Windows

# Run the comparison script
python src/compare_models.py # assumes the working directory is the project root
```

### Model Conversion with Qualcomm AI Hub for Snapdragon X Devices
The `src/aihub_conversion.py` script is provided to convert .pth models to .pt and then .onnx using Qualcomm AI Hub for NPU acceleration. You can also convert the models on your own if you prefer.

## Custom Configurations
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

## Proximal Policy Optimization at a Glance
Proximal Policy Optimization, or PPO, is a popular reinforcement learning algorithm that balances exploration and exploitation by optimizing a surrogate objective function. It uses a clipped objective to prevent large policy updates, which helps maintain stability during training.

#### Core Components
- *Policy Network*: Outputs action probabilities based on the current state
- *Value Network*: Estimates the value of states to compute advantages
- *Shared Encoder*: Common feature extraction layers for both policy and value networks

#### Learning Signals
- *Clipped Surrogate Loss*: Prevents large updates to the policy
- *Generalized Advantage Estimation (GAE)*: Reduces variance in advantage estimates while maintaining bias
- *Entropy Bonus*: Encourages exploration by penalizing deterministic policies

### Training Process
1. **Environment Reset**: Start new episode with random initial state
2. **Action Selection**: Policy network chooses actions based on current state
3. **Experience Collection**: Store states, actions, rewards, and probabilities
4. **Policy Updates**: Every 200 steps, update the policy using PPO loss
5. **Performance Tracking**: Log episode rewards and training statistics

### Performance Expectations
- Cart-Pole is considered "solved" when achieving an average reward of 195+ over 100 consecutive episodes
- Training typically shows improvement within the first few episodes
- Complete learning usually occurs within 100-500 episodes depending on initialization

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
├── example/                      # Example models in different formats
│   ├── compare_models.py         # Script to compare model performance
│   ├── debug_qnn.py              # Debug script for QNN validation
│   ├── model.pth                 # PyTorch model
│   ├── model.pt                  # TorchScript model
│   ├── model.onnx                # ONNX model
│   └── training-log.log          # Training log
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

# Windows
.\scripts\run_tests.ps1 all # PowerShell
.\scripts\run_tests.bat all # Command Prompt

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