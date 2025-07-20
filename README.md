# Cart-Pole PPO Reinforcement Learning with Visualization

This project demonstrates a complete reinforcement learning pipeline for the cart-pole balancing problem using Proximal Policy Optimization (PPO). It includes a custom cart-pole environment, a PPO agent implemented in PyTorch, and a web-based visualization interface to observe training progress in real-time.

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Researcher Notes](#researcher-notes)
   - [PPO Algorithm Details](#ppo-algorithm-details)
   - [Customization with the Configuration File](#customization-with-the-configuration-file)
- [Developer Notes](#developer-notes)
   - [Technical Implementation](#technical-implementation)
   - [Project Layout](#project-layout)
- [License](#license)

## Quick Start

### Prerequisites
Requires either Python 3.8+ with dependencies installed or Docker and Docker Compose. Use the appropriate method in the run instructions below.

### Run the Application
1. Clone this repository:
   ```bash
   git clone https://github.com/thatrandomfrenchdude/cart-pole-ppo.git
   cd cart-pole-ppo
   ```
2. Start the application:
   ```bash
   # python
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   python main.py

   # docker
   docker-compose up
   ```

3. Open your web browser and go to: `http://localhost:8080`

The training will start automatically and you'll see:
- Real-time cart-pole simulation
- Training metrics and episode rewards
- Live logging in the console output

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

### Project Layout

- `main.py`: Python backend with PPO implementation and Flask web server
- `index.html`: Web interface structure
- `visualization.js`: Real-time visualization and data polling
- `styles.css`: Styling for the web interface
- `docker-compose.yaml`: Docker configuration for easy deployment
- `config.yaml`: Configuration file for hyperparameters and settings

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.