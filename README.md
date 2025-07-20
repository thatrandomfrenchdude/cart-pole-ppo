# Cart-Pole PPO Reinforcement Learning with Visualization

This project demonstrates a complete reinforcement learning pipeline featuring:

- **PPO Algorithm**: A minimal but effective implementation of Proximal Policy Optimization
- **Cart-Pole Environment**: Implemented from scratch using NumPy
- **Real-time Visualization**: Web-based interface showing the cart-pole simulation and training progress
- **Console Logging**: Detailed training information logged to the console and a training file
- **Configurable Hyperparameters**: Easily adjustable learning rate, discount factor, clipping ratio, etc.
- **Model Persistence**: Automatic saving and loading of trained models to resume training
- **Dockerized Deployment**: Easy setup and deployment using Docker Compose

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [PPO Algorithm Details](#ppo-algorithm-details)
- [Customization with the Configuration File](#customization-with-the-configuration-file)
- [Technical Notes](#technical-notes)
- [Performance Expectations](#performance-expectations)

## Project Layout

- `main.py`: Python backend with PPO implementation and Flask web server
- `index.html`: Web interface structure
- `visualization.js`: Real-time visualization and data polling
- `styles.css`: Styling for the web interface
- `docker-compose.yaml`: Docker configuration for easy deployment
- `config.yaml`: Configuration file for hyperparameters and settings

## Quick Start

### Prerequisites
- Docker and Docker Compose installed on your system

### Run the Application
1. Clone or download this repository
2. Navigate to the project directory
3. Start the application:
   ```bash
   docker-compose up
   ```

4. Open your web browser and go to: `http://localhost:8080`

The training will start automatically and you'll see:
- Real-time cart-pole simulation
- Training metrics and episode rewards
- Console logs in the Docker output

## PPO Algorithm Details

The implementation includes:
- **Policy Network**: Outputs action probabilities
- **Value Network**: Estimates state values for advantage calculation
- **Shared Feature Extraction**: Common layers for both policy and value functions
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation**: Improves learning stability

### Hyperparameters
- Learning Rate: 3e-4
- Discount Factor (γ): 0.99
- Clipping Ratio (ε): 0.2
- Update Epochs: 4
- Update Frequency: Every 200 steps

### Training Process

1. **Environment Reset**: Start new episode with random initial state
2. **Action Selection**: Policy network chooses actions based on current state
3. **Experience Collection**: Store states, actions, rewards, and probabilities
4. **Policy Updates**: Every 200 steps, update the policy using PPO loss
5. **Performance Tracking**: Log episode rewards and training statistics

## Customization with the Configuration File
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

## Technical Notes

- The implementation uses PyTorch for neural networks
- Flask serves both the API and static files
- The training runs in a separate thread to avoid blocking the web server
- State normalization and advantage estimation improve learning stability
- The visualization polls the backend every 100ms for smooth animation

## Performance Expectations

- Cart-Pole is considered "solved" when achieving an average reward of 195+ over 100 consecutive episodes
- Training typically shows improvement within the first few episodes
- Complete learning usually occurs within 100-500 episodes depending on initialization

Enjoy watching your PPO agent learn to balance the cart-pole!
