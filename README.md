# Cart-Pole PPO Reinfo### Features

### PPO Implementation
- Actor-Critic neural network architecture
- Experience collection and policy updates
- Advantage estimation and reward normalization
- Configurable hyperparameters (learning rate, clipping ratio, etc.)
- **Model Save/Load**: Automatic saving and loading of trained models to resume training

### Model Persistence
- **Automatic Saving**: Models are saved every 50 episodes by default
- **Resume Training**: Automatically loads existing models when restarting the application
- **Configurable Path**: Model save location can be customized in `config.yaml`
- **Checkpoint Data**: Saves both model weights and optimizer state for seamless resumption

### Logging System
- **Dual Output**: Logs are written to both console and file simultaneously
- **Session Overwrite**: Each run overwrites the previous log file (keeps only the latest session)
- **Configurable File**: Log filename can be customized in `config.yaml`
- **Detailed Tracking**: Comprehensive logging of training progress, model saves, and system events Learning

A minimal imple4. Open your web browser and go to: `http://localhost:8080`entation of Proximal Policy Optimization (PPO) for the Cart-Pole environment with real-time web visualization.

## Overview

This project demonstrates a complete reinforcement learning pipeline featuring:

- **PPO Algorithm**: A minimal but effective implementation of Proximal Policy Optimization
- **Cart-Pole Environment**: Implemented from scratch using NumPy
- **Real-time Visualization**: Web-based interface showing the cart-pole simulation and training progress
- **Console Logging**: Detailed training information logged to the console
- **Dockerized Deployment**: Easy setup and deployment using Docker Compose

## Architecture

- `main.py`: Python backend with PPO implementation and Flask web server
- `index.html`: Web interface structure
- `visualization.js`: Real-time visualization and data polling
- `styles.css`: Styling for the web interface
- `docker-compose.yaml`: Docker configuration for easy deployment

## Features

### PPO Implementation
- Actor-Critic neural network architecture
- Experience collection and policy updates
- Advantage estimation and reward normalization
- Configurable hyperparameters (learning rate, clipping ratio, etc.)

### Visualization
- Real-time cart-pole animation showing position and angle
- Episode reward chart tracking training progress
- Live metrics display (position, velocity, angle, angular velocity)
- Responsive design for different screen sizes

### Monitoring
- Detailed console logging of each training step
- Episode completion statistics
- Average reward tracking
- PPO update notifications

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

## Understanding the Visualization

### Cart-Pole Display
- **Blue rectangle**: The cart that moves left and right
- **Red line**: The pole that needs to be balanced
- **Dashed line**: Center position reference
- **Position/Angle labels**: Current state values

### Reward Chart
- Shows episode rewards over time
- Green line indicates training progress
- Y-axis scales automatically based on performance range

### Metrics Panel
- **Position**: Cart's horizontal position (-∞ to +∞)
- **Velocity**: Cart's horizontal velocity
- **Angle**: Pole's angle from vertical (radians)
- **Angular Velocity**: Pole's rotational velocity
- **Current Reward**: Reward from the last step
- **Average Reward**: Running average across recent episodes

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

## Training Process

1. **Environment Reset**: Start new episode with random initial state
2. **Action Selection**: Policy network chooses actions based on current state
3. **Experience Collection**: Store states, actions, rewards, and probabilities
4. **Policy Updates**: Every 200 steps, update the policy using PPO loss
5. **Performance Tracking**: Log episode rewards and training statistics

## Troubleshooting

### Common Issues
- **Port 5000 already in use**: Stop other services using port 5000 or modify the port in `docker-compose.yaml`
- **Docker not found**: Ensure Docker is installed and running
- **Slow performance**: The simulation runs with a 50ms delay between steps for visualization purposes

### Monitoring Training
- Watch the console output for detailed training logs
- Episode rewards should generally increase over time
- The cart-pole should balance for longer periods as training progresses

## Customization

### Configuration File
The `config.yaml` file allows you to customize all aspects of training:

```yaml
# Model persistence
training:
  model_save_path: "models/ppo_cartpole.pth"  # Where to save/load models
  save_frequency: 50                          # Save every N episodes
  
# PPO hyperparameters
ppo:
  learning_rate: 0.0003
  discount_factor: 0.99
  clip_ratio: 0.2
  
# Environment physics
environment:
  gravity: 9.8
  cart_mass: 1.0
  pole_mass: 0.1

# Logging configuration
logging:
  level: "INFO"                               # Logging level (DEBUG, INFO, WARNING, ERROR)
  log_file: "training.log"                    # Log file name (overwrites each run)
  episode_summary_frequency: 10               # Episodes between summary logs
```

### Resume Training
The application automatically:
1. Checks for an existing model at the configured path
2. Loads the model if found and continues training
3. Creates a new model if no saved model exists
4. Saves checkpoints periodically during training

### Log Files
Training sessions generate detailed log files:
- **Overwrite Mode**: Each run creates a fresh log file (previous logs are replaced)
- **Dual Output**: All logs appear in both console and file
- **Session Tracking**: Clear start/end markers for each training session
- **Comprehensive Details**: Episode progress, model saves, errors, and system events
- **Configurable Name**: Log filename can be customized in `config.yaml`

Example log content:
```
2025-01-01 10:00:00,000 - ============================================================
2025-01-01 10:00:00,001 - PPO Cart-Pole Training Session Started
2025-01-01 10:00:00,002 - ============================================================
2025-01-01 10:00:00,003 - Configuration loaded successfully
2025-01-01 10:00:00,004 - Logging to file: training.log (overwrite mode)
2025-01-01 10:00:01,000 - Starting Episode 1
2025-01-01 10:00:05,000 - Episode 1 finished with reward: 23.0
2025-01-01 10:00:15,000 - Reached save frequency trigger: Episode 50, Save frequency: 50
2025-01-01 10:00:15,001 - Model successfully saved to models/ppo_cartpole.pth
```

### Modify Hyperparameters
Edit values in `config.yaml` or modify the PPOAgent initialization in `main.py`:
```python
agent = PPOAgent(lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)
```

### Adjust Update Frequency
Change the `update_timestep` variable in `main.py`:
```python
update_timestep = 200  # Update every N steps
```

### Modify Visualization Speed
Adjust the sleep time in the training loop:
```python
time.sleep(0.05)  # 50ms between steps
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
