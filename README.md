# Cart-Pole PPO Reinfo### Features

### PPO Implementation
- Actor-Critic neural network architecture
- Experience collection and policy updates
- Advantage estimation and reward normalization
- Configurable hyperparameters (learning rate, clipping ratio, etc.)
- **Model Save/Load**: Automatic saving and loading of trained models to resume training

### Model Persistence
- **Intelligent Auto-Saving**: Models are automatically saved with adaptive frequency:
  - Every episode for the first 10 episodes
  - Every 5 episodes for episodes 11-50
  - Every 50 episodes (configurable) thereafter
- **Complete State Restoration**: Automatically loads existing models and resumes training exactly where it left off
- **Training State Persistence**: Saves and restores episode count, timestep count, and reward histories
- **Graceful Interruption Handling**: Always saves current state when training is interrupted (Ctrl+C)
- **Configurable Path**: Model save location can be customized in `config.yaml`
- **Comprehensive Checkpoints**: Saves model weights, optimizer state, hyperparameters, and training progress

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
- **Episode**: Current episode number (highlighted in red)
- **Timestep**: Total timesteps completed (highlighted in red)
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
The application now features robust training resumption:

#### Automatic Resume
1. **Smart Detection**: Automatically detects and loads existing models at startup
2. **Complete State Restoration**: Restores episode count, timestep count, reward histories, and neural network state
3. **Seamless Continuation**: Training continues exactly where it left off with no data loss
4. **Progress Preservation**: All training metrics and learning progress are maintained

#### Interruption Recovery
- **Graceful Shutdown**: Press Ctrl+C to safely interrupt training - the current state is automatically saved
- **No Progress Loss**: Even sudden interruptions are handled - auto-saves occur frequently during early training
- **Immediate Resume**: Simply restart the application to continue from the last save point

#### Save Frequency
The system uses intelligent saving patterns:
- **Episodes 1-10**: Saved after every episode (maximum safety)
- **Episodes 11-50**: Saved every 5 episodes  
- **Episodes 50+**: Saved every 50 episodes (configurable)
- **On Interrupt**: Always saves immediately when stopping

#### Training State Details
Saved state includes:
- Neural network weights and biases
- Optimizer state (learning rates, momentum, etc.)
- Current episode and timestep counters
- Complete reward history for visualization
- Episode reward averages for progress tracking
- All hyperparameters for consistency

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
2025-01-01 10:00:01,000 - Model loaded from models/ppo_cartpole.pth
2025-01-01 10:00:01,001 - Training state restored - Episode: 15, Timestep: 342
2025-01-01 10:00:01,002 - Resuming training from episode 16, timestep 342
2025-01-01 10:00:01,003 - Starting Episode 16
2025-01-01 10:00:05,000 - Episode 16 finished with reward: 23.0
2025-01-01 10:00:05,001 - Auto-saving at episode 16
2025-01-01 10:00:05,002 - Model successfully saved to models/ppo_cartpole.pth
2025-01-01 10:00:05,003 - Training state saved - Episode: 16, Timestep: 365
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
