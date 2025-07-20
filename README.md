# Cart-Pole PPO Reinforcement Learning

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

### Modify Hyperparameters
Edit the PPOAgent initialization in `main.py`:
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
