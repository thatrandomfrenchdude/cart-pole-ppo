# Visualization Module Structure

This directory contains the modular visualization system for the multi-environment PPO training project. The visualization has been split into separate files for better maintainability and readability.

## File Structure

### Core Files
- **`base-visualizer.js`** - Base class containing shared functionality for all environments
- **`multi-environment-visualizer.js`** - Main orchestrator that manages environment detection and switching

### Environment-Specific Visualizers
- **`cartpole-visualizer.js`** - CartPole environment visualization
- **`mountain-car-visualizer.js`** - Mountain Car environment visualization  
- **`pendulum-visualizer.js`** - Pendulum environment visualization
- **`acrobot-visualizer.js`** - Acrobot environment visualization

### Web Files
- **`index.html`** - Main HTML page that loads all visualizer scripts
- **`styles.css`** - CSS styling for all environments

## Architecture

### BaseVisualizer Class
Contains shared functionality including:
- Canvas management and resizing
- Environment detection logic
- Data polling from the training server
- Reward chart rendering
- UI theme switching
- Common utility methods

### Environment-Specific Classes
Each environment visualizer extends `BaseVisualizer` and implements:
- `drawEnvironment(state)` - Main drawing method
- Environment-specific drawing logic
- Custom visual elements for each environment type

### MultiEnvironmentVisualizer Class
The main orchestrator that:
- Creates instances of all environment visualizers
- Detects environment changes during training
- Switches between visualizers automatically
- Manages the polling and update cycle

## Usage

The visualization automatically detects which environment is being trained and switches to the appropriate visualizer. No manual configuration is needed.

### Loading Order
The HTML file loads scripts in this order:
1. `base-visualizer.js` (foundation)
2. Environment-specific visualizers (extend base)
3. `multi-environment-visualizer.js` (orchestrator)

### Environment Detection
The system detects environments based on:
1. Server data (`data.environment` field)
2. State characteristics (fallback detection)
3. Stability checking to avoid rapid switching

## Benefits

1. **Modularity** - Each environment has its own focused visualizer
2. **Maintainability** - Easy to modify one environment without affecting others
3. **Extensibility** - Simple to add new environments
4. **Code Reuse** - Shared functionality in base class
5. **Separation of Concerns** - Clear responsibility boundaries

## Testing

To test the visualization:
1. Start training with `python main.py`
2. Open http://localhost:8080 in your browser
3. The visualization will automatically detect and display the current environment
4. Switch environments in `config.yaml` to test different visualizers
