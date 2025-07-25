/**
 * Multi-Environment Visualizer
 * Main visualizer class that manages different environment visualizers
 */
class MultiEnvironmentVisualizer {
    constructor() {
        // Create instances of all environment visualizers
        this.visualizers = {
            'cartpole': new CartPoleVisualizer(),
            'mountain_car': new MountainCarVisualizer(),
            'pendulum': new PendulumVisualizer(),
            'acrobot': new AcrobotVisualizer()
        };
        
        // Start with CartPole as default
        this.currentVisualizer = this.visualizers['cartpole'];
        this.currentEnvironment = 'cartpole';
        
        // Override the base polling to use environment detection
        this.setupEnvironmentSwitching();
    }
    
    setupEnvironmentSwitching() {
        // Override the startPolling for all visualizers to prevent them from polling independently
        Object.values(this.visualizers).forEach(visualizer => {
            // Store original method
            visualizer._originalStartPolling = visualizer.startPolling;
            // Override to do nothing
            visualizer.startPolling = () => {};
        });
        
        // Set up our own polling that handles environment switching
        this.statePollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/state');
                if (response.ok) {
                    const data = await response.json();
                    
                    // Detect environment and switch if needed
                    const detectedEnv = this.detectEnvironment(data);
                    if (detectedEnv !== this.currentEnvironment) {
                        this.switchEnvironment(detectedEnv);
                    }
                    
                    // Update display with current visualizer
                    this.currentVisualizer.updateDisplay(data);
                    this.currentVisualizer.updateStatus('Training in progress...', 'success');
                }
            } catch (error) {
                this.currentVisualizer.updateStatus('Connection error: ' + error.message, 'error');
            }
        }, 100);
        
        // Poll reward history less frequently
        this.historyPollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/history');
                if (response.ok) {
                    const data = await response.json();
                    this.currentVisualizer.updateRewards(data);
                }
            } catch (error) {
                console.error('Failed to fetch reward history:', error);
            }
        }, 1000);
    }
    
    detectEnvironment(data) {
        // Try to detect environment from server data first
        if (data.environment) {
            return data.environment.toLowerCase();
        }
        
        // Fall back to state-based detection with more robust logic
        // CartPole: has position, velocity, angle, angular_velocity (all defined)
        // And position is typically in [-2.4, 2.4] range, angle in reasonable range
        if (data.position !== undefined && data.velocity !== undefined && 
            data.angle !== undefined && data.angular_velocity !== undefined) {
            
            // Check if this looks like pendulum (position and velocity near zero, just rotation)
            if (Math.abs(data.position) < 0.01 && Math.abs(data.velocity) < 0.01 && 
                (Math.abs(data.angle) > 0.1 || Math.abs(data.angular_velocity) > 0.1)) {
                return 'pendulum';
            }
            
            // Check if this looks like acrobot (larger joint ranges, complex dynamics)
            if (Math.abs(data.angle) > Math.PI || Math.abs(data.position) > Math.PI) {
                return 'acrobot';
            }
            
            return 'cartpole';
        }
        
        // Mountain Car: has position and velocity, angle/angular_velocity undefined or very small
        if (data.position !== undefined && data.velocity !== undefined &&
            data.position >= -1.3 && data.position <= 0.7 &&
            (data.angle === undefined || Math.abs(data.angle) < 0.05)) {
            return 'mountain_car';
        }
        
        // Pendulum: has angle and angular_velocity, position/velocity undefined or very small
        if (data.angle !== undefined && data.angular_velocity !== undefined &&
            (data.position === undefined || Math.abs(data.position) < 0.05) &&
            (data.velocity === undefined || Math.abs(data.velocity) < 0.05)) {
            return 'pendulum';
        }
        
        // Default to cartpole if can't determine
        return 'cartpole';
    }
    
    switchEnvironment(newEnvironment) {
        if (this.visualizers[newEnvironment] && newEnvironment !== this.currentEnvironment) {
            console.log(`Switching from ${this.currentEnvironment} to ${newEnvironment}`);
            
            this.currentEnvironment = newEnvironment;
            this.currentVisualizer = this.visualizers[newEnvironment];
            
            // Update the theme and labels
            this.currentVisualizer.updateEnvironmentTheme(newEnvironment);
        }
    }
}

// Initialize visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MultiEnvironmentVisualizer();
});
