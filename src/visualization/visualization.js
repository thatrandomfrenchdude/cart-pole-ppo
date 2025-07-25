class MultiEnvironmentVisualizer {
    constructor() {
        this.simulationCanvas = document.getElementById('simulation-canvas');
        this.rewardCanvas = document.getElementById('reward-canvas');
        this.simulationCtx = this.simulationCanvas.getContext('2d');
        this.rewardCtx = this.rewardCanvas.getContext('2d');
        
        this.rewardHistory = [];
        this.maxRewards = 100;
        this.currentEnvironment = 'cartpole'; // Default
        
        // Add environment detection stability
        this.environmentDetectionHistory = [];
        this.environmentStabilityThreshold = 3; // Require 3 consistent detections to change
        
        this.resizeCanvases();
        this.initializeCanvases();
        this.setupTrainingMetrics();
        this.startPolling();
        
        // Add resize listener
        window.addEventListener('resize', () => this.resizeCanvases());
    }
    
    setupTrainingMetrics() {
        // Add special styling to episode and timestep metrics
        const episodeItem = document.getElementById('episode').closest('.info-item');
        const timestepItem = document.getElementById('timestep').closest('.info-item');
        
        if (episodeItem) episodeItem.classList.add('training-metric');
        if (timestepItem) timestepItem.classList.add('training-metric');
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
            
            // Check if this looks like mountain car (position in mountain car range [-1.2, 0.6])
            // AND angle and angular velocity are consistently very small (not meaningful)
            if (data.position >= -1.3 && data.position <= 0.7 && 
                Math.abs(data.angle) < 0.05 && Math.abs(data.angular_velocity) < 0.05) {
                return 'mountain_car';
            }
            
            // Check if this looks like acrobot (two joint angles, different value ranges)
            if (Math.abs(data.position) > 3.0 || Math.abs(data.angle) > 3.0) {
                return 'acrobot';
            }
            
            // Default to cartpole if all four states are present and in cartpole-like ranges
            // CartPole position is typically [-2.4, 2.4], angle [-0.2, 0.2] for most operation
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
    
    getStableEnvironmentDetection(data) {
        const detectedEnv = this.detectEnvironment(data);
        
        // Add to history
        this.environmentDetectionHistory.push(detectedEnv);
        if (this.environmentDetectionHistory.length > 5) {
            this.environmentDetectionHistory.shift(); // Keep only last 5 detections
        }
        
        // Check if we have enough consistent detections to change environment
        if (this.environmentDetectionHistory.length >= this.environmentStabilityThreshold) {
            const recentDetections = this.environmentDetectionHistory.slice(-this.environmentStabilityThreshold);
            const allSame = recentDetections.every(env => env === recentDetections[0]);
            
            if (allSame && recentDetections[0] !== this.currentEnvironment) {
                return recentDetections[0];
            }
        }
        
        // Return current environment if not enough consistent detections
        return this.currentEnvironment;
    }
    
    updateEnvironmentTheme(environment) {
        if (this.currentEnvironment === environment) return;
        
        this.currentEnvironment = environment;
        const body = document.body;
        
        // Remove existing theme classes
        body.classList.remove('cartpole-theme', 'mountain-car-theme', 'pendulum-theme', 'acrobot-theme');
        
        // Add new theme class
        body.classList.add(`${environment.replace('_', '-')}-theme`);
        
        // Update titles and labels based on environment
        this.updateEnvironmentLabels(environment);
    }
    
    updateEnvironmentLabels(environment) {
        const envNames = {
            'cartpole': 'CartPole',
            'mountain_car': 'Mountain Car',
            'pendulum': 'Pendulum',
            'acrobot': 'Acrobot'
        };
        
        const simTitles = {
            'cartpole': 'Cart-Pole Simulation',
            'mountain_car': 'Mountain Car Simulation', 
            'pendulum': 'Pendulum Simulation',
            'acrobot': 'Acrobot Simulation'
        };
        
        document.getElementById('main-title').textContent = `${envNames[environment]} PPO Training`;
        document.getElementById('simulation-title').textContent = simTitles[environment];
        document.getElementById('environment').textContent = envNames[environment];
        
        // Update field labels based on environment
        const labels = this.getEnvironmentLabels(environment);
        Object.keys(labels).forEach(field => {
            const element = document.getElementById(`${field}-label`);
            if (element) {
                element.textContent = labels[field];
            }
            
            // Show/hide fields based on environment
            const item = document.getElementById(`${field}-item`);
            if (item) {
                if (labels[field] === null) {
                    item.classList.add('hidden');
                } else {
                    item.classList.remove('hidden');
                }
            }
        });
    }
    
    getEnvironmentLabels(environment) {
        const labelMaps = {
            'cartpole': {
                'position': 'Cart Position:',
                'velocity': 'Cart Velocity:',
                'angle': 'Pole Angle:',
                'angular-velocity': 'Pole Angular Vel:'
            },
            'mountain_car': {
                'position': 'Car Position:',
                'velocity': 'Car Velocity:',
                'angle': null,  // Hide
                'angular-velocity': null  // Hide
            },
            'pendulum': {
                'position': null,  // Hide
                'velocity': null,  // Hide
                'angle': 'Pendulum Angle:',
                'angular-velocity': 'Angular Velocity:'
            },
            'acrobot': {
                'position': 'Joint 1 Angle:',
                'velocity': 'Joint 1 Velocity:',
                'angle': 'Joint 2 Angle:',
                'angular-velocity': 'Joint 2 Velocity:'
            }
        };
        
        return labelMaps[environment] || labelMaps['cartpole'];
    }
    
    resizeCanvases() {
        // Get the container width
        const simulationContainer = this.simulationCanvas.parentElement;
        const rewardContainer = this.rewardCanvas.parentElement;
        
        const containerWidth = simulationContainer.offsetWidth - 40; // Account for padding
        
        // Resize simulation canvas
        this.simulationCanvas.width = containerWidth;
        this.simulationCanvas.height = 300;
        
        // Resize reward canvas  
        this.rewardCanvas.width = containerWidth;
        this.rewardCanvas.height = 200;
        
        // Reinitialize after resize
        this.initializeCanvases();
    }
    
    initializeCanvases() {
        // Set up simulation canvas
        this.simulationCtx.fillStyle = '#f8f9fa';
        this.simulationCtx.fillRect(0, 0, this.simulationCanvas.width, this.simulationCanvas.height);
        
        // Set up reward canvas
        this.rewardCtx.fillStyle = '#f8f9fa';
        this.rewardCtx.fillRect(0, 0, this.rewardCanvas.width, this.rewardCanvas.height);
    }
    
    drawEnvironment(state) {
        switch(this.currentEnvironment) {
            case 'cartpole':
                drawCartPole(this.simulationCtx, this.simulationCanvas, state);
                break;
            case 'mountain_car':
                drawMountainCar(this.simulationCtx, this.simulationCanvas, state);
                break;
            case 'pendulum':
                drawPendulum(this.simulationCtx, this.simulationCanvas, state);
                break;
            case 'acrobot':
                drawAcrobot(this.simulationCtx, this.simulationCanvas, state);
                break;
            default:
                drawCartPole(this.simulationCtx, this.simulationCanvas, state); // Fallback
        }
    }

    
    calculateSlope(rewards) {
        if (rewards.length < 2) return { slope: 0, intercept: 0 };
        
        const n = rewards.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (let i = 0; i < n; i++) {
            sumX += i;
            sumY += rewards[i];
            sumXY += i * rewards[i];
            sumX2 += i * i;
        }
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        return { slope, intercept };
    }
    
    drawRewardChart(rewards) {
        const ctx = this.rewardCtx;
        const canvas = this.rewardCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (rewards.length === 0) return;
        
        // Draw axes
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(40, 10);
        ctx.lineTo(40, canvas.height - 30);
        ctx.lineTo(canvas.width - 10, canvas.height - 30);
        ctx.stroke();
        
        // Find min/max for scaling
        const maxReward = Math.max(...rewards);
        const minReward = Math.min(...rewards);
        const rewardRange = maxReward - minReward || 1;
        
        // Draw reward line
        if (rewards.length > 1) {
            ctx.strokeStyle = '#27ae60';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const stepX = (canvas.width - 50) / Math.max(rewards.length - 1, 1);
            
            for (let i = 0; i < rewards.length; i++) {
                const x = 40 + i * stepX;
                const y = canvas.height - 30 - ((rewards[i] - minReward) / rewardRange) * (canvas.height - 40);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Draw points
            ctx.fillStyle = '#27ae60';
            for (let i = 0; i < rewards.length; i++) {
                const x = 40 + i * stepX;
                const y = canvas.height - 30 - ((rewards[i] - minReward) / rewardRange) * (canvas.height - 40);
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Calculate and draw trend line (slope)
            const trendData = this.calculateSlope(rewards);
            if (rewards.length >= 2) {
                ctx.strokeStyle = '#ff69b4'; // Pink color
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]); // Dotted line
                ctx.beginPath();
                
                // Calculate trend line endpoints
                const startX = 40;
                const endX = 40 + (rewards.length - 1) * stepX;
                const startY = canvas.height - 30 - ((trendData.intercept - minReward) / rewardRange) * (canvas.height - 40);
                const endY = canvas.height - 30 - (((trendData.slope * (rewards.length - 1) + trendData.intercept) - minReward) / rewardRange) * (canvas.height - 40);
                
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
                ctx.stroke();
                ctx.setLineDash([]); // Reset line dash
                
                // Display slope value
                ctx.fillStyle = '#ff69b4';
                ctx.font = '11px Arial';
                ctx.textAlign = 'left';
                const slopeText = `Trend: ${trendData.slope.toFixed(4)}`;
                ctx.fillText(slopeText, canvas.width - 120, 20);
            }
        }
        
        // Draw labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '10px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(maxReward.toFixed(0), 35, 15);
        ctx.fillText(minReward.toFixed(0), 35, canvas.height - 35);
        
        ctx.textAlign = 'center';
        ctx.fillText('Episodes', canvas.width / 2, canvas.height - 5);
        
        // Y-axis label
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Reward', 0, 0);
        ctx.restore();
    }
    
    updateDisplay(data) {
        // Use stable environment detection to prevent flipping
        const stableEnv = this.getStableEnvironmentDetection(data);
        this.updateEnvironmentTheme(stableEnv);
        
        // Update info panel
        document.getElementById('episode').textContent = data.episode || 0;
        document.getElementById('timestep').textContent = data.timestep || 0;
        document.getElementById('reward').textContent = data.reward.toFixed(3);
        
        // Update environment-specific fields
        if (data.position !== undefined) {
            document.getElementById('position').textContent = data.position.toFixed(3);
        }
        if (data.velocity !== undefined) {
            document.getElementById('velocity').textContent = data.velocity.toFixed(3);
        }
        if (data.angle !== undefined) {
            document.getElementById('angle').textContent = (data.angle * 180 / Math.PI).toFixed(1) + '°';
        }
        if (data.angular_velocity !== undefined) {
            document.getElementById('angular-velocity').textContent = data.angular_velocity.toFixed(3);
        }
        
        // Draw the appropriate environment
        this.drawEnvironment(data);
    }
    
    updateRewards(historyData) {
        if (historyData.rewards && historyData.rewards.length > 0) {
            this.rewardHistory = historyData.rewards.slice(-this.maxRewards);
            this.drawRewardChart(this.rewardHistory);
            
            // Update slope display in info panel
            if (this.rewardHistory.length >= 2) {
                const trendData = this.calculateSlope(this.rewardHistory);
                document.getElementById('reward-trend').textContent = trendData.slope.toFixed(4);
            } else {
                document.getElementById('reward-trend').textContent = '0.000';
            }
        }
        
        if (historyData.avg_reward !== undefined) {
            document.getElementById('avg-reward').textContent = historyData.avg_reward.toFixed(3);
        }
    }
    
    updateStatus(message, type = '') {
        const statusEl = document.getElementById('status');
        statusEl.textContent = message;
        statusEl.className = 'status' + (type ? ' ' + type : '');
    }
    
    startPolling() {
        // Poll state frequently for smooth animation
        setInterval(async () => {
            try {
                const response = await fetch('/state');
                if (response.ok) {
                    const data = await response.json();
                    this.updateDisplay(data);
                    this.updateStatus('Training in progress...', 'success');
                }
            } catch (error) {
                this.updateStatus('Connection error: ' + error.message, 'error');
            }
        }, 100);
        
        // Poll reward history less frequently
        setInterval(async () => {
            try {
                const response = await fetch('/history');
                if (response.ok) {
                    const data = await response.json();
                    this.updateRewards(data);
                }
            } catch (error) {
                // Silently fail - history is not critical for visualization
            }
        }, 1000);
    }
}

// Initialize visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MultiEnvironmentVisualizer();
});