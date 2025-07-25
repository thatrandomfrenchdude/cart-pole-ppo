/**
 * Base class for environment visualizers
 * Contains shared functionality for all environment types
 */
class BaseVisualizer {
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
    
    getStableEnvironmentDetection(data) {
        const detectedEnv = this.detectEnvironment(data);
        
        // Add to history
        this.environmentDetectionHistory.push(detectedEnv);
        if (this.environmentDetectionHistory.length > 5) {
            this.environmentDetectionHistory.shift();
        }
        
        // Check if we have enough consistent detections to change environment
        if (this.environmentDetectionHistory.length >= this.environmentStabilityThreshold) {
            const recentDetections = this.environmentDetectionHistory.slice(-this.environmentStabilityThreshold);
            const isConsistent = recentDetections.every(env => env === detectedEnv);
            
            if (isConsistent && detectedEnv !== this.currentEnvironment) {
                this.updateEnvironmentTheme(detectedEnv);
                return detectedEnv;
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
            if (element && labels[field]) {
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
        
        const margin = 40;
        const chartWidth = canvas.width - 2 * margin;
        const chartHeight = canvas.height - 2 * margin;
        
        // Find min and max rewards for scaling
        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);
        const rewardRange = maxReward - minReward || 1; // Avoid division by zero
        
        // Draw axes
        ctx.strokeStyle = '#34495e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, canvas.height - margin);
        ctx.lineTo(canvas.width - margin, canvas.height - margin);
        ctx.stroke();
        
        // Draw reward line
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < rewards.length; i++) {
            const x = margin + (i / (rewards.length - 1)) * chartWidth;
            const y = canvas.height - margin - ((rewards[i] - minReward) / rewardRange) * chartHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Draw trend line
        if (rewards.length >= 2) {
            const trendData = this.calculateSlope(rewards);
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            
            const startY = canvas.height - margin - ((trendData.intercept - minReward) / rewardRange) * chartHeight;
            const endY = canvas.height - margin - ((trendData.intercept + trendData.slope * (rewards.length - 1) - minReward) / rewardRange) * chartHeight;
            
            ctx.moveTo(margin, startY);
            ctx.lineTo(canvas.width - margin, endY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        // Add labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Episode Rewards', canvas.width / 2, 20);
        
        // Y-axis labels
        ctx.textAlign = 'right';
        ctx.fillText(maxReward.toFixed(1), margin - 5, margin + 5);
        ctx.fillText(minReward.toFixed(1), margin - 5, canvas.height - margin + 5);
        
        // X-axis labels
        ctx.textAlign = 'center';
        if (rewards.length > 1) {
            ctx.fillText('0', margin, canvas.height - margin + 15);
            ctx.fillText((rewards.length - 1).toString(), canvas.width - margin, canvas.height - margin + 15);
        }
    }
    
    updateDisplay(data) {
        // Detect and set environment
        const detectedEnv = this.getStableEnvironmentDetection(data);
        
        // Update episode and timestep
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
        this.statePollingInterval = setInterval(async () => {
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
        this.historyPollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/history');
                if (response.ok) {
                    const data = await response.json();
                    this.updateRewards(data);
                }
            } catch (error) {
                console.error('Failed to fetch reward history:', error);
            }
        }, 1000);
    }
    
    // Abstract method to be implemented by specific visualizers
    drawEnvironment(state) {
        throw new Error('drawEnvironment must be implemented by subclasses');
    }
}
