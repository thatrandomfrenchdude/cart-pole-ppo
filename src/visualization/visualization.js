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
                this.drawCartPole(state);
                break;
            case 'mountain_car':
                this.drawMountainCar(state);
                break;
            case 'pendulum':
                this.drawPendulum(state);
                break;
            case 'acrobot':
                this.drawAcrobot(state);
                break;
            default:
                this.drawCartPole(state); // Fallback
        }
    }

    drawCartPole(state) {
        const ctx = this.simulationCtx;
        const canvas = this.simulationCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw ground line
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - 50);
        ctx.lineTo(canvas.width, canvas.height - 50);
        ctx.stroke();
        
        // Calculate positions
        const centerX = canvas.width / 2;
        const groundY = canvas.height - 50;
        
        // Cart position (scale for visualization)
        const cartX = centerX + (state.position * 100);
        const cartY = groundY - 25;
        
        // Draw cart
        ctx.fillStyle = '#3498db';
        ctx.fillRect(cartX - 25, cartY - 15, 50, 30);
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.strokeRect(cartX - 25, cartY - 15, 50, 30);
        
        // Draw wheels
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(cartX - 15, cartY + 15, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cartX + 15, cartY + 15, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw pole
        const poleLength = 80;
        const poleEndX = cartX + Math.sin(state.angle) * poleLength;
        const poleEndY = cartY - Math.cos(state.angle) * poleLength;
        
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(cartX, cartY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();
        
        // Draw pole tip
        ctx.fillStyle = '#c0392b';
        ctx.beginPath();
        ctx.arc(poleEndX, poleEndY, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw center mark
        ctx.strokeStyle = '#95a5a6';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, canvas.height);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Add position indicators
        ctx.fillStyle = '#7f8c8d';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Center', centerX, 20);
        ctx.fillText(`Pos: ${state.position.toFixed(3)}`, cartX, cartY - 40);
        ctx.fillText(`Angle: ${(state.angle * 180 / Math.PI).toFixed(1)}°`, poleEndX, poleEndY - 15);
    }

    drawMountainCar(state) {
        const ctx = this.simulationCtx;
        const canvas = this.simulationCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw mountain valley (cosine function as described)
        const mountainBase = canvas.height - 80;
        const amplitude = 60;  // Height variation
        ctx.strokeStyle = '#8e44ad';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        // Create the cosine valley shape from x=-1.2 to x=0.6
        const points = [];
        for (let x = 0; x <= canvas.width; x += 2) {
            const realX = (x / canvas.width) * 1.8 - 1.2; // Scale to [-1.2, 0.6]
            // Valley floor defined by cosine function (inverted cosine for valley)
            const y = mountainBase + amplitude * Math.cos(3 * realX);
            points.push([x, y]);
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Fill mountain/valley
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.fillStyle = 'rgba(142, 68, 173, 0.3)';  // Semi-transparent purple
        ctx.fill();
        
        // Calculate car position on the valley floor
        const carRealX = state.position; // Position in [-1.2, 0.6]
        const carScreenX = ((carRealX + 1.2) / 1.8) * canvas.width;
        const carY = mountainBase + amplitude * Math.cos(3 * carRealX) - 15; // On valley floor
        
        // Draw car with more detail
        ctx.fillStyle = '#8e44ad';
        ctx.fillRect(carScreenX - 15, carY - 10, 30, 20);
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        ctx.strokeRect(carScreenX - 15, carY - 10, 30, 20);
        
        // Draw car body details
        ctx.fillStyle = '#9b59b6';
        ctx.fillRect(carScreenX - 12, carY - 7, 24, 14);
        
        // Draw wheels
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(carScreenX - 8, carY + 10, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(carScreenX + 8, carY + 10, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw goal area at x >= 0.45
        const goalStartX = ((0.45 + 1.2) / 1.8) * canvas.width;
        const goalEndX = canvas.width;
        const goalY = mountainBase + amplitude * Math.cos(3 * 0.45);
        
        // Goal zone background
        ctx.fillStyle = 'rgba(39, 174, 96, 0.2)';
        ctx.fillRect(goalStartX, 0, goalEndX - goalStartX, canvas.height);
        
        // Goal flag at x=0.5
        const flagX = ((0.5 + 1.2) / 1.8) * canvas.width;
        const flagY = mountainBase + amplitude * Math.cos(3 * 0.5);
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(flagX, flagY);
        ctx.lineTo(flagX, flagY - 50);
        ctx.stroke();
        
        // Draw flag
        ctx.fillStyle = '#27ae60';
        ctx.beginPath();
        ctx.moveTo(flagX, flagY - 50);
        ctx.lineTo(flagX + 25, flagY - 35);
        ctx.lineTo(flagX, flagY - 20);
        ctx.fill();
        
        // Draw velocity arrow to show momentum
        if (Math.abs(state.velocity) > 0.01) {
            const arrowLength = Math.min(Math.abs(state.velocity) * 500, 40);
            const arrowDirection = state.velocity > 0 ? 1 : -1;
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(carScreenX, carY - 20);
            ctx.lineTo(carScreenX + arrowDirection * arrowLength, carY - 20);
            ctx.stroke();
            
            // Arrow head
            ctx.beginPath();
            ctx.moveTo(carScreenX + arrowDirection * arrowLength, carY - 20);
            ctx.lineTo(carScreenX + arrowDirection * (arrowLength - 8), carY - 25);
            ctx.lineTo(carScreenX + arrowDirection * (arrowLength - 8), carY - 15);
            ctx.fill();
        }
        
        // Add labels
        ctx.fillStyle = '#7f8c8d';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Position: ${state.position.toFixed(3)}`, carScreenX, carY - 40);
        ctx.fillText(`Velocity: ${state.velocity.toFixed(3)}`, carScreenX, carY - 55);
        ctx.fillText('GOAL', flagX, flagY - 65);
        ctx.fillText('(x ≥ 0.45)', flagX, flagY - 80);
    }

    drawPendulum(state) {
        const ctx = this.simulationCtx;
        const canvas = this.simulationCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Calculate pendulum center
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const pendulumLength = 120;
        
        // Get angle from state (note: state.angle is θ directly)
        const angle = state.angle;
        
        // Draw reference circle to show full range
        ctx.strokeStyle = 'rgba(189, 195, 199, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pendulumLength, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Draw upright target zone (highlighted sector around θ=0)
        const targetZone = 0.3; // ±0.3 radians around upright
        ctx.fillStyle = 'rgba(39, 174, 96, 0.1)';
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, pendulumLength + 10, -targetZone, targetZone);
        ctx.closePath();
        ctx.fill();
        
        // Draw target position indicator (upright)
        const targetX = centerX;
        const targetY = centerY - pendulumLength;
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(targetX, targetY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#27ae60';
        ctx.beginPath();
        ctx.arc(targetX, targetY, 10, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw pivot point
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
        ctx.fill();
        
        // Calculate pendulum bob position (θ=0 is upright, θ increases clockwise)
        const bobX = centerX + Math.sin(angle) * pendulumLength;
        const bobY = centerY - Math.cos(angle) * pendulumLength;
        
        // Draw pendulum rod
        ctx.strokeStyle = '#e67e22';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(bobX, bobY);
        ctx.stroke();
        
        // Draw pendulum bob (mass)
        ctx.fillStyle = '#e67e22';
        ctx.beginPath();
        ctx.arc(bobX, bobY, 16, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw angular velocity indicator (if significant)
        if (Math.abs(state.angular_velocity) > 0.5) {
            const velocityRadius = 40;
            const velocityAngle = Math.sign(state.angular_velocity) * Math.min(Math.abs(state.angular_velocity) / 8, 1) * Math.PI * 0.3;
            
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.arc(centerX, centerY, velocityRadius, angle - velocityAngle/2, angle + velocityAngle/2);
            ctx.stroke();
            
            // Arrow indicating direction
            const arrowX = centerX + Math.sin(angle + velocityAngle/2) * velocityRadius;
            const arrowY = centerY - Math.cos(angle + velocityAngle/2) * velocityRadius;
            ctx.fillStyle = '#e74c3c';
            ctx.beginPath();
            ctx.arc(arrowX, arrowY, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Show reward zones visualization
        const currentCost = angle * angle + 0.1 * state.angular_velocity * state.angular_velocity;
        ctx.fillStyle = `rgba(231, 76, 60, ${Math.min(currentCost / 10, 0.8)})`;
        ctx.beginPath();
        ctx.arc(bobX, bobY, 20, 0, 2 * Math.PI);
        ctx.fill();
        
        // Add labels and info
        ctx.fillStyle = '#7f8c8d';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`θ = ${(angle * 180 / Math.PI).toFixed(1)}°`, centerX, 30);
        ctx.fillText(`ω = ${state.angular_velocity.toFixed(2)} rad/s`, centerX, 50);
        
        // Show cost components
        const angleCost = angle * angle;
        const velocityCost = 0.1 * state.angular_velocity * state.angular_velocity;
        ctx.font = '12px Arial';
        ctx.fillText(`Angle Cost: ${angleCost.toFixed(3)}`, centerX, canvas.height - 60);
        ctx.fillText(`Velocity Cost: ${velocityCost.toFixed(3)}`, centerX, canvas.height - 40);
        ctx.fillText(`Total Cost: ${(angleCost + velocityCost).toFixed(3)}`, centerX, canvas.height - 20);
        
        // Target label
        ctx.fillText('TARGET', targetX, targetY - 20);
        ctx.fillText('(θ = 0°)', targetX, targetY - 35);
    }

    drawAcrobot(state) {
        const ctx = this.simulationCtx;
        const canvas = this.simulationCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Calculate positions
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const linkLength = 80; // Increased for better visibility
        
        // State: [θ1, θ2, θ1_dot, θ2_dot] where θ1 is shoulder, θ2 is elbow
        const theta1 = state.position; // First joint angle (shoulder)
        const theta2 = state.angle;     // Second joint angle (elbow relative to first link)
        
        // Calculate joint positions (θ=0 is hanging down, positive is clockwise)
        // First link end (elbow position)
        const joint1X = centerX + Math.sin(theta1) * linkLength;
        const joint1Y = centerY + Math.cos(theta1) * linkLength;
        
        // Second link end (hand/foot position) - angle is relative to first link
        const endX = joint1X + Math.sin(theta1 + theta2) * linkLength;
        const endY = joint1Y + Math.cos(theta1 + theta2) * linkLength;
        
        // Draw goal height threshold line
        const goalHeight = centerY - linkLength * 1.8; // Height target for the foot
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        ctx.beginPath();
        ctx.moveTo(centerX - 120, goalHeight);
        ctx.lineTo(centerX + 120, goalHeight);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Goal zone background
        ctx.fillStyle = 'rgba(39, 174, 96, 0.1)';
        ctx.fillRect(0, 0, canvas.width, goalHeight);
        
        // Draw pivot point (shoulder)
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw first link (upper arm)
        ctx.strokeStyle = '#16a085';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(joint1X, joint1Y);
        ctx.stroke();
        
        // Draw elbow joint
        ctx.fillStyle = '#16a085';
        ctx.beginPath();
        ctx.arc(joint1X, joint1Y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw second link (forearm) - this is the actuated joint
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(joint1X, joint1Y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Draw end effector (hand/foot) - this is what needs to reach the goal
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(endX, endY, 12, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw trajectory trace (if moving)
        if (Math.abs(state.velocity) > 0.1 || Math.abs(state.angular_velocity) > 0.1) {
            ctx.strokeStyle = 'rgba(231, 76, 60, 0.3)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            // Simple trail effect
            for (let i = 1; i <= 3; i++) {
                const trailX = endX - state.velocity * i * 5;
                const trailY = endY - state.angular_velocity * i * 5;
                if (i === 1) ctx.moveTo(trailX, trailY);
                else ctx.lineTo(trailX, trailY);
            }
            ctx.stroke();
        }
        
        // Show angular velocity indicators
        if (Math.abs(state.velocity) > 0.5) {
            // First joint velocity indicator
            const radius1 = 30;
            const velArc1 = Math.sign(state.velocity) * Math.min(Math.abs(state.velocity) / 5, 1) * Math.PI * 0.2;
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius1, theta1 - velArc1/2, theta1 + velArc1/2);
            ctx.stroke();
        }
        
        if (Math.abs(state.angular_velocity) > 0.5) {
            // Second joint velocity indicator
            const radius2 = 25;
            const velArc2 = Math.sign(state.angular_velocity) * Math.min(Math.abs(state.angular_velocity) / 5, 1) * Math.PI * 0.2;
            ctx.strokeStyle = '#f39c12';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(joint1X, joint1Y, radius2, theta1 + theta2 - velArc2/2, theta1 + theta2 + velArc2/2);
            ctx.stroke();
        }
        
        // Height indicator line from foot to goal
        ctx.strokeStyle = endY <= goalHeight ? '#27ae60' : '#e74c3c';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX, goalHeight);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Add detailed labels
        ctx.fillStyle = '#7f8c8d';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Shoulder: ${(theta1 * 180 / Math.PI).toFixed(1)}°`, centerX, 25);
        ctx.fillText(`Elbow: ${(theta2 * 180 / Math.PI).toFixed(1)}°`, centerX, 45);
        
        // Joint velocities
        ctx.font = '12px Arial';
        ctx.fillText(`ω₁: ${state.velocity.toFixed(2)} rad/s`, centerX, 65);
        ctx.fillText(`ω₂: ${state.angular_velocity.toFixed(2)} rad/s`, centerX, 80);
        
        // Height information
        const heightToGo = Math.max(0, endY - goalHeight);
        ctx.fillText(`Height to goal: ${heightToGo.toFixed(0)}px`, endX, endY + 25);
        
        // Goal status
        if (endY <= goalHeight) {
            ctx.fillStyle = '#27ae60';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('🎯 GOAL REACHED!', centerX, canvas.height - 20);
        } else {
            ctx.fillStyle = '#e74c3c';
            ctx.font = '12px Arial';
            ctx.fillText('Swing foot above goal line', centerX, canvas.height - 20);
        }
        
        // Goal line label
        ctx.fillStyle = '#27ae60';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('GOAL HEIGHT', centerX - 115, goalHeight - 5);
        
        // Joint labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('SHOULDER\n(passive)', centerX - 20, centerY + 20);
        ctx.fillStyle = '#e74c3c';
        ctx.fillText('ELBOW\n(actuated)', joint1X + 15, joint1Y);
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
                console.error('Failed to fetch reward history:', error);
            }
        }, 1000);
    }
}

// Initialize visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MultiEnvironmentVisualizer();
});