/**
 * Acrobot Visualizer
 * Handles visualization for the Acrobot environment
 */
class AcrobotVisualizer extends BaseVisualizer {
    constructor() {
        super();
        this.currentEnvironment = 'acrobot';
        this.updateEnvironmentTheme('acrobot');
    }
    
    drawEnvironment(state) {
        this.drawAcrobot(state);
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
}
