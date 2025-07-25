/**
 * Pendulum Environment Drawing Function
 * Draws the pendulum visualization
 */
function drawPendulum(ctx, canvas, state) {
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
