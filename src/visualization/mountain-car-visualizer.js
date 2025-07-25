/**
 * Mountain Car Visualizer
 * Handles visualization for the Mountain Car environment
 */
class MountainCarVisualizer extends BaseVisualizer {
    constructor() {
        super();
        this.currentEnvironment = 'mountain_car';
        this.updateEnvironmentTheme('mountain_car');
    }
    
    drawEnvironment(state) {
        this.drawMountainCar(state);
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
}
