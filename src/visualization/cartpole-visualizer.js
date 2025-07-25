/**
 * CartPole Visualizer
 * Handles visualization for the CartPole environment
 */
class CartPoleVisualizer extends BaseVisualizer {
    constructor() {
        super();
        this.currentEnvironment = 'cartpole';
        this.updateEnvironmentTheme('cartpole');
    }
    
    drawEnvironment(state) {
        this.drawCartPole(state);
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
}
