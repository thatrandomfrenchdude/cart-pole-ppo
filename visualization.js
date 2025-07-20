class CartPoleVisualizer {
    constructor() {
        this.cartpoleCanvas = document.getElementById('cartpole-canvas');
        this.rewardCanvas = document.getElementById('reward-canvas');
        this.cartpoleCtx = this.cartpoleCanvas.getContext('2d');
        this.rewardCtx = this.rewardCanvas.getContext('2d');
        
        this.rewardHistory = [];
        this.maxRewards = 100;
        
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
    
    resizeCanvases() {
        // Get the container width
        const cartpoleContainer = this.cartpoleCanvas.parentElement;
        const rewardContainer = this.rewardCanvas.parentElement;
        
        const containerWidth = cartpoleContainer.offsetWidth - 40; // Account for padding
        
        // Resize cartpole canvas
        this.cartpoleCanvas.width = containerWidth;
        this.cartpoleCanvas.height = 300;
        
        // Resize reward canvas  
        this.rewardCanvas.width = containerWidth;
        this.rewardCanvas.height = 200;
        
        // Reinitialize after resize
        this.initializeCanvases();
    }
    
    initializeCanvases() {
        // Set up cartpole canvas
        this.cartpoleCtx.fillStyle = '#f8f9fa';
        this.cartpoleCtx.fillRect(0, 0, this.cartpoleCanvas.width, this.cartpoleCanvas.height);
        
        // Set up reward canvas
        this.rewardCtx.fillStyle = '#f8f9fa';
        this.rewardCtx.fillRect(0, 0, this.rewardCanvas.width, this.rewardCanvas.height);
    }
    
    drawCartPole(state) {
        const ctx = this.cartpoleCtx;
        const canvas = this.cartpoleCanvas;
        
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
        // Update info panel
        document.getElementById('episode').textContent = data.episode || 0;
        document.getElementById('timestep').textContent = data.timestep || 0;
        document.getElementById('position').textContent = data.position.toFixed(3);
        document.getElementById('velocity').textContent = data.velocity.toFixed(3);
        document.getElementById('angle').textContent = (data.angle * 180 / Math.PI).toFixed(1) + '°';
        document.getElementById('angular-velocity').textContent = data.angular_velocity.toFixed(3);
        document.getElementById('reward').textContent = data.reward.toFixed(3);
        
        // Draw cart-pole
        this.drawCartPole(data);
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
    new CartPoleVisualizer();
});
