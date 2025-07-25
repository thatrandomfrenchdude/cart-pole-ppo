import numpy as np
import math
from typing import Dict, Any, Tuple


class EnvironmentFactory:
    """Factory class to create different environments based on configuration."""
    
    @staticmethod
    def create_environment(config: Dict[str, Any]):
        """Create an environment based on the game configuration."""
        game_type = config['game']['environment'].lower()
        
        if game_type == 'cartpole':
            return CartPoleEnv(config)
        elif game_type == 'mountain_car':
            return MountainCarEnv(config)
        elif game_type == 'pendulum':
            return PendulumEnv(config)
        elif game_type == 'acrobot':
            return AcrobotEnv(config)
        else:
            raise ValueError(f"Unknown environment: {game_type}")
    
    @staticmethod
    def get_environment_specs(config: Dict[str, Any]) -> Dict[str, int]:
        """Get the input and output dimensions for the current environment."""
        game_type = config['game']['environment'].lower()
        
        if game_type == 'cartpole':
            return {'input_dim': 4, 'output_dim': 2}  # Discrete actions: left/right
        elif game_type == 'mountain_car':
            return {'input_dim': 2, 'output_dim': 1}  # Continuous action: force
        elif game_type == 'pendulum':
            return {'input_dim': 3, 'output_dim': 1}  # Continuous action: torque
        elif game_type == 'acrobot':
            return {'input_dim': 4, 'output_dim': 3}  # Discrete actions: -1/0/+1 torque
        else:
            raise ValueError(f"Unknown environment: {game_type}")


class CartPoleEnv:
    """Cart-Pole balancing environment."""
    
    def __init__(self, config):
        env_config = config['environment']
        self.gravity = env_config['gravity']
        self.masscart = env_config['cart_mass']
        self.masspole = env_config['pole_mass']
        self.total_mass = self.masspole + self.masscart
        self.length = env_config['pole_half_length']  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = env_config['force_magnitude']
        self.tau = env_config['time_step']  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = env_config['angle_threshold_degrees'] * 2 * math.pi / 360
        self.x_threshold = env_config['position_threshold']
        
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # Correct CartPole dynamics according to specification
        # F is the force applied to the cart (positive = right, negative = left)
        F = force
        
        # Angular acceleration according to the spec formula
        temp = (-F - self.masspole * self.length * theta_dot ** 2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta + costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / (self.masscart + self.masspole))
        )
        
        # Cart acceleration according to the spec formula
        xacc = (F + self.masspole * self.length * (theta_dot ** 2 * sintheta - thetaacc * costheta)) / (self.masscart + self.masspole)
        
        # Euler integration
        x_dot = x_dot + xacc * self.tau
        x = x + x_dot * self.tau
        theta_dot = theta_dot + thetaacc * self.tau
        theta = theta + theta_dot * self.tau
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        reward = 1.0 if not done else 0.0
        return self.state, reward, done


class MountainCarEnv:
    """Mountain Car Continuous environment."""
    
    def __init__(self, config):
        env_config = config['mountain_car']
        self.min_position = env_config['min_position']
        self.max_position = env_config['max_position']
        self.max_speed = env_config['max_speed']
        self.goal_position = env_config['goal_position']
        self.goal_velocity = env_config['goal_velocity']
        self.force = env_config['force']
        self.gravity = env_config['gravity']
        self.tau = env_config['time_step']
        
        self.step_count = 0
        self.max_steps = 999  # Episode horizon
        
        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        
        self.reset()
    
    def reset(self):
        # Start at random position in valley with zero velocity
        self.state = np.array([
            np.random.uniform(low=-0.6, high=-0.4),
            0.0
        ])
        self.step_count = 0
        return self.state
    
    def step(self, action):
        position, velocity = self.state
        
        # Handle continuous action (should be between -1 and 1)
        if isinstance(action, (list, np.ndarray)):
            a = np.clip(action[0], -1.0, 1.0)
        else:
            a = np.clip(action, -1.0, 1.0)
        
        # Physics update according to specification:
        # v := v + 0.001*a - 0.0025*cos(3*x)
        # x := x + v
        velocity = velocity + 0.001 * a - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = position + velocity
        
        # Handle boundaries
        if position < self.min_position:
            position = self.min_position
            velocity = 0.0
        elif position > self.max_position:
            position = self.max_position
            velocity = 0.0
        
        self.state = np.array([position, velocity])
        self.step_count += 1
        
        # Check if goal is reached (x >= 0.45)
        goal_reached = bool(position >= self.goal_position)
        
        # Check if max steps reached
        max_steps_reached = bool(self.step_count >= self.max_steps)
        
        done = goal_reached or max_steps_reached
        
        # Reward structure: -0.1*a^2 each step, +100 bonus for reaching goal
        reward = -0.1 * (a ** 2)
        if goal_reached:
            reward += 100.0
        
        return self.state, reward, done


class PendulumEnv:
    """Pendulum swing-up environment with continuous control."""
    
    def __init__(self, config):
        env_config = config['pendulum']
        self.max_speed = env_config['max_speed']
        self.max_torque = env_config['max_torque']
        self.tau = env_config['time_step']
        self.gravity = env_config['gravity']
        self.mass = env_config['mass']
        self.length = env_config['length']
        
        self.step_count = 0
        self.max_steps = 200  # Fixed horizon
        
        self.reset()
    
    def reset(self):
        # Start hanging down with small random perturbations
        theta = np.pi + np.random.uniform(low=-0.1, high=0.1)  # Start near hanging down position
        theta_dot = np.random.uniform(low=-0.1, high=0.1)  # Small initial velocity
        self.state = np.array([theta, theta_dot])
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self):
        """Convert angle and angular velocity to observation."""
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot])
    
    def step(self, action):
        theta, theta_dot = self.state
        
        # Handle continuous action (should be between -1 and 1, then scaled to max_torque)
        if isinstance(action, (list, np.ndarray)):
            u = np.clip(action[0], -1.0, 1.0) * self.max_torque
        else:
            u = np.clip(action, -1.0, 1.0) * self.max_torque
        
        # Pendulum dynamics according to specification:
        # θ̈ = -3g/(2l) * sin(θ) + 3/(ml²) * u
        g = self.gravity
        m = self.mass
        l = self.length
        
        theta_ddot = -3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l ** 2) * u
        
        # Euler integration
        theta_dot_new = theta_dot + theta_ddot * self.tau
        theta_dot_new = np.clip(theta_dot_new, -self.max_speed, self.max_speed)
        
        theta_new = theta + theta_dot_new * self.tau
        
        # Normalize angle to [-pi, pi]
        theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
        
        self.state = np.array([theta_new, theta_dot_new])
        self.step_count += 1
        
        # New reward function based on angle relative to upright position
        # Upright position: theta = 0 (reward = 1)
        # Hanging down: theta = ±π (reward = -1)
        # Continuous reward function that increases as pendulum moves towards upright
        
        # Calculate angle distance from upright (0) position
        angle_from_upright = abs(theta_new)
        
        # Use cosine to create smooth transition: cos(0) = 1, cos(π) = -1
        position_reward = np.cos(angle_from_upright)
        
        # Add small penalty for high angular velocity to encourage stability
        velocity_penalty = -0.01 * (theta_dot_new ** 2)
        
        # Add small penalty for large control actions to encourage efficiency
        control_penalty = -0.001 * (u ** 2)
        
        # Total reward
        reward = position_reward + velocity_penalty + control_penalty
        
        # Episodes run for fixed horizon
        done = bool(self.step_count >= self.max_steps)
        
        return self._get_obs(), reward, done


class AcrobotEnv:
    """Acrobot (double pendulum) environment."""
    
    def __init__(self, config):
        env_config = config['acrobot']
        self.LINK_LENGTH_1 = env_config['link_length_1']
        self.LINK_LENGTH_2 = env_config['link_length_2']
        self.LINK_MASS_1 = env_config['link_mass_1']
        self.LINK_MASS_2 = env_config['link_mass_2']
        self.LINK_COM_POS_1 = env_config['link_com_pos_1']
        self.LINK_COM_POS_2 = env_config['link_com_pos_2']
        self.LINK_MOI = env_config['link_moi']
        self.MAX_VEL_1 = env_config['max_vel_1']
        self.MAX_VEL_2 = env_config['max_vel_2']
        self.AVAIL_TORQUE = [-1.0, 0.0, +1.0]
        self.torque_noise_max = env_config['torque_noise_max']
        self.tau = env_config['time_step']
        self.gravity = env_config['gravity']
        
        # Goal: reach height threshold (end-effector above first joint level)
        # Standard Acrobot goal: get end-effector above the level of the first joint
        self.goal_height = self.LINK_LENGTH_1
        
        self.step_count = 0
        self.max_steps = 500  # Episode horizon
        
        self.reset()
    
    def reset(self):
        # Start hanging down with small random perturbations
        # State: [theta1, theta2, theta1_dot, theta2_dot]
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        self.state[0] += np.pi  # First link starts hanging down
        self.step_count = 0
        return self.state
    
    def step(self, action):
        # Get torque from discrete action
        torque = self.AVAIL_TORQUE[action]
        
        # Add noise if specified
        if self.torque_noise_max > 0:
            torque += np.random.uniform(-self.torque_noise_max, self.torque_noise_max)
        
        # Current state
        theta1, theta2, theta1_dot, theta2_dot = self.state
        
        # Acrobot dynamics using Lagrangian mechanics
        # Simplified version based on double pendulum with only second joint actuated
        
        # Constants for cleaner notation
        m1, m2 = self.LINK_MASS_1, self.LINK_MASS_2
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        lc1, lc2 = self.LINK_COM_POS_1, self.LINK_COM_POS_2
        I1, I2 = self.LINK_MOI, self.LINK_MOI
        g = self.gravity
        
        # Inertia matrix elements
        d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d12 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
        d22 = m2 * lc2**2 + I2
        
        # Coriolis and centrifugal terms
        h1 = -m2 * l1 * lc2 * np.sin(theta2) * theta2_dot**2 - 2 * m2 * l1 * lc2 * np.sin(theta2) * theta1_dot * theta2_dot
        h2 = m2 * l1 * lc2 * np.sin(theta2) * theta1_dot**2
        
        # Gravity terms
        phi1 = (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi/2) + m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi/2)
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi/2)
        
        # Total forces
        tau1 = 0.0  # No torque on first joint
        tau2 = torque  # Torque on second joint
        
        f1 = tau1 - h1 - phi1
        f2 = tau2 - h2 - phi2
        
        # Solve for accelerations: [d11 d12; d12 d22] * [theta1_ddot; theta2_ddot] = [f1; f2]
        det = d11 * d22 - d12**2
        if abs(det) < 1e-6:
            det = 1e-6  # Avoid division by zero
        
        theta1_ddot = (d22 * f1 - d12 * f2) / det
        theta2_ddot = (d11 * f2 - d12 * f1) / det
        
        # Integrate using Euler method
        theta1_dot_new = theta1_dot + theta1_ddot * self.tau
        theta2_dot_new = theta2_dot + theta2_ddot * self.tau
        
        # Clip velocities
        theta1_dot_new = np.clip(theta1_dot_new, -self.MAX_VEL_1, self.MAX_VEL_1)
        theta2_dot_new = np.clip(theta2_dot_new, -self.MAX_VEL_2, self.MAX_VEL_2)
        
        theta1_new = theta1 + theta1_dot_new * self.tau
        theta2_new = theta2 + theta2_dot_new * self.tau
        
        # Wrap angles to [-pi, pi]
        theta1_new = wrap_angle(theta1_new)
        theta2_new = wrap_angle(theta2_new)
        
        self.state = np.array([theta1_new, theta2_new, theta1_dot_new, theta2_dot_new])
        self.step_count += 1
        
        # Check if goal is reached (end-effector height)
        # End-effector position: foot of second link
        # Height should be positive when above the pivot (goal) and negative when below
        y = l1 * np.cos(theta1_new) + l2 * np.cos(theta1_new + theta2_new)
        
        goal_reached = bool(y >= self.goal_height)
        max_steps_reached = bool(self.step_count >= self.max_steps)
        
        done = goal_reached or max_steps_reached
        
        # Sparse reward: -1 each step until goal is reached
        reward = 0.0 if goal_reached else -1.0
        
        return self.state, reward, done


def angle_normalize(x):
    """Normalize angle to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
