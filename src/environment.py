import numpy as np
import math


class CartPoleEnv:
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
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        reward = 1.0 if not done else 0.0
        return self.state, reward, done
