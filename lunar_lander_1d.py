"""
1D Lunar Lander Environment - MVP
A simple vertical landing environment where the agent must learn to slow descent
and land safely on the lunar surface.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image


class LunarLander1D(gym.Env):
    """
    1D Lunar Lander Environment

    Observation Space:
        - altitude (m): distance from surface [0, max_altitude]
        - velocity (m/s): vertical velocity (negative = descending)
        - fuel_remaining (kg): fuel available [0, initial_fuel]

    Action Space:
        - thrust_level: continuous [0, 1] scaled to [0, max_thrust]

    Episode Termination:
        - Altitude <= 0 (landed or crashed)
        - Fuel depleted
        - Max steps reached

    Rewards:
        - +100 for safe landing (velocity < safe_landing_velocity)
        - -100 for crash (velocity >= safe_landing_velocity)
        - Small penalty for fuel consumption
        - Shaped reward for reducing velocity
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        # Physical constants
        self.lunar_gravity = 1.62  # m/s^2
        self.dt = 0.016666667  # ~60 Hz simulation (16ms timestep)

        # Lander parameters
        self.dry_mass = 1000.0  # kg (mass without fuel)
        self.initial_fuel = 100.0  # kg
        self.max_thrust = 5000.0  # N
        self.fuel_consumption_rate = 1.0  # kg/s at max thrust

        # Episode parameters
        self.initial_altitude_range = (100.0, 500.0)  # m (reduced for faster episodes)
        self.initial_velocity_range = (-30.0, -10.0)  # m/s (negative = falling)
        self.safe_landing_velocity = 2.0  # m/s

        # Observation space: [altitude, velocity, fuel_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -200.0, 0.0], dtype=np.float32),
            high=np.array([3000.0, 100.0, self.initial_fuel], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: thrust_level [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State variables
        self.altitude = 0.0
        self.velocity = 0.0
        self.fuel = 0.0
        self.steps = 0
        self.last_thrust = 0.0

        self.render_mode = render_mode

        # For rendering
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial conditions
        self.altitude = self.np_random.uniform(*self.initial_altitude_range)
        self.velocity = self.np_random.uniform(*self.initial_velocity_range)
        self.fuel = self.initial_fuel
        self.steps = 0
        self.last_thrust = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Extract thrust level and clip to valid range
        thrust_level = np.clip(action[0], 0.0, 1.0)
        self.last_thrust = thrust_level  # Store for rendering

        # Calculate current mass
        current_mass = self.dry_mass + self.fuel

        # Calculate thrust force
        thrust_force = thrust_level * self.max_thrust

        # Fuel consumption (proportional to thrust)
        fuel_consumed = thrust_level * self.fuel_consumption_rate * self.dt
        self.fuel = max(0.0, self.fuel - fuel_consumed)

        # If out of fuel, no thrust
        if self.fuel <= 0:
            thrust_force = 0.0

        # Physics: F = ma => a = F/m - g
        thrust_acceleration = thrust_force / current_mass
        net_acceleration = thrust_acceleration - self.lunar_gravity

        # Update velocity and position (Euler integration)
        self.velocity += net_acceleration * self.dt
        self.altitude += self.velocity * self.dt

        self.steps += 1

        # Check termination conditions
        terminated = False
        reward = 0.0

        # Small time penalty to encourage landing quickly
        reward = -0.01

        # Check if lander has landed or crashed
        if self.altitude <= 0:
            terminated = True
            landing_speed = abs(self.velocity)

            if landing_speed < self.safe_landing_velocity:
                # Safe landing - SUCCESS!
                reward = 100.0
            else:
                # Crashed - FAILURE!
                reward = -100.0

            self.altitude = 0.0

        # Note: Timeout is handled by Gymnasium's TimeLimit wrapper via max_episode_steps

        observation = self._get_obs()
        info = self._get_info()
        info['termination_reason'] = self._get_termination_reason(terminated)

        return observation, reward, terminated, False, info
    
    def _get_weighted_velocity_reward_by_height(self, altitude, velocity):
        scaling_constant_k = 0.05 # scaling constant (no idea)
        epsilon = 0.0001 # prevent divide by 0
        weight = scaling_constant_k / (altitude + epsilon)
        return weight * -abs(velocity - 0.05)

    def _get_obs(self):
        """Get current observation."""
        return np.array([self.altitude, self.velocity, self.fuel], dtype=np.float32)

    def _get_info(self):
        """Get additional info."""
        return {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "fuel": self.fuel,
            "steps": self.steps,
            "mass": self.dry_mass + self.fuel,
        }

    def _get_termination_reason(self, terminated):
        """Get the reason for episode termination."""
        if not terminated:
            return "ongoing"
        elif self.altitude <= 0:
            landing_speed = abs(self.velocity)
            if landing_speed < self.safe_landing_velocity:
                return "success_landing"
            else:
                return "failure_crash"
        else:
            return "unknown"

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Alt: {self.altitude:7.1f}m | Vel: {self.velocity:6.2f}m/s | Fuel: {self.fuel:5.1f}kg")
            return None

        elif self.render_mode == "rgb_array":
            # Create figure if it doesn't exist
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(6, 8))

            self.ax.clear()

            # Set up the plot
            max_alt = self.initial_altitude_range[1]
            self.ax.set_xlim(-50, 50)
            self.ax.set_ylim(-10, max_alt + 50)
            self.ax.set_aspect('equal')

            # Draw ground
            self.ax.axhline(y=0, color='gray', linewidth=3, label='Surface')
            self.ax.fill_between([-50, 50], -10, 0, color='gray', alpha=0.3)

            # Draw lander as a rectangle
            lander_width = 10
            lander_height = 15
            lander_x = -lander_width / 2
            lander_y = self.altitude

            # Lander color based on status
            if self.altitude <= 0:
                if abs(self.velocity) < self.safe_landing_velocity:
                    lander_color = 'green'  # Safe landing
                else:
                    lander_color = 'red'  # Crash
            else:
                lander_color = 'blue'  # Flying

            lander = patches.Rectangle(
                (lander_x, lander_y), lander_width, lander_height,
                linewidth=2, edgecolor='black', facecolor=lander_color
            )
            self.ax.add_patch(lander)

            # Draw thrust indicator
            if hasattr(self, 'last_thrust'):
                thrust_length = self.last_thrust * 20  # Scale for visibility
                if thrust_length > 0:
                    self.ax.arrow(0, lander_y, 0, -thrust_length,
                                head_width=3, head_length=2, fc='orange', ec='red')

            # Add telemetry text
            telemetry = f"Altitude: {self.altitude:.1f}m\n"
            telemetry += f"Velocity: {self.velocity:.2f}m/s\n"
            telemetry += f"Fuel: {self.fuel:.1f}kg\n"
            telemetry += f"Steps: {self.steps}"

            self.ax.text(0.02, 0.98, telemetry,
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=10, family='monospace')

            self.ax.set_xlabel('Lateral Position')
            self.ax.set_ylabel('Altitude (m)')
            self.ax.set_title('1D Lunar Lander')
            self.ax.grid(True, alpha=0.3)

            # Convert plot to RGB array
            buf = BytesIO()
            self.fig.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            img = Image.open(buf)
            # Convert to RGB (remove alpha channel if present)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_array = np.array(img)
            buf.close()

            return img_array

        return None

    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Register environment with Gymnasium
gym.register(
    id='LunarLander1D-v0',
    entry_point='lunar_lander_1d:LunarLander1D',
    max_episode_steps=3000,  # Only place to configure episode timeout
)


if __name__ == "__main__":
    # Quick test of the environment
    env = LunarLander1D()
    obs, info = env.reset()

    print("Testing 1D Lunar Lander Environment")
    print(f"Initial state: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test a few random steps
    print("\nRunning test episode with random actions:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action[0]:.2f}, alt={obs[0]:.1f}m, vel={obs[1]:.2f}m/s, fuel={obs[2]:.1f}kg, reward={reward:.2f}")

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\nEnvironment test complete!")
