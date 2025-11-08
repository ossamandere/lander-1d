#!/usr/bin/env python3
"""
PPO Training Script for 1D Lunar Lander

This script trains a PPO agent on the 1D vertical landing task.
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Import our custom 1D environment
import lunar_lander_1d


class ActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor network (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output [0, 1] for thrust level
        )

        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        return features

    def act(self, state):
        """Sample action from policy."""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        # Clip to [0, 1] range
        action = torch.clamp(action, 0.0, 1.0)
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action, action_log_prob

    def evaluate(self, state, action):
        """Evaluate action probabilities and state value."""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)

        dist = Normal(action_mean, action_std)
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        state_value = self.critic(features).squeeze()

        return action_log_prob, state_value, entropy


class RolloutBuffer:
    """Store trajectories for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.dones)),
        )


class PPO:
    """Proximal Policy Optimization agent."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=10, batch_size=64, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """Select action for interaction with environment."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.act(state)
            features = self.policy.forward(state)
            value = self.policy.critic(features).squeeze()

        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        values = values.tolist() + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)

    def update(self, next_state):
        """Update policy using PPO."""
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            features = self.policy.forward(next_state)
            next_value = self.policy.critic(features).squeeze().cpu().item()

        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, state_values, entropy = self.policy.evaluate(batch_states, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()

        return actor_loss.item(), critic_loss.item()

    def save(self, filepath):
        """Save model weights to disk."""
        torch.save(self.policy.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights from disk."""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


def record_video_episode(agent, video_dir, episode_num, device):
    """Record a single episode as video."""
    # Create environment with video recording wrapper
    env = gym.make('LunarLander1D-v0', render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=f'episode_{episode_num}',
        episode_trigger=lambda x: True  # Record every episode (we only run 1)
    )

    state, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _ = agent.policy.act(state_tensor)
            action = action.cpu().numpy()[0]

        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()
    termination_reason = info.get('termination_reason', 'unknown')
    print(f"  Video recorded: Episode {episode_num}, Reward: {episode_reward:.2f}, Steps: {steps}, Outcome: {termination_reason}")
    return episode_reward


def train(args):
    """Main training loop."""
    # Create the 1D Lunar Lander environment
    env = gym.make('LunarLander1D-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Environment: LunarLander1D-v0")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        device=device
    )

    episode_rewards = []
    avg_rewards = []

    os.makedirs(args.save_dir, exist_ok=True)

    # Create video directory
    video_dir = os.path.join(args.save_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    # Calculate video recording episodes (25%, 50%, 100%)
    video_episodes = [
        int(args.episodes * 0.25),
        int(args.episodes * 0.50),
        args.episodes
    ]
    print(f"Videos will be recorded at episodes: {video_episodes}")

    # Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'PPO_LunarLander1D_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Rollout length: {args.rollout_length} steps\n")

    # Log hyperparameters
    writer.add_text('hyperparameters/lr', str(args.lr), 0)
    writer.add_text('hyperparameters/gamma', str(args.gamma), 0)
    writer.add_text('hyperparameters/gae_lambda', str(args.gae_lambda), 0)
    writer.add_text('hyperparameters/clip_epsilon', str(args.clip_epsilon), 0)
    writer.add_text('hyperparameters/rollout_length', str(args.rollout_length), 0)
    writer.add_text('hyperparameters/batch_size', str(args.batch_size), 0)
    writer.add_text('hyperparameters/ppo_epochs', str(args.ppo_epochs), 0)

    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    step_count = 0
    episode_length = 0

    for total_steps in range(args.episodes * args.rollout_length):
        step_count += 1
        episode_length += 1

        action, log_prob, value = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, log_prob, value, done)

        episode_reward += reward
        state = next_state

        if step_count % args.rollout_length == 0:
            actor_loss, critic_loss = agent.update(next_state)

            writer.add_scalar('Loss/actor', actor_loss, total_steps)
            writer.add_scalar('Loss/critic', critic_loss, total_steps)

            action_std = torch.exp(agent.policy.actor_log_std).mean().item()
            writer.add_scalar('Policy/action_std', action_std, total_steps)

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)

            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)

            writer.add_scalar('Episode/reward', episode_reward, episode_count)
            writer.add_scalar('Episode/avg_reward_100', avg_reward, episode_count)
            writer.add_scalar('Episode/length', episode_length, episode_count)

            if episode_count % args.log_interval == 0:
                print(f"Episode {episode_count:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(100): {avg_reward:7.2f}")

            if episode_count % args.save_interval == 0:
                agent.save(os.path.join(args.save_dir, f'ppo_1d_episode_{episode_count}.pt'))

            # Record video at milestone episodes
            if episode_count in video_episodes:
                print(f"\n[Recording video at episode {episode_count}]")
                record_video_episode(agent, video_dir, episode_count, device)

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Consider solved if average reward >= 80 over 100 episodes
            if len(episode_rewards) >= 100 and avg_reward >= 80:
                print(f"\nEnvironment solved in {episode_count} episodes!")
                print(f"Average reward: {avg_reward:.2f}")
                agent.save(os.path.join(args.save_dir, 'ppo_1d_solved.pt'))
                break

    agent.save(os.path.join(args.save_dir, 'ppo_1d_final.pt'))

    # Create visualization
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (100 episodes)')
    plt.axhline(y=80, color='r', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress - LunarLander1D')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'training_curve_1d.png'))
    print(f"\nTraining curve saved to {os.path.join(args.save_dir, 'training_curve_1d.png')}")

    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")
    print(f"View with: tensorboard --logdir runs")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on 1D Lunar Lander')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--rollout_length', type=int, default=512, help='Steps before policy update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PPO updates')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO update epochs')

    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip parameter')

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10, help='Episodes between logs')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between saves')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')

    # Device configuration
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')

    args = parser.parse_args()

    train(args)
