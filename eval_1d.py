#!/usr/bin/env python3
"""
Evaluate trained 1D Lunar Lander agent
"""

import gymnasium as gym
import torch
import numpy as np
import lunar_lander_1d
from train_1d import ActorCritic

def evaluate(model_path, num_episodes=10):
    """Evaluate a trained agent."""
    env = gym.make('LunarLander1D-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = ActorCritic(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    rewards = []
    successes = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        print(f"\n=== Episode {ep+1} ===")
        print(f"Initial: Alt={state[0]:.1f}m, Vel={state[1]:.2f}m/s, Fuel={state[2]:.1f}kg")

        while not done and steps < 1000:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = policy.act(state_tensor)
                action = action.cpu().numpy()[0]

            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

            # Print every 20 steps
            if steps % 20 == 0 or done:
                print(f"Step {steps:3d}: Alt={state[0]:7.1f}m, Vel={state[1]:6.2f}m/s, "
                      f"Fuel={state[2]:5.1f}kg, Thrust={action[0]:.3f}")

        print(f"Final: Alt={state[0]:.1f}m, Vel={state[1]:.2f}m/s, Fuel={state[2]:.1f}kg")
        print(f"Episode reward: {episode_reward:.2f}")

        # Check if it was a safe landing
        if abs(state[1]) < 2.0 and state[0] <= 0:
            print("SUCCESS: Safe landing!")
            successes += 1
        else:
            print("FAILED: Crash!")

        rewards.append(episode_reward)

    print(f"\n=== Summary ===")
    print(f"Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")

    env.close()

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/ppo_1d_final.pt'
    evaluate(model_path, num_episodes=5)
