# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements reinforcement learning algorithms for the LunarLander-v3 continuous control environment from Gymnasium. The continuous action space makes this more challenging than the discrete version, requiring algorithms that can handle continuous control (e.g., DDPG, TD3, SAC, PPO).

## Environment Setup

The project uses a Python virtual environment located in `venv/`.

**Activate the virtual environment:**
```bash
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Dependencies:**
- gymnasium>=1.2.0 - For the LunarLander environment
- torch>=2.0.0 - Deep learning framework
- numpy>=2.0.0 - Numerical computations
- matplotlib>=3.10.0 - Plotting training curves
- pygame>=2.6.0 - Rendering support
- box2d>=2.3.10 - Physics engine for LunarLander
- tensorboard>=2.20.0 - Training visualization and monitoring

## Development Commands

**Run training with default parameters:**
```bash
python train.py
```

**Quick test run (5 episodes):**
```bash
python train.py --episodes 5 --log_interval 1 --rollout_length 512
```

**Run with custom hyperparameters:**
```bash
python train.py --episodes 2000 --lr 3e-4 --gamma 0.99 --rollout_length 2048
```

**View training progress with TensorBoard:**
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser.

**Available training arguments:**
- `--episodes`: Number of episodes to train (default: 2000)
- `--rollout_length`: Steps before policy update (default: 2048)
- `--batch_size`: Batch size for PPO updates (default: 64)
- `--ppo_epochs`: PPO update epochs (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--gae_lambda`: GAE lambda (default: 0.95)
- `--clip_epsilon`: PPO clip parameter (default: 0.2)
- `--log_interval`: Episodes between logs (default: 10)
- `--save_interval`: Episodes between saves (default: 100)
- `--save_dir`: Directory to save models (default: models)
- `--cpu`: Force CPU usage (by default uses CUDA if available)

**TensorBoard Metrics:**

The training script automatically logs the following metrics to `./runs` directory:
- `Episode/reward` - Reward for each episode
- `Episode/avg_reward_100` - Moving average of last 100 episodes
- `Episode/length` - Number of steps per episode
- `Loss/actor` - Policy (actor) loss
- `Loss/critic` - Value (critic) loss
- `Policy/action_std` - Action standard deviation (exploration level)
- `hyperparameters/*` - All training hyperparameters

## Architecture Notes

### LunarLander-v3 Continuous Environment

- **Observation space:** 8-dimensional continuous vector (position, velocity, angle, angular velocity, leg contact)
- **Action space:** 2-dimensional continuous vector (main engine, side engines)
- **Reward structure:** Shaped reward based on position, velocity, and landing success
- **Episode termination:** Landing, crashing, or going out of bounds

### Project Structure

**train.py** - Main training script implementing PPO algorithm with:
- `ActorCritic`: Combined actor-critic neural network
  - Shared feature extraction layer (256 units)
  - Actor network outputs continuous actions via tanh activation
  - Learnable log standard deviation for action exploration
  - Critic network outputs state values
- `RolloutBuffer`: Stores trajectories for PPO updates
- `PPO`: Main agent class implementing:
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Mini-batch updates with multiple epochs
  - Gradient clipping for stability

**Training outputs:**
- `models/ppo_episode_<N>.pt` - Periodic checkpoints every save_interval episodes
- `models/ppo_solved.pt` - Saved when average reward > 200
- `models/ppo_final.pt` - Final model at end of training
- `models/training_curve.png` - Training progress visualization
- `runs/PPO_LunarLander_<timestamp>/` - TensorBoard event files for each training run

### Key Implementation Details

**PPO Algorithm:**
- Uses clipped surrogate objective to prevent large policy updates
- GAE (Generalized Advantage Estimation) for variance reduction
- Advantage normalization for stable training
- Entropy bonus (coefficient: 0.01) to encourage exploration
- Value loss coefficient: 0.5
- Gradient clipping at norm 0.5

**Action Exploration:**
- Actions sampled from Gaussian distribution: N(μ(s), σ)
- Mean μ(s) output by actor network with tanh activation (bounded to [-1, 1])
- Standard deviation σ is a learnable parameter (initialized to 0, i.e., σ=1)
- Log probabilities computed for PPO ratio calculation

**Training Process:**
1. Collect rollout_length steps of experience
2. Compute advantages using GAE
3. Update policy with multiple epochs of mini-batch SGD
4. Repeat until solved (avg reward > 200) or max episodes reached

**Performance Notes:**
- The environment is considered "solved" at average reward ≥ 200 over 100 episodes
- Initial episodes typically have large negative rewards (-300 to -400)
- Training usually converges within 500-1500 episodes with default hyperparameters
- CUDA support automatically enabled if GPU available
