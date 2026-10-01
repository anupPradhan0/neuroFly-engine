"""
BioFly RL Training Pipeline
===========================
Trains the Drosophila connectome neural network using Proximal Policy Optimization (PPO).
Calibrated for 4GB VRAM GPU (RTX 3050) with low batch size and 4-8 parallel envs.
"""

import os
import argparse
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from biofly.envs.arcade_env import FlyArcadeEnv
from biofly.brain.policy import ConnectomeFlyPolicy


class FlyTrainingCallback(BaseCallback):
    """Logs rewards, obstacle pass rates, and survival time during PPO training."""

    def __init__(self, log_freq: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_scores = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Check infos for episode terminations
        for info in self.locals.get("infos", []):
            if "score" in info:
                self.episode_scores.append(info["score"])

        if self.n_calls % self.log_freq == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.n_calls / max(1.0, elapsed))
            mean_score = np.mean(self.episode_scores[-20:]) if self.episode_scores else 0.0
            print(
                f"[Step {self.n_calls:7d} | {elapsed:5.1f}s | {fps:4d} FPS] "
                f"Obstacles Cleared (Mean Last 20): {mean_score:4.1f}"
            )
        return True


def make_env(rank: int, seed: int = 0, mode: str = "obstacle_dodge"):
    """Creates a seeded FlyArcadeEnv instance."""
    def _init():
        env = FlyArcadeEnv(
            mode=mode,
            dict_obs=True,
            render_mode=None,
            max_steps=1200,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_biofly(
    total_timesteps: int = 50000,
    n_envs: int = 4,
    device: str = "auto",
    save_dir: str = "checkpoints",
    env_mode: str = "obstacle_dodge",
):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("      BIOFLY-ARCADE ENGINE: CONNECTOME RL TRAINING (PPO)")
    print("=" * 70)
    print(f"Target Timesteps:    {total_timesteps:,}")
    print(f"Parallel Envs:       {n_envs}")
    print(f"Environment Mode:    {env_mode}")
    print(f"Device:              {device}")
    print(f"Checkpoint Dir:      {save_dir}")

    # Set headless MuJoCo flag if needed
    os.environ["MUJOCO_GL"] = "egl"

    # Create vectorized environments
    env_fns = [make_env(i, mode=env_mode) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Initialize PPO with Connectome Policy
    # Optimized for 4GB VRAM: n_steps=512, batch_size=64 (small minibatch)
    model = PPO(
        policy=ConnectomeFlyPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
    )

    callback = FlyTrainingCallback(log_freq=2000)

    print("\n[+] Starting PPO optimization over Drosophila visual connectome...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - t0

    # Save final model
    final_path = os.path.join(save_dir, "biofly_ppo_final.zip")
    model.save(final_path)
    print(f"\n[✓] Training complete in {training_time:.1f}s. Model saved to: {final_path}")

    vec_env.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioFly RL Connectome Agent")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total environment steps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments (4-8 recommended)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--mode", type=str, default="obstacle_dodge", help="Game mode: 'obstacle_dodge' or 'flappy'")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    train_biofly(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        device=args.device,
        save_dir=args.save_dir,
        env_mode=args.mode,
    )
