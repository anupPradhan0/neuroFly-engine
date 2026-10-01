"""
BioFly Evaluation and Benchmark Tool
====================================
Evaluates a trained or reflex fly policy over N episodes and reports
statistical performance metrics: clearance rate, food collected, and flight stability.
"""

import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from biofly.envs.arcade_env import FlyArcadeEnv
from biofly.brain.policy import StandaloneBioFlyNet


def evaluate_policy(
    model_path: str = None,
    num_episodes: int = 10,
    render: bool = False,
    mode: str = "obstacle_dodge",
):
    print("=" * 65)
    print("             BIOFLY BENCHMARK EVALUATION")
    print("=" * 65)
    print(f"Model Path:     {model_path or 'Biological Reflex Controller'}")
    print(f"Episodes:       {num_episodes}")
    print(f"Game Mode:      {mode}")

    render_mode = "human" if render else None
    env = FlyArcadeEnv(mode=mode, render_mode=render_mode, dict_obs=True)

    sb3_model = None
    if model_path and model_path.endswith(".zip"):
        print("[+] Loading Stable-Baselines3 PPO model...")
        sb3_model = PPO.load(model_path, env=env)
    elif model_path and model_path.endswith(".pt"):
        print("[+] Loading Standalone PyTorch weights...")
        standalone_net = StandaloneBioFlyNet()
        standalone_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        standalone_net.eval()
    else:
        standalone_net = StandaloneBioFlyNet()
        standalone_net.eval()

    episode_rewards = []
    episode_scores = []
    episode_foods = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep * 42)
        done = False
        total_rew = 0.0
        steps = 0

        while not done:
            if sb3_model is not None:
                action, _ = sb3_model.predict(obs, deterministic=True)
            else:
                vis = torch.tensor(obs["visual"], dtype=torch.float32).unsqueeze(0)
                proprio = torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act_t, _, acts = standalone_net(vis, proprio)
                action = act_t.squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            total_rew += reward
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(total_rew)
        episode_scores.append(info["score"])
        episode_foods.append(info["food"])
        episode_lengths.append(steps)

        print(f"Episode {ep+1:2d} | Steps: {steps:4d} | Obstacles Passed: {info['score']:2d} | Sucrose: {info['food']:2d} | Reward: {total_rew:6.1f}")

    env.close()

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Mean Survival Steps:     {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Obstacles Cleared:  {np.mean(episode_scores):.1f} ± {np.std(episode_scores):.1f}")
    print(f"Mean Sucrose Collected:  {np.mean(episode_foods):.1f} ± {np.std(episode_foods):.1f}")
    print(f"Mean Episode Reward:     {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BioFly Policy")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.zip or .pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of benchmark episodes")
    parser.add_argument("--render", action="store_true", help="Render gameplay to screen")
    parser.add_argument("--mode", type=str, default="obstacle_dodge")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render,
        mode=args.mode,
    )
