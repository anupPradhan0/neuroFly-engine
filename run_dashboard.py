#!/usr/bin/env python3
"""
BioFly-Arcade Engine: Live Dashboard Launcher
=============================================
Launches the dual-panel interactive window showing live arcade flight
and real-time Drosophila connectome neural telemetry.

Usage:
    python run_dashboard.py
    python run_dashboard.py --model checkpoints/biofly_ppo_final.zip
"""

import argparse
from biofly.ui.dashboard import BioFlyDashboard
from biofly.envs.arcade_env import FlyArcadeEnv


def main():
    parser = argparse.ArgumentParser(description="BioFly Live Execution & Biological Telemetry Dashboard")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PyTorch or SB3 checkpoint")
    parser.add_argument("--mode", type=str, default="obstacle_dodge", choices=["obstacle_dodge", "flappy"])
    parser.add_argument("--width", type=int, default=1200, help="Window width")
    parser.add_argument("--height", type=int, default=700, help="Window height")
    args = parser.parse_args()

    env = FlyArcadeEnv(mode=args.mode, dict_obs=True, render_mode="rgb_array")
    dash = BioFlyDashboard(env=env, model_path=args.model, width=args.width, height=args.height)
    print("\n[+] Starting BioFly-Arcade Live Dashboard...")
    print("Controls:")
    print("  'M'     - Cycle control mode (Trained RL / Biological Reflex / Manual Play)")
    print("  'SPACE' - Pause / Resume simulation")
    print("  'R'     - Reset flight episode")
    print("  'ESC'   - Exit dashboard")
    dash.run()


if __name__ == "__main__":
    main()
