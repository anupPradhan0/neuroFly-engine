# BioFly-Arcade Engine 🪰🕹️

> **Closed-Loop Reinforcement Learning Simulation with Connectome-Constrained Drosophila Melanogaster Gaming Agent**

---

## 🌟 Overview

**BioFly-Arcade Engine** connects deep biological connectomics with reinforcement learning and 6-DoF biomechanical flight simulation. An autonomous virtual fruit fly (*Drosophila melanogaster*) observes an arcade game interface through simulated compound eye ommatidia, processes optical flow and looming cues across the visual connectome (Lamina, Medulla, Lobula Plate), and outputs real-time physical flight forces to navigate complex obstacles.

```
   ┌────────────────────────────────────────────────────────┐
   │                  1. GAME ENVIRONMENT                   │
   │  • Gymnasium FlyArcadeEnv (Obstacle Dodge / Flappy)    │
   │  • Retinotopic compound eye rendering & food rewards   │
   └───────────────────────────┬────────────────────────────┘
                               │ Retinal Ommatidia View / State
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                  2. FLY BRAIN POLICY                   │
   │  • Drosophila Optic Lobe Connectome (PyTorch / SB3)    │
   │  • Ommatidia -> Lamina L1/L2 -> Medulla delay lines    │
   │  • T4/T5 directional motion detectors (EMD)            │
   │  • LPTC (HS yaw, VS pitch) + LPLC2 Looming alert       │
   └───────────────────────────┬────────────────────────────┘
                               │ Action Vector (Flap Lift / Pitch Torques)
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                 3. PHYSICS & EXECUTION                 │
   │  • 6-DoF Aerodynamic Equations of Motion               │
   │  • Bilateral wing flapping & aerodynamic lift/drag     │
   │  • Headless MuJoCo 3D fly biomechanics model           │
   └────────────────────────────────────────────────────────┘
```

---

## 🔬 Biological Architecture

The fly visual system models the canonical connectome architecture:
1. **Compound Eye (Retina)**: Hexagonal array of ommatidia photoreceptors (R1-R6) with logarithmic luminance adaptation ($R = \log(1 + I / I_0)$).
2. **Lamina (L1 & L2)**: Temporal contrast derivative splitting into **L1 (ON)** and **L2 (OFF)** channels.
3. **Medulla Delay Lines**: Mi1, Tm3, Tm1, and Tm2 interneurons with temporal integration filters.
4. **Lobula Plate (T4 & T5 EMD)**:
   - **T4 cells**: Compute directional motion for ON stimuli along 4 cardinal directions ($\rightarrow, \leftarrow, \uparrow, \downarrow$).
   - **T5 cells**: Compute directional motion for OFF stimuli along 4 cardinal directions.
5. **Lobula Plate Tangential Cells (LPTC)**:
   - **HS (Horizontal System)**: Wide-field integration of horizontal optic flow (yaw rotatory stabilization).
   - **VS (Vertical System)**: Wide-field integration of vertical optic flow (pitch/roll stabilization).
   - **LPLC2 (Looming-Sensitive Cells)**: Detects expanding dark borders of approaching obstacles, triggering reflexive escape saccades.

---

## 🚀 Quickstart

### 1. Installation
All code is located in `/home/mors/Code/gemini-project`:
```bash
pip install -e . --no-deps
```

### 2. Live Execution & Biological Telemetry Dashboard
Launch the real-time dual-viewport dashboard:
```bash
python run_dashboard.py
```
- **Left Panel**: Real-time arcade flight simulation with animated flapping wings, particle wake, obstacle tunnels, and sucrose food droplets.
- **Right Panel**: Live biological connectome meters showing:
  - 16x16 Retinotopic Ommatidia heatmap
  - T4 & T5 direction-selective optical flow meters
  - HS yaw & VS pitch tangential cell activations
  - LPLC2 Looming Collision Alert meter
  - Wing flap frequency and aerodynamic lift force

**Interactive Controls**:
- Press **`M`**: Cycle between **RL Agent**, **Biological Reflex (Optomotor + Looming)**, and **Manual Keyboard Play** (Arrow keys / W/S).
- Press **`SPACE`**: Pause / Resume flight.
- Press **`R`**: Reset episode.
- Press **`ESC`**: Exit dashboard.

### 3. RL Training (PPO)
Train the connectome policy using Stable-Baselines3 PPO:
```bash
python train.py --timesteps 50000 --n-envs 4 --device cuda
```
- Specially tuned for **4GB VRAM** GPUs (RTX 3050 Mobile) with low batch sizes and headless execution (`MUJOCO_GL=egl`).

### 4. Benchmark Evaluation
```bash
python evaluate.py --model checkpoints/biofly_ppo_final.zip --episodes 10
```

### 5. Running Test Suite
```bash
pytest tests/test_biofly.py -v
```

---

## 📁 Repository Structure

```
.
├── biofly/
│   ├── physics/
│   │   ├── aerodynamics.py       # 6-DoF Drosophila flight dynamics & wing kinematics
│   │   └── fly_mujoco.py         # MuJoCo 3D fly model generator & simulation wrapper
│   ├── brain/
│   │   ├── connectome_vision.py  # Ommatidia, Lamina L1/L2, T4/T5, LPTC (HS/VS), LPLC2
│   │   └── policy.py             # SB3 ConnectomeFlyPolicy & StandaloneBioFlyNet
│   ├── envs/
│   │   ├── arcade_env.py         # Gymnasium FlyArcadeEnv with obstacles & rewards
│   │   └── rendering.py          # Pygame arcade renderer with wing flapping & HUD
│   └── ui/
│       └── dashboard.py          # Live dual-viewport dashboard with neural meters
├── train.py                      # PPO training pipeline
├── evaluate.py                   # Benchmark and evaluation script
├── run_dashboard.py              # Live interactive dashboard launcher
├── tests/
│   └── test_biofly.py            # Comprehensive pytest test suite
├── requirements.txt
├── setup.py
└── README.md
```
