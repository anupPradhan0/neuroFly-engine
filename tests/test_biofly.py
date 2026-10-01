"""
Unit and Integration Tests for BioFly-Arcade Engine
==================================================
Tests aerodynamics, connectome neural pathways, MuJoCo biomechanics,
Gymnasium environment lifecycle, and SB3 policy compatibility.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym

from biofly.physics.aerodynamics import FlyFlightDynamics, FlyMorphology
from biofly.physics.fly_mujoco import MuJoCoFly, MUJOCO_AVAILABLE
from biofly.brain.connectome_vision import (
    OmmatidiaRetina,
    LaminaONOFF,
    MedullaDelayLine,
    T4T5ElementaryMotionDetector,
    LPTCTangentialCells,
    DrosophilaVisionConnectome,
)
from biofly.brain.policy import StandaloneBioFlyNet, BioFlyFeaturesExtractor, ConnectomeFlyPolicy
from biofly.envs.arcade_env import FlyArcadeEnv, Obstacle


class TestPhysics:
    def test_aerodynamics_reset(self):
        dynamics = FlyFlightDynamics()
        state = dynamics.reset(initial_pos=np.array([0.0, 0.0, 0.5]))
        assert np.allclose(state["pos"], [0.0, 0.0, 0.5])
        assert state["wing_freq"] > 150.0
        assert state["airspeed"] > 0.0

    def test_aerodynamics_step_climb_and_dive(self):
        dynamics = FlyFlightDynamics(dt=0.02)
        dynamics.reset(initial_pos=np.array([0.0, 0.0, 0.5]))

        # Flap upward at max lift
        state_climb = dynamics.step(action=np.array([1.0, 0.2]))
        assert state_climb["pos"][2] >= 0.5  # Altitude should increase or stay stable

        # Glide / zero flap dive
        dynamics.reset(initial_pos=np.array([0.0, 0.0, 0.5]), initial_vel=np.array([0.0, 0.0, 0.0]))
        for _ in range(5):
            state_dive = dynamics.step(action=np.array([-1.0, -0.5]))
        assert state_dive["pos"][2] < 0.5  # Altitude dropped under gravity

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_mujoco_fly_model(self):
        fly = MuJoCoFly(headless=True)
        init_state = fly.reset(initial_height=1.0)
        assert len(init_state["position"]) == 3
        # Apply flap control
        next_state = fly.step(np.array([0.5, 0.1, 0.5, 0.1]))
        assert len(next_state["position"]) == 3


class TestConnectomeBrain:
    def test_ommatidia_retina(self):
        retina = OmmatidiaRetina()
        dummy_rgb = torch.rand(2, 3, 16, 16)
        out = retina(dummy_rgb)
        assert out.shape == (2, 1, 16, 16)
        assert (out >= 0.0).all()

    def test_lamina_on_off(self):
        lamina = LaminaONOFF()
        frame1 = torch.ones(1, 1, 16, 16) * 0.2
        frame2 = torch.ones(1, 1, 16, 16) * 0.8  # Brightening (ON)
        frame3 = torch.ones(1, 1, 16, 16) * 0.3  # Dimming (OFF)

        lamina(frame1)
        l1_on, l2_off = lamina(frame2)
        assert (l1_on > 0.0).any()
        assert (l2_off == 0.0).all()

        l1_on, l2_off = lamina(frame3)
        assert (l1_on == 0.0).all()
        assert (l2_off > 0.0).any()

    def test_drosophila_vision_pipeline(self):
        connectome = DrosophilaVisionConnectome(eye_resolution=(16, 16))
        sample_input = torch.rand(2, 1, 16, 16)
        emb, acts = connectome(sample_input)
        assert emb.shape == (2, connectome.feature_dim)
        assert "hs_cell" in acts
        assert "vs_cell" in acts
        assert "lplc2_looming" in acts

    def test_standalone_biofly_net(self):
        net = StandaloneBioFlyNet(visual_shape=(1, 16, 16), proprio_dim=7, action_dim=2)
        vis = torch.rand(1, 1, 16, 16)
        proprio = torch.rand(1, 7)
        action, value, acts = net(vis, proprio)
        assert action.shape == (1, 2)
        assert value.shape == (1, 1)
        assert -1.0 <= action[0, 0].item() <= 1.0


class TestGymnasiumEnvironment:
    def test_env_lifecycle_dict_obs(self):
        env = FlyArcadeEnv(mode="obstacle_dodge", dict_obs=True)
        obs, info = env.reset(seed=42)
        assert "visual" in obs
        assert "proprioception" in obs
        assert obs["visual"].shape == (1, 16, 16)
        assert obs["proprioception"].shape == (7,)

        action = np.array([0.5, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "score" in info
        env.close()

    def test_env_lifecycle_flat_obs(self):
        env = FlyArcadeEnv(mode="obstacle_dodge", dict_obs=False)
        obs, info = env.reset(seed=123)
        assert obs.shape == (23,)
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert next_obs.shape == (23,)
        env.close()

    def test_obstacle_collision_logic(self):
        obs = Obstacle(x=2.0, gap_y=0.5, gap_height=0.3, width=0.2)
        # Inside the open gap
        assert not obs.check_collision(fly_x=2.0, fly_y=0.5)
        # Hitting top pillar
        assert obs.check_collision(fly_x=2.0, fly_y=0.85)
        # Hitting bottom pillar
        assert obs.check_collision(fly_x=2.0, fly_y=0.15)
        # Far before obstacle
        assert not obs.check_collision(fly_x=1.0, fly_y=0.85)


class TestSB3PolicyIntegration:
    def test_sb3_feature_extractor(self):
        env = FlyArcadeEnv(dict_obs=True)
        extractor = BioFlyFeaturesExtractor(env.observation_space, features_dim=64)
        obs, _ = env.reset()
        obs_t = {
            "visual": torch.tensor(obs["visual"], dtype=torch.float32).unsqueeze(0),
            "proprioception": torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0),
        }
        feats = extractor(obs_t)
        assert feats.shape == (1, 64)
        env.close()
