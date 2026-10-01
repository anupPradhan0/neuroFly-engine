"""
FlyBrainPolicy: Connectome-Constrained RL Actor-Critic Policy
============================================================
Integrates the Drosophila visual connectome with proprioceptive flight state
for Stable-Baselines3 (PPO/SAC) and standalone PyTorch RL training.
"""

from typing import Dict, Tuple
import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from biofly.brain.connectome_vision import DrosophilaVisionConnectome


class BioFlyFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 Custom Feature Extractor using Connectome-Constrained Drosophila Vision.
    Processes compound eye visual fields alongside proprioceptive state vectors.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        # Determine total features dim
        super().__init__(observation_space, features_dim=features_dim)

        self.is_dict = isinstance(observation_space, spaces.Dict)
        if self.is_dict:
            visual_shape = observation_space.spaces["visual"].shape
            proprio_dim = observation_space.spaces["proprioception"].shape[0]
            # Height, Width from visual shape (C, H, W) or (H, W)
            if len(visual_shape) == 3:
                eye_res = (visual_shape[1], visual_shape[2])
            else:
                eye_res = (visual_shape[0], visual_shape[1])
        else:
            # Single flattened or vector observation
            proprio_dim = observation_space.shape[0]
            eye_res = (16, 16)

        self.connectome = DrosophilaVisionConnectome(eye_resolution=eye_res)
        self.proprio_dim = proprio_dim

        # Fusion layer: visual connectome embedding (32) + proprioception (7) -> features_dim (64)
        in_dim = self.connectome.feature_dim + proprio_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_dict and isinstance(observations, dict):
            visual = observations["visual"]
            proprio = observations["proprioception"]
            vis_embedding, _ = self.connectome(visual)
            combined = torch.cat([vis_embedding, proprio], dim=1)
        elif isinstance(observations, dict) and "proprioception" in observations:
            proprio = observations["proprioception"]
            vis_embedding = torch.zeros((proprio.size(0), self.connectome.feature_dim), device=proprio.device)
            combined = torch.cat([vis_embedding, proprio], dim=1)
        else:
            # Vector observation input
            if observations.dim() == 1:
                observations = observations.unsqueeze(0)
            vis_dummy = torch.zeros((observations.size(0), self.connectome.feature_dim), device=observations.device)
            combined = torch.cat([vis_dummy, observations], dim=1)

        return self.fusion(combined)


class ConnectomeFlyPolicy(ActorCriticPolicy):
    """
    Connectome-constrained Actor-Critic policy for Stable-Baselines3 PPO.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=BioFlyFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[32, 32], vf=[32, 32]),
            *args,
            **kwargs,
        )


class StandaloneBioFlyNet(nn.Module):
    """
    Direct PyTorch neural controller for autonomous gaming without SB3 dependencies.
    Includes bio-reflex optomotor and looming circuits.
    """

    def __init__(self, visual_shape=(1, 16, 16), proprio_dim: int = 7, action_dim: int = 2):
        super().__init__()
        self.connectome = DrosophilaVisionConnectome(eye_resolution=(visual_shape[1], visual_shape[2]))
        in_dim = self.connectome.feature_dim + proprio_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Actor head: mean action
        self.actor = nn.Linear(32, action_dim)
        # Critic head: state value
        self.critic = nn.Linear(32, 1)

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        vis_emb, acts = self.connectome(visual)
        combined = torch.cat([vis_emb, proprio], dim=1)
        features = self.trunk(combined)

        # Action: tanh bounded in [-1.0, 1.0]
        action_mean = torch.tanh(self.actor(features))
        value = self.critic(features)

        # Biological looming avoidance bias:
        # If looming detector (LPLC2) is high, trigger pitch climb reflex
        looming_signal = acts.get("lplc2_looming", 0.0)
        if isinstance(looming_signal, (int, float)) and looming_signal > 0.3:
            # Reflexive pitch up and flap boost
            action_mean[:, 0] = torch.clamp(action_mean[:, 0] + 0.3, -1.0, 1.0)

        return action_mean, value, acts
