"""
Gymnasium FlyArcadeEnv: Fruit Fly Arcade Flight Simulation
==========================================================
A closed-loop Gymnasium environment where a virtual Drosophila melanogaster
navigates dynamic obstacle tunnels, collecting sucrose droplets while
processing visual compound eye inputs and physical flight forces.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from biofly.physics.aerodynamics import FlyFlightDynamics, FlyMorphology


class Obstacle:
    """Represents a gate or pillar with a navigable gap."""

    def __init__(self, x: float, gap_y: float, gap_height: float, width: float = 0.4):
        self.x = x
        self.gap_y = gap_y          # Center height of the gap (in normalized units 0.1 to 0.9)
        self.gap_height = gap_height  # Height of the opening
        self.width = width
        self.passed = False
        self.has_food = np.random.rand() > 0.4
        self.food_collected = False

    def check_collision(self, fly_x: float, fly_y: float, fly_radius: float = 0.04) -> bool:
        """Check if fly circles collide with obstacle pillars."""
        in_x = (self.x - self.width / 2.0 - fly_radius) <= fly_x <= (self.x + self.width / 2.0 + fly_radius)
        if not in_x:
            return False

        # Safe gap boundaries
        gap_top = self.gap_y + self.gap_height / 2.0 - fly_radius
        gap_bottom = self.gap_y - self.gap_height / 2.0 + fly_radius

        # Collision if fly is above the gap or below the gap
        if fly_y > gap_top or fly_y < gap_bottom:
            return True
        return False

    def check_food(self, fly_x: float, fly_y: float, collect_radius: float = 0.08) -> bool:
        if self.has_food and not self.food_collected:
            dist = np.hypot(fly_x - self.x, fly_y - self.gap_y)
            if dist < collect_radius:
                self.food_collected = True
                return True
        return False


class FlyArcadeEnv(gym.Env):
    """
    Gymnasium Environment for Drosophila Arcade Flight.
    Modes:
      - 'obstacle_dodge': Pillars with randomized gaps and sucrose targets.
      - 'flappy': Fast-paced flap-to-fly gravity challenge.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        mode: str = "obstacle_dodge",
        render_mode: Optional[str] = None,
        eye_res: Tuple[int, int] = (16, 16),
        dict_obs: bool = True,
        max_steps: int = 1500,
    ):
        super().__init__()
        self.mode = mode
        self.render_mode = render_mode
        self.eye_res = eye_res
        self.dict_obs = dict_obs
        self.max_steps = max_steps
        self.dt = 0.02  # 50 Hz physics step

        # Physics engine
        self.physics = FlyFlightDynamics(dt=self.dt)

        # Arena boundaries (normalized coordinates: x goes forward, y in [0, 1])
        self.arena_height = 1.0  # normalized vertical bound
        self.fly_radius = 0.035

        # Action space: [0: flap_lift (-1 to 1), 1: pitch_torque (-1 to 1)]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation spaces
        vis_shape = (1, eye_res[0], eye_res[1])
        self.vis_space = spaces.Box(low=0.0, high=1.0, shape=vis_shape, dtype=np.float32)
        # Proprioception: [norm_y, vel_y, vel_x, pitch, dist_next_obs, gap_rel_y, looming_rate]
        self.proprio_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        if self.dict_obs:
            self.observation_space = spaces.Dict({
                "visual": self.vis_space,
                "proprioception": self.proprio_space,
            })
        else:
            # Flattened vector observation: proprio (7) + pooled visual (16) = 23
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
            )

        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.score = 0
        self.total_food = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.physics.reset(
            initial_pos=np.array([0.0, 0.0, 0.5]),
            initial_vel=np.array([1.0, 0.0, 0.0]),
        )
        self.step_count = 0
        self.score = 0
        self.total_food = 0
        self.last_action = np.zeros(2, dtype=np.float32)

        # Generate initial obstacles along corridor
        self.obstacles = []
        start_x = 2.0
        spacing = 2.5
        gap_h = 0.35 if self.mode == "obstacle_dodge" else 0.40
        for i in range(12):
            gap_y = np.random.uniform(0.25, 0.75)
            self.obstacles.append(Obstacle(x=start_x + i * spacing, gap_y=gap_y, gap_height=gap_h))

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_action = action

        # 1. Update 6-DoF flight dynamics
        # Map action[0] (lift) and action[1] (pitch) into physics
        phys_state = self.physics.step(action)

        fly_x = phys_state["pos"][0]
        fly_y = phys_state["pos"][2]  # Z axis is altitude in physics
        fly_pitch = phys_state["angles"][0]
        vel_x = phys_state["vel"][0]
        vel_y = phys_state["vel"][2]

        # 2. Recycle / generate obstacles as fly advances
        if self.obstacles and (fly_x - self.obstacles[0].x > 3.0):
            old = self.obstacles.pop(0)
            last_x = self.obstacles[-1].x
            gap_y = np.random.uniform(0.25, 0.75)
            gap_h = 0.35
            self.obstacles.append(Obstacle(x=last_x + 2.5, gap_y=gap_y, gap_height=gap_h))

        # 3. Check collisions and events
        terminated = False
        reward = 1.0  # Base survival reward per step

        # Ceiling and ground boundaries
        if fly_y <= self.fly_radius or fly_y >= (self.arena_height - self.fly_radius):
            terminated = True
            reward = -10.0

        # Obstacle collisions & rewards
        next_obs = None
        for obs in self.obstacles:
            if not obs.passed:
                next_obs = obs
                break

        if next_obs is not None:
            # Check collision
            if next_obs.check_collision(fly_x, fly_y, self.fly_radius):
                terminated = True
                reward = -10.0

            # Check passing obstacle
            if not next_obs.passed and fly_x > (next_obs.x + next_obs.width / 2.0):
                next_obs.passed = True
                self.score += 1
                reward += 10.0  # Big reward for navigating through gap!

            # Check sucrose food collection
            if next_obs.check_food(fly_x, fly_y):
                self.total_food += 1
                reward += 15.0

            # Alignment shaping reward (encourages centering on gap)
            gap_dist = abs(fly_y - next_obs.gap_y)
            reward += max(0.0, 0.3 - gap_dist)

        # Forward velocity encouragement
        reward += 0.2 * max(0.0, vel_x)

        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def _synthesize_compound_eye_view(self) -> np.ndarray:
        """
        Synthesizes a retinotopic visual matrix (eye_res x eye_res)
        representing what the fruit fly ommatidia see looking forward.
        Bright = open sky/gap, Dark = approaching obstacle pillars.
        """
        h, w = self.eye_res
        view = np.ones((h, w), dtype=np.float32) * 0.85  # Default ambient luminance

        fly_x = self.physics.pos[0]
        fly_y = self.physics.pos[2]

        # Find closest upcoming obstacle within visual range (~3.0 meters)
        for obs in self.obstacles:
            dx = obs.x - fly_x
            if 0.05 < dx < 3.5:
                # Apparent vertical gap position and angular size
                # Perspective projection: angular size ~ width / dx
                dist_factor = np.clip(1.0 - (dx / 3.5), 0.0, 1.0)
                # Apparent gap top and bottom in visual coordinate [-1, 1]
                gap_top = obs.gap_y + obs.gap_height / 2.0
                gap_bot = obs.gap_y - obs.gap_height / 2.0

                # Relative to fly altitude
                norm_top = int(np.clip(((gap_top - fly_y) * 2.0 + 0.5) * h, 0, h))
                norm_bot = int(np.clip(((gap_bot - fly_y) * 2.0 + 0.5) * h, 0, h))

                # Darken top and bottom obstacle pillars
                pillar_darkness = 0.15 + (1.0 - dist_factor) * 0.4
                if norm_top < h:
                    view[norm_top:, :] = pillar_darkness
                if norm_bot > 0:
                    view[:norm_bot, :] = pillar_darkness

                # Bright center for food droplet if present
                if obs.has_food and not obs.food_collected:
                    food_y = int(np.clip(((obs.gap_y - fly_y) * 2.0 + 0.5) * h, 0, h - 1))
                    food_w = max(1, int(3 * dist_factor))
                    y_start = max(0, food_y - food_w)
                    y_end = min(h, food_y + food_w + 1)
                    view[y_start:y_end, w // 2 - food_w : w // 2 + food_w + 1] = 1.0

                break  # Only closest obstacle dominates forward visual field

        # Add ommatidial sensory noise
        noise = np.random.normal(0.0, 0.02, size=view.shape).astype(np.float32)
        view = np.clip(view + noise, 0.0, 1.0)
        return view

    def _get_observation(self) -> Any:
        vis_2d = self._synthesize_compound_eye_view()
        vis_tensor = vis_2d[np.newaxis, :, :]  # (1, H, W)

        fly_x = self.physics.pos[0]
        fly_y = self.physics.pos[2]
        vel_x = self.physics.vel[0]
        vel_y = self.physics.vel[2]
        pitch = self.physics.angles[0]

        # Obstacle relative metrics
        next_obs = None
        for obs in self.obstacles:
            if not obs.passed:
                next_obs = obs
                break

        if next_obs is not None:
            dist_x = max(0.0, next_obs.x - fly_x)
            gap_rel_y = next_obs.gap_y - fly_y
            # Looming expansion rate ~ vel_x / (dist_x^2 + eps)
            looming = vel_x / (dist_x + 0.2) ** 2
        else:
            dist_x = 5.0
            gap_rel_y = 0.0
            looming = 0.0

        proprio = np.array([
            float(fly_y),
            float(vel_y),
            float(vel_x),
            float(pitch),
            float(dist_x),
            float(gap_rel_y),
            float(looming),
        ], dtype=np.float32)

        if self.dict_obs:
            return {"visual": vis_tensor, "proprioception": proprio}
        else:
            # Downsampled visual 4x4 (16 values) + proprio (7) = 23
            pooled_vis = vis_2d[::4, ::4].flatten()
            return np.concatenate([proprio, pooled_vis])

    def _get_info(self) -> Dict[str, Any]:
        phys = self.physics.get_state()
        return {
            "score": self.score,
            "food": self.total_food,
            "step": self.step_count,
            "altitude": float(self.physics.pos[2]),
            "airspeed": phys["airspeed"],
            "wing_freq": phys["wing_freq"],
            "amp_left": phys["amp_left"],
            "amp_right": phys["amp_right"],
            "lift_force": phys["lift_force"],
            "pitch_angle": float(self.physics.angles[0]),
            "obstacles": self.obstacles,
            "pos": self.physics.pos,
        }

    def render(self):
        if self.render_mode is None:
            return None
        from biofly.envs.rendering import ArcadeRenderer
        if self.renderer is None:
            self.renderer = ArcadeRenderer(env=self, mode=self.render_mode)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
