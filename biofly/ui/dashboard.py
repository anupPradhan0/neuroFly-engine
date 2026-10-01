"""
BioFly Live Execution & Biological Neural Telemetry Dashboard
=============================================================
A real-time interactive Pygame dashboard displaying:
1. Arcade Game Viewport (Fruit fly flight, flapping wings, obstacle tunnels)
2. Live Drosophila Brain Connectome Firing Rates:
   - Compound Eye Retinotopic Heatmap (Ommatidia)
   - Lamina L1 (ON) / L2 (OFF) contrast channels
   - T4 & T5 Direction-Selective EMD meters (Up, Down, Left, Right)
   - Lobula Plate Tangential Cells (HS yaw, VS pitch, LPLC2 looming collision)
   - Motor output meters (Wingbeat frequency, Bilateral stroke amplitudes, Lift)
"""

from typing import Optional, Dict, Any
import numpy as np
import pygame
import torch

from biofly.envs.arcade_env import FlyArcadeEnv
from biofly.brain.connectome_vision import DrosophilaVisionConnectome
from biofly.brain.policy import StandaloneBioFlyNet


class BioFlyDashboard:
    """Real-time rendering loop and biological connectome neural monitor."""

    def __init__(
        self,
        env: Optional[FlyArcadeEnv] = None,
        model_path: Optional[str] = None,
        width: int = 1200,
        height: int = 700,
    ):
        self.width = width
        self.height = height
        self.game_w = 680
        self.dash_w = width - self.game_w

        pygame.init()
        pygame.display.set_caption("BioFly-Arcade Engine | Connectome Neural Telemetry")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_sub = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_text = pygame.font.SysFont("monospace", 12)
        self.font_small = pygame.font.SysFont("monospace", 10)

        # Environment
        self.env = env or FlyArcadeEnv(mode="obstacle_dodge", dict_obs=True, render_mode="rgb_array")
        self.obs, self.info = self.env.reset()

        # Connectome model
        self.brain = StandaloneBioFlyNet(visual_shape=(1, 16, 16), proprio_dim=7, action_dim=2)
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                self.brain.load_state_dict(state_dict)
                print(f"[BioFlyDashboard] Loaded model weights from {model_path}")
            except Exception as e:
                print(f"[BioFlyDashboard] Warning: Could not load weights ({e}). Using initialized connectome.")
        self.brain.eval()

        # Agent control mode: 'rl', 'reflex', 'manual'
        # Default to 'reflex' (Biological Heuristic) if no trained model is loaded,
        # otherwise the untrained RL network will just crash randomly.
        self.control_mode = "rl" if model_path else "reflex"
        self.paused = False
        self.episodes = 0
        self.cumulative_reward = 0.0

        # Cached neural activations
        self.activations: Dict[str, Any] = {}
        self.looming_history = []
        self.altitude_history = []

    def run(self):
        """Main execution and rendering loop."""
        running = True
        while running:
            # Handle user events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.obs, self.info = self.env.reset()
                        self.brain.connectome.reset_states()
                        self.cumulative_reward = 0.0
                    elif event.key == pygame.K_m:
                        # Cycle control modes
                        modes = ["rl", "reflex", "manual"]
                        idx = (modes.index(self.control_mode) + 1) % len(modes)
                        self.control_mode = modes[idx]

            # Step physics and policy if not paused
            if not self.paused:
                action = self._compute_action()
                next_obs, reward, terminated, truncated, self.info = self.env.step(action)
                self.cumulative_reward += reward
                self.obs = next_obs

                # Track histories
                self.altitude_history.append(float(self.env.physics.pos[2]))
                if len(self.altitude_history) > 100:
                    self.altitude_history.pop(0)

                looming_val = self.activations.get("lplc2_looming", 0.0)
                self.looming_history.append(float(looming_val))
                if len(self.looming_history) > 100:
                    self.looming_history.pop(0)

                if terminated or truncated:
                    self.episodes += 1
                    self.obs, self.info = self.env.reset()
                    self.brain.connectome.reset_states()
                    self.cumulative_reward = 0.0

            # Render frame
            self._render_dashboard()
            pygame.display.flip()
            self.clock.tick(50)

        pygame.quit()

    def _compute_action(self) -> np.ndarray:
        """Determines action based on control mode and computes connectome activations."""
        vis_np = self.obs["visual"]  # (1, 16, 16)
        proprio_np = self.obs["proprioception"]  # (7,)

        vis_tensor = torch.tensor(vis_np, dtype=torch.float32).unsqueeze(0)  # (1, 1, 16, 16)
        proprio_tensor = torch.tensor(proprio_np, dtype=torch.float32).unsqueeze(0)  # (1, 7)

        with torch.no_grad():
            action_mean, value, self.activations = self.brain(vis_tensor, proprio_tensor)

        if self.control_mode == "manual":
            # Keyboard override
            keys = pygame.key.get_pressed()
            lift_act = 0.0
            pitch_act = 0.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                lift_act += 0.8
                pitch_act += 0.5
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                lift_act -= 0.6
                pitch_act -= 0.5
            return np.array([lift_act, pitch_act], dtype=np.float32)

        elif self.control_mode == "reflex":
            # Drosophila biological heuristic:
            # Optomotor reflex: stabilizes altitude and steers away from looming obstacles
            looming = float(self.activations.get("lplc2_looming", 0.0))
            gap_rel_y = float(proprio_np[5])
            fly_y = float(proprio_np[0])

            # Proportional reflex towards gap center
            flap_cmd = np.clip(gap_rel_y * 3.0 - (proprio_np[1] * 0.4), -1.0, 1.0)
            pitch_cmd = np.clip(gap_rel_y * 1.5, -0.8, 0.8)

            # Looming escape jump
            if looming > 0.35:
                # Saccade strongly towards the safe gap direction
                direction = 1.0 if gap_rel_y > 0 else -1.0
                flap_cmd = np.clip(flap_cmd + direction * 0.7, -1.0, 1.0)
                pitch_cmd = np.clip(pitch_cmd + direction * 0.4, -1.0, 1.0)

            return np.array([flap_cmd, pitch_cmd], dtype=np.float32)

        else:
            # Trained neural policy
            return action_mean.squeeze(0).numpy()

    def _render_dashboard(self):
        """Draw dual layout: Game viewport on left, Neural HUD on right."""
        self.screen.fill((15, 17, 23))

        # 1. Left Viewport: Game Simulation
        game_surf = pygame.Surface((self.game_w, self.height))
        # Use env renderer to draw to this surface
        if self.env.renderer is None:
            from biofly.envs.rendering import ArcadeRenderer
            self.env.renderer = ArcadeRenderer(self.env, width=self.game_w, height=self.height, mode="rgb_array")
        self.env.renderer.render()
        # Blit env renderer surface
        game_surf.blit(self.env.renderer.screen, (0, 0))

        # Mode banner on top-left
        mode_text = f"MODE: [{self.control_mode.upper()}] (Press 'M' to switch, 'SPACE' to pause, 'R' to reset)"
        t_surf = self.font_small.render(mode_text, True, (244, 208, 63))
        game_surf.blit(t_surf, (20, self.height - 25))

        self.screen.blit(game_surf, (0, 0))

        # Divider line
        pygame.draw.line(self.screen, (40, 48, 64), (self.game_w, 0), (self.game_w, self.height), 2)

        # 2. Right Viewport: Connectome Neural Telemetry
        dash_x = self.game_w + 20
        y_cursor = 15

        # Header
        head_text = self.font_title.render("DROSOPHILA CONNECTOME", True, (56, 189, 248))
        self.screen.blit(head_text, (dash_x, y_cursor))
        y_cursor += 24
        sub_text = self.font_text.render("Real-Time Optic Lobe Neural Activations", True, (148, 163, 184))
        self.screen.blit(sub_text, (dash_x, y_cursor))
        y_cursor += 30

        # Section 1: Retinotopic Ommatidia (Compound Eye View)
        y_cursor = self._draw_ommatidia_section(dash_x, y_cursor)

        # Section 2: Lamina L1/L2 and T4/T5 Motion Detectors
        y_cursor = self._draw_motion_detectors(dash_x, y_cursor)

        # Section 3: LPTC Tangential Cells & Looming Detector
        y_cursor = self._draw_lptc_section(dash_x, y_cursor)

        # Section 4: Flight Telemetry & Motor Signals
        self._draw_flight_meters(dash_x, y_cursor)

    def _draw_ommatidia_section(self, x: int, y: int) -> int:
        lbl = self.font_sub.render("1. Ommatidia Retina (R1-R6)", True, (226, 232, 240))
        self.screen.blit(lbl, (x, y))
        y += 20

        # Draw 16x16 eye matrix
        vis = self.obs["visual"][0]  # (16, 16)
        cell_size = 7
        grid_w = vis.shape[1] * cell_size
        grid_h = vis.shape[0] * cell_size

        for r in range(vis.shape[0]):
            for c in range(vis.shape[1]):
                val = float(vis[r, c])
                # Amber-red Drosophila eye color gradient
                color = (int(255 * val), int(140 * val), int(50 * val))
                rect = pygame.Rect(x + c * cell_size, y + r * cell_size, cell_size - 1, cell_size - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw Ommatidia annotations on side
        anno_x = x + grid_w + 15
        lines = [
            f"Resolution: {vis.shape[1]}x{vis.shape[0]} ommatidia",
            f"Mean Lum:   {np.mean(vis):.2f}",
            f"Min/Max:    {np.min(vis):.2f} / {np.max(vis):.2f}",
            "Pathway:    Retina -> LMC L1/L2",
        ]
        for i, text in enumerate(lines):
            t = self.font_small.render(text, True, (156, 163, 175))
            self.screen.blit(t, (anno_x, y + i * 18))

        return y + grid_h + 20

    def _draw_motion_detectors(self, x: int, y: int) -> int:
        lbl = self.font_sub.render("2. Elementary Motion (T4 ON / T5 OFF)", True, (226, 232, 240))
        self.screen.blit(lbl, (x, y))
        y += 20

        # Cardinal motion channels
        dirs = [
            ("Flow Right", self.activations.get("flow_right", 0.0), (34, 197, 94)),
            ("Flow Left", self.activations.get("flow_left", 0.0), (59, 130, 246)),
            ("Flow Up", self.activations.get("flow_up", 0.0), (234, 179, 8)),
            ("Flow Down", self.activations.get("flow_down", 0.0), (168, 85, 247)),
        ]

        bar_w = 180
        bar_h = 10
        for name, val, col in dirs:
            val_norm = np.clip(float(val) * 10.0, 0.0, 1.0)
            self.screen.blit(self.font_small.render(f"{name:10s}", True, (203, 213, 225)), (x, y))
            # Background bar
            bg_rect = pygame.Rect(x + 90, y + 2, bar_w, bar_h)
            pygame.draw.rect(self.screen, (30, 41, 59), bg_rect, border_radius=3)
            # Active fill
            fill_rect = pygame.Rect(x + 90, y + 2, int(bar_w * val_norm), bar_h)
            pygame.draw.rect(self.screen, col, fill_rect, border_radius=3)
            # Value
            val_t = self.font_small.render(f"{val:.3f}", True, (148, 163, 184))
            self.screen.blit(val_t, (x + 90 + bar_w + 8, y))
            y += 18

        return y + 10

    def _draw_lptc_section(self, x: int, y: int) -> int:
        lbl = self.font_sub.render("3. Lobula Plate Tangential & Looming Cells", True, (226, 232, 240))
        self.screen.blit(lbl, (x, y))
        y += 20

        # HS (Horizontal System) Yaw cell [-1.0, 1.0]
        hs_val = float(self.activations.get("hs_cell", 0.0))
        self._draw_bipolar_meter(x, y, "HS (Yaw)", hs_val, 180, (56, 189, 248))
        y += 24

        # VS (Vertical System) Pitch cell [-1.0, 1.0]
        vs_val = float(self.activations.get("vs_cell", 0.0))
        self._draw_bipolar_meter(x, y, "VS (Pitch)", vs_val, 180, (251, 146, 60))
        y += 24

        # LPLC2 Looming Collision Detector [0.0, 1.0]
        looming = float(self.activations.get("lplc2_looming", 0.0))
        is_alert = looming > 0.35
        looming_col = (239, 68, 68) if is_alert else (148, 163, 184)
        name_str = "LPLC2 (LOOMING ALERT!)" if is_alert else "LPLC2 (Collision)"
        self.screen.blit(self.font_small.render(name_str, True, looming_col), (x, y))

        bg_rect = pygame.Rect(x + 160, y + 2, 110, 12)
        pygame.draw.rect(self.screen, (30, 41, 59), bg_rect, border_radius=3)
        fill_w = int(110 * np.clip(looming * 2.0, 0.0, 1.0))
        fill_rect = pygame.Rect(x + 160, y + 2, fill_w, 12)
        pygame.draw.rect(self.screen, (239, 68, 68), fill_rect, border_radius=3)
        val_t = self.font_small.render(f"{looming:.3f}", True, (203, 213, 225))
        self.screen.blit(val_t, (x + 280, y))

        return y + 30

    def _draw_bipolar_meter(self, x: int, y: int, label: str, val: float, width: int, color):
        self.screen.blit(self.font_small.render(f"{label:12s}", True, (203, 213, 225)), (x, y))
        center_x = x + 120 + width // 2
        bg_rect = pygame.Rect(x + 120, y + 2, width, 10)
        pygame.draw.rect(self.screen, (30, 41, 59), bg_rect, border_radius=3)
        # Center tick
        pygame.draw.line(self.screen, (100, 116, 139), (center_x, y), (center_x, y + 14), 1)

        val_clamped = np.clip(val * 5.0, -1.0, 1.0)
        bar_len = int((width // 2) * abs(val_clamped))
        if val_clamped >= 0:
            fill_rect = pygame.Rect(center_x, y + 2, bar_len, 10)
        else:
            fill_rect = pygame.Rect(center_x - bar_len, y + 2, bar_len, 10)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=2)
        self.screen.blit(self.font_small.render(f"{val:+.3f}", True, (148, 163, 184)), (x + 120 + width + 8, y))

    def _draw_flight_meters(self, x: int, y: int):
        lbl = self.font_sub.render("4. Motor Torques & Wing Dynamics", True, (226, 232, 240))
        self.screen.blit(lbl, (x, y))
        y += 20

        phys = self.env.physics.get_state()
        lines = [
            f"Wing Beat Freq:  {phys['wing_freq']:.1f} Hz  (Stroke Amp: {np.rad2deg(phys['amp_left']):.1f}°)",
            f"Lift Force:      {phys['lift_force'] * 1e6:.1f} µN  (Thrust: {phys['thrust_force'] * 1e6:.1f} µN)",
            f"Flight Speed:    {phys['airspeed']:.2f} m/s",
            f"Current Episode: {self.episodes}  |  Cum. Reward: {self.cumulative_reward:.1f}",
        ]
        for i, line in enumerate(lines):
            t = self.font_text.render(line, True, (156, 163, 175))
            self.screen.blit(t, (x, y + i * 20))
