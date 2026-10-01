"""
FlyArcade Pygame Renderer
=========================
Renders the arcade simulation: fruit fly avatar with biomechanically animated
flapping wings, vortex wake particles, dynamic obstacle gates, and telemetry HUD.
"""

from typing import Optional
import numpy as np
import pygame


class ArcadeRenderer:
    """Pygame-based high-frame-rate arcade renderer for FlyArcadeEnv."""

    def __init__(self, env, width: int = 640, height: int = 480, mode: str = "human"):
        self.env = env
        self.width = width
        self.height = height
        self.mode = mode

        pygame.init()
        if self.mode == "human":
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("BioFly-Arcade Engine: Flight Simulation")
        else:
            self.screen = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)

        # Wake particles
        self.particles = []

    def render(self) -> Optional[np.ndarray]:
        """Draws current frame and returns RGB array."""
        surf = self.screen
        surf.fill((18, 22, 30))  # Dark deep-slate background

        # Draw grid lines for speed illusion
        fly_x = float(self.env.physics.pos[0])
        fly_y = float(self.env.physics.pos[2])
        grid_offset = int((fly_x * 80) % 40)
        for gx in range(-grid_offset, self.width, 40):
            pygame.draw.line(surf, (28, 34, 46), (gx, 0), (gx, self.height), 1)
        for gy in range(0, self.height, 40):
            pygame.draw.line(surf, (28, 34, 46), (0, gy), (self.width, gy), 1)

        # Camera view: fly stays centered horizontally at x = 150
        camera_x = fly_x - 1.5
        pixels_per_meter = 120.0

        # Draw obstacle pillars
        for obs in self.env.obstacles:
            screen_x = int((obs.x - camera_x) * pixels_per_meter)
            if -100 <= screen_x <= self.width + 100:
                obs_w = int(obs.width * pixels_per_meter)
                # Gap top and bottom in screen coordinates (inverted Y)
                gap_top_y = int(self.height * (1.0 - (obs.gap_y + obs.gap_height / 2.0)))
                gap_bot_y = int(self.height * (1.0 - (obs.gap_y - obs.gap_height / 2.0)))

                # Top pillar
                if gap_top_y > 0:
                    top_rect = pygame.Rect(screen_x - obs_w // 2, 0, obs_w, gap_top_y)
                    pygame.draw.rect(surf, (55, 65, 81), top_rect, border_radius=4)
                    pygame.draw.rect(surf, (75, 85, 99), top_rect, width=2, border_radius=4)
                    # Hazard stripes on edge
                    pygame.draw.line(surf, (239, 68, 68), (screen_x - obs_w // 2, gap_top_y), (screen_x + obs_w // 2, gap_top_y), 3)

                # Bottom pillar
                if gap_bot_y < self.height:
                    bot_rect = pygame.Rect(screen_x - obs_w // 2, gap_bot_y, obs_w, self.height - gap_bot_y)
                    pygame.draw.rect(surf, (55, 65, 81), bot_rect, border_radius=4)
                    pygame.draw.rect(surf, (75, 85, 99), bot_rect, width=2, border_radius=4)
                    # Hazard stripes on edge
                    pygame.draw.line(surf, (239, 68, 68), (screen_x - obs_w // 2, gap_bot_y), (screen_x + obs_w // 2, gap_bot_y), 3)

                # Sucrose Food Droplet
                if obs.has_food and not obs.food_collected:
                    food_y = int(self.height * (1.0 - obs.gap_y))
                    pulse = int(3 * np.sin(pygame.time.get_ticks() * 0.008))
                    pygame.draw.circle(surf, (245, 158, 11), (screen_x, food_y), 8 + pulse)
                    pygame.draw.circle(surf, (253, 230, 138), (screen_x, food_y), 4 + pulse)

        # Fly screen position
        fly_screen_x = int((fly_x - camera_x) * pixels_per_meter)
        fly_screen_y = int(self.height * (1.0 - fly_y))

        # Add wingtip wake particles
        if len(self.particles) < 40 and np.random.rand() > 0.3:
            self.particles.append([fly_screen_x - 12, fly_screen_y, 1.0, (96, 165, 250)])
        new_particles = []
        for p in self.particles:
            p[0] -= 3.0  # drifts backward
            p[2] -= 0.05  # fades out
            if p[2] > 0.0:
                alpha_r = int(p[3][0] * p[2])
                alpha_g = int(p[3][1] * p[2])
                alpha_b = int(p[3][2] * p[2])
                pygame.draw.circle(surf, (alpha_r, alpha_g, alpha_b), (int(p[0]), int(p[1])), int(4 * p[2]))
                new_particles.append(p)
        self.particles = new_particles

        # Draw Drosophila Fly Avatar
        self._draw_fly(surf, fly_screen_x, fly_screen_y)

        # HUD Overlay
        self._draw_hud(surf)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(50)

        # Return RGB array (H, W, 3)
        rgb = pygame.surfarray.array3d(surf)
        return np.transpose(rgb, (1, 0, 2))

    def _draw_fly(self, surf: pygame.Surface, x: int, y: int):
        """Draws realistic fruit fly with stroke phase flapping wings."""
        pitch = float(self.env.physics.angles[0])
        wing_phase = float(self.env.physics.wing_phase)

        # Pitch rotation vector
        cos_p = np.cos(-pitch)
        sin_p = np.sin(-pitch)

        # Fly Thorax (Oval)
        thorax_color = (180, 110, 45)  # Amber Drosophila cuticle
        pygame.draw.ellipse(surf, thorax_color, (x - 12, y - 7, 24, 14))

        # Abdomen (Striped brown segments)
        ab_x = x - 18
        ab_y = y - 4
        pygame.draw.ellipse(surf, (140, 80, 30), (ab_x, ab_y, 16, 10))
        # Stripes
        pygame.draw.line(surf, (70, 40, 15), (ab_x + 5, ab_y + 1), (ab_x + 5, ab_y + 8), 1)
        pygame.draw.line(surf, (70, 40, 15), (ab_x + 10, ab_y + 1), (ab_x + 10, ab_y + 8), 1)

        # Head with compound eyes
        head_x = x + 10
        head_y = y
        pygame.draw.circle(surf, (150, 90, 35), (head_x, head_y), 6)
        # Red ommatidia compound eyes
        eye_color = (220, 38, 38)
        pygame.draw.circle(surf, eye_color, (head_x + 2, head_y - 3), 3)
        pygame.draw.circle(surf, eye_color, (head_x + 2, head_y + 3), 3)

        # Flapping Wings (stroke amplitude oscillation)
        wing_angle = np.sin(wing_phase) * 0.9  # flapping elevation
        wing_len = 22
        wing_w = 9

        # Left/Top wing
        wing_surf = pygame.Surface((wing_len * 2, wing_w * 2), pygame.SRCALPHA)
        wing_color = (210, 230, 245, 170)  # Translucent wing membrane
        pygame.draw.ellipse(wing_surf, wing_color, (0, 0, wing_len * 2, wing_w * 2))
        pygame.draw.ellipse(wing_surf, (180, 200, 220, 220), (0, 0, wing_len * 2, wing_w * 2), width=1)
        # Veins
        pygame.draw.line(wing_surf, (160, 180, 200, 180), (4, wing_w), (wing_len * 2 - 4, wing_w), 1)

        # Rotate wing by flap angle + body pitch
        rot_deg = np.rad2deg(-pitch + wing_angle)
        rot_wing = pygame.transform.rotate(wing_surf, rot_deg)
        w_rect = rot_wing.get_rect(center=(x, y - 6))
        surf.blit(rot_wing, w_rect.topleft)

    def _draw_hud(self, surf: pygame.Surface):
        """Draws score and flight dynamics metrics."""
        phys = self.env.physics.get_state()
        score_text = self.font_large.render(f"SCORE: {self.env.score}   SUCROSE: {self.env.total_food}", True, (245, 158, 11))
        surf.blit(score_text, (20, 15))

        stats = [
            f"Airspeed:    {phys['airspeed']:.2f} m/s",
            f"Altitude:    {phys['pos'][2]:.2f} m",
            f"Wingbeat:    {phys['wing_freq']:.0f} Hz",
            f"Lift Force:  {phys['lift_force'] * 1e6:.1f} uN",
            f"Pitch Angle: {np.rad2deg(phys['angles'][0]):+.1f} deg",
        ]

        for i, text in enumerate(stats):
            t_surf = self.font.render(text, True, (156, 163, 175))
            surf.blit(t_surf, (20, 48 + i * 16))

    def close(self):
        pygame.quit()
