"""
Connectome-Constrained Drosophila Visual Pathway
================================================
A biologically grounded deep mechanistic vision network modeling the Drosophila
optic lobe: Ommatidia -> Lamina (L1/L2) -> Medulla (Mi1/Tm3/Tm1/Tm2) ->
Lobula Plate (T4/T5 directional motion detectors) -> LPTC (HS/VS) and LPLC2 looming cells.

Based on connectome visual architectures (TuragaLab/flyvis; Borst et al., 2020;
Klapoetke et al., 2022).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmmatidiaRetina(nn.Module):
    """
    Retinotopic compound eye photoreceptors (R1-R6).
    Performs spatial low-pass filtering (angular acceptance function)
    and logarithmic luminance adaptation: R = log(1 + I / I_0).
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 3):
        super().__init__()
        # Gaussian smoothing filter representing the ommatidial acceptance angle (~5.5 deg)
        sigma = 1.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("spatial_kernel", kernel.view(1, 1, kernel_size, kernel_size))
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB to luminance if 3 channels
        if x.dim() == 4 and x.size(1) == 3:
            # Grayscale luminance weights
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1, 3, 1, 1)
            x = (x * weights).sum(dim=1, keepdim=True)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Spatial ommatidial blur
        smoothed = F.conv2d(x, self.spatial_kernel, padding=self.pad)
        # Non-linear photoreceptor response (Naka-Rushton / Weber adaptation)
        r_photoreceptors = torch.log1p(torch.relu(smoothed))
        return r_photoreceptors


class LaminaONOFF(nn.Module):
    """
    Lamina Monopolar Cells (LMCs):
    L1: ON-pathway (responds to positive temporal contrast changes / brightening)
    L2: OFF-pathway (responds to negative temporal contrast changes / dimming)
    """

    def __init__(self):
        super().__init__()
        self.prev_activation = None

    def reset_state(self):
        self.prev_activation = None

    def forward(self, r_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prev_activation is None or self.prev_activation.shape != r_current.shape:
            # Temporal derivative is initially zero
            delta = torch.zeros_like(r_current)
        else:
            delta = r_current - self.prev_activation

        # Update cached state (detached to prevent backprop leak through episodes)
        self.prev_activation = r_current.detach()

        # L1: Half-wave rectified ON contrast
        l1_on = torch.relu(delta)
        # L2: Half-wave rectified OFF contrast
        l2_off = torch.relu(-delta)
        return l1_on, l2_off


class MedullaDelayLine(nn.Module):
    """
    Medulla interneurons creating spatiotemporal delay filters:
    Mi1 & Tm3 (ON pathway interneurons with delay tau1, tau2)
    Tm1, Tm2, Tm4 (OFF pathway interneurons)
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # Fast and delayed temporal decay factors (leaky integrator)
        self.alpha_fast = 0.85
        self.alpha_slow = 0.40
        self.delayed_on = None
        self.delayed_off = None

    def reset_state(self):
        self.delayed_on = None
        self.delayed_off = None

    def forward(self, l1_on: torch.Tensor, l2_off: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.delayed_on is None or self.delayed_on.shape != l1_on.shape:
            self.delayed_on = torch.zeros_like(l1_on)
            self.delayed_off = torch.zeros_like(l2_off)

        # Leaky temporal integration
        self.delayed_on = self.alpha_slow * self.delayed_on + (1.0 - self.alpha_slow) * l1_on.detach()
        self.delayed_off = self.alpha_slow * self.delayed_off + (1.0 - self.alpha_slow) * l2_off.detach()

        mi1 = l1_on  # Undelayed branch
        tm3 = self.delayed_on  # Delayed branch
        tm1 = l2_off  # Undelayed OFF
        tm2 = self.delayed_off  # Delayed OFF

        return {"mi1": mi1, "tm3": tm3, "tm1": tm1, "tm2": tm2}


class T4T5ElementaryMotionDetector(nn.Module):
    """
    Hassenstein-Reichardt Elementary Motion Detectors (EMD):
    T4 neurons (ON-motion) and T5 neurons (OFF-motion) computing directional
    motion along 4 cardinal directions:
    - Right (0), Left (1), Up (2), Down (3)
    """

    def __init__(self):
        super().__init__()
        # Spatial neighbor shift kernels for 4 cardinal directions
        # Right shift: neighbor at (x-1, y)
        k_right = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        k_right[0, 0, 1, 0] = 1.0

        # Left shift: neighbor at (x+1, y)
        k_left = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        k_left[0, 0, 1, 2] = 1.0

        # Up shift: neighbor at (x, y+1)
        k_up = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        k_up[0, 0, 2, 1] = 1.0

        # Down shift: neighbor at (x, y-1)
        k_down = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        k_down[0, 0, 0, 1] = 1.0

        self.register_buffer("k_right", k_right)
        self.register_buffer("k_left", k_left)
        self.register_buffer("k_up", k_up)
        self.register_buffer("k_down", k_down)

    def _correlate_direction(self, direct: torch.Tensor, delayed: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # Spatial neighbor of delayed signal
        delayed_neighbor = F.conv2d(delayed, kernel, padding=1)
        # Non-linear correlation / shunting product
        corr = direct * delayed_neighbor
        return torch.relu(corr)

    def forward(self, medulla_signals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mi1 = medulla_signals["mi1"]
        tm3 = medulla_signals["tm3"]
        tm1 = medulla_signals["tm1"]
        tm2 = medulla_signals["tm2"]

        # T4 (ON motion): correlate Mi1 with delayed Tm3
        t4_right = self._correlate_direction(mi1, tm3, self.k_right)
        t4_left = self._correlate_direction(mi1, tm3, self.k_left)
        t4_up = self._correlate_direction(mi1, tm3, self.k_up)
        t4_down = self._correlate_direction(mi1, tm3, self.k_down)

        # T5 (OFF motion): correlate Tm1 with delayed Tm2
        t5_right = self._correlate_direction(tm1, tm2, self.k_right)
        t5_left = self._correlate_direction(tm1, tm2, self.k_left)
        t5_up = self._correlate_direction(tm1, tm2, self.k_up)
        t5_down = self._correlate_direction(tm1, tm2, self.k_down)

        return {
            "t4_right": t4_right,
            "t4_left": t4_left,
            "t4_up": t4_up,
            "t4_down": t4_down,
            "t5_right": t5_right,
            "t5_left": t5_left,
            "t5_up": t5_up,
            "t5_down": t5_down,
        }


class LPTCTangentialCells(nn.Module):
    """
    Lobula Plate Tangential Cells (LPTCs) & Looming Detectors:
    - HS (Horizontal System): computes wide-field horizontal optic flow (Yaw = Right - Left)
    - VS (Vertical System): computes wide-field vertical optic flow (Pitch = Up - Down)
    - LPLC2 (Lobula Plate Looming-sensitive Cells): computes radial divergence / looming expansion
      indicating approaching collision.
    """

    def __init__(self):
        super().__init__()
        # Divergence / expansion kernel for LPLC2 radial optic flow
        # Measures outwards flow from center
        self.register_buffer("dummy", torch.tensor(0.0))

    def forward(self, t4t5: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Aggregate ON and OFF motion for total directional flow
        flow_right = (t4t5["t4_right"] + t4t5["t5_right"]).mean(dim=(1, 2, 3), keepdim=True)
        flow_left = (t4t5["t4_left"] + t4t5["t5_left"]).mean(dim=(1, 2, 3), keepdim=True)
        flow_up = (t4t5["t4_up"] + t4t5["t5_up"]).mean(dim=(1, 2, 3), keepdim=True)
        flow_down = (t4t5["t4_down"] + t4t5["t5_down"]).mean(dim=(1, 2, 3), keepdim=True)

        # HS (Horizontal System): yaw rotation flow
        hs_cell = flow_right - flow_left

        # VS (Vertical System): pitch translation/rotation flow
        vs_cell = flow_up - flow_down

        # LPLC2 Looming Collision Detector:
        # Fired by expanding dark edges (OFF motion diverging outwards)
        # In Drosophila, LPLC2 dendrites point radially outward in the lobula plate
        t5_r = t4t5["t5_right"]
        t5_l = t4t5["t5_left"]
        t5_u = t4t5["t5_up"]
        t5_d = t4t5["t5_down"]

        b, c, h, w = t5_r.shape
        # Spatial coordinate weights from image center
        y = torch.linspace(-1.0, 1.0, h, device=t5_r.device).view(1, 1, h, 1)
        x = torch.linspace(-1.0, 1.0, w, device=t5_r.device).view(1, 1, 1, w)

        # Outward radial component: (flow_x * x) + (flow_y * y)
        radial_expansion = (t5_r - t5_l) * x + (t5_u - t5_d) * y
        lplc2_looming = torch.relu(radial_expansion).mean(dim=(1, 2, 3), keepdim=True)

        return {
            "hs_cell": hs_cell.squeeze(-1).squeeze(-1),
            "vs_cell": vs_cell.squeeze(-1).squeeze(-1),
            "lplc2_looming": lplc2_looming.squeeze(-1).squeeze(-1),
            "flow_right": flow_right.squeeze(-1).squeeze(-1),
            "flow_left": flow_left.squeeze(-1).squeeze(-1),
            "flow_up": flow_up.squeeze(-1).squeeze(-1),
            "flow_down": flow_down.squeeze(-1).squeeze(-1),
        }


class DrosophilaVisionConnectome(nn.Module):
    """
    Complete biological vision pipeline.
    Transforms raw retina image -> Ommatidia -> Lamina -> Medulla -> T4/T5 -> LPTC / LPLC2.
    Also caches real-time firing rates for live GUI telemetry.
    """

    def __init__(self, eye_resolution: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.eye_resolution = eye_resolution
        self.retina = OmmatidiaRetina()
        self.lamina = LaminaONOFF()
        self.medulla = MedullaDelayLine()
        self.emd = T4T5ElementaryMotionDetector()
        self.lptc = LPTCTangentialCells()

        # Connectome feature compression layer projecting biological activations
        # into compact embedding vector
        # Features: [HS (1), VS (1), LPLC2 (1), Flow R/L/U/D (4), Ommatidia pooled (16), T4 pooled (4), T5 pooled (4)] = 31
        self.feature_dim = 32
        self.proj = nn.Linear(31, self.feature_dim)

        # Telemetry activity cache
        self.last_activations: Dict[str, torch.Tensor] = {}

    def reset_states(self):
        """Reset temporal state across episodes."""
        self.lamina.reset_state()
        self.medulla.reset_state()
        self.last_activations = {}

    def forward(self, visual_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through connectome visual system.
        visual_input: [Batch, Channels, Height, Width] or [Batch, Height, Width]
        Returns:
            embedding: [Batch, feature_dim] tensor
            activations: Dictionary containing layer activations for dashboard
        """
        # Ensure 4D tensor
        if visual_input.dim() == 3:
            visual_input = visual_input.unsqueeze(1)
        elif visual_input.dim() == 2:
            visual_input = visual_input.unsqueeze(0).unsqueeze(0)

        # 1. Ommatidia Photoreceptors
        r_photoreceptors = self.retina(visual_input)

        # 2. Lamina L1 (ON) and L2 (OFF)
        l1_on, l2_off = self.lamina(r_photoreceptors)

        # 3. Medulla Delay Lines
        medulla_out = self.medulla(l1_on, l2_off)

        # 4. T4 (ON) & T5 (OFF) Directional Motion Detectors
        t4t5_out = self.emd(medulla_out)

        # 5. LPTC & Looming Detectors
        lptc_out = self.lptc(t4t5_out)

        # Retinotopic spatial pooling for global summary (flattened to 2D B x D)
        pool_r = F.adaptive_avg_pool2d(r_photoreceptors, (4, 4)).flatten(start_dim=1)  # (B, 16)
        pool_t4 = torch.cat([
            t4t5_out["t4_right"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t4_left"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t4_up"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t4_down"].mean(dim=(1, 2, 3), keepdim=True),
        ], dim=1).flatten(start_dim=1)  # (B, 4)
        pool_t5 = torch.cat([
            t4t5_out["t5_right"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t5_left"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t5_up"].mean(dim=(1, 2, 3), keepdim=True),
            t4t5_out["t5_down"].mean(dim=(1, 2, 3), keepdim=True),
        ], dim=1).flatten(start_dim=1)  # (B, 4)

        # Concatenate biological feature vector (all 2D: Batch x Channels)
        bio_features = torch.cat([
            lptc_out["hs_cell"].flatten(start_dim=1),
            lptc_out["vs_cell"].flatten(start_dim=1),
            lptc_out["lplc2_looming"].flatten(start_dim=1),
            lptc_out["flow_right"].flatten(start_dim=1),
            lptc_out["flow_left"].flatten(start_dim=1),
            lptc_out["flow_up"].flatten(start_dim=1),
            lptc_out["flow_down"].flatten(start_dim=1),
            pool_r,
            pool_t4,
            pool_t5,
        ], dim=1)

        embedding = F.gelu(self.proj(bio_features))

        # Cache activations for dashboard telemetry
        self.last_activations = {
            "ommatidia": r_photoreceptors.detach(),
            "l1_on": l1_on.detach(),
            "l2_off": l2_off.detach(),
            "t4_right": t4t5_out["t4_right"].detach(),
            "t4_left": t4t5_out["t4_left"].detach(),
            "t4_up": t4t5_out["t4_up"].detach(),
            "t4_down": t4t5_out["t4_down"].detach(),
            "t5_right": t4t5_out["t5_right"].detach(),
            "t5_left": t4t5_out["t5_left"].detach(),
            "t5_up": t4t5_out["t5_up"].detach(),
            "t5_down": t4t5_out["t5_down"].detach(),
            "hs_cell": float(lptc_out["hs_cell"][0].item()),
            "vs_cell": float(lptc_out["vs_cell"][0].item()),
            "lplc2_looming": float(lptc_out["lplc2_looming"][0].item()),
            "flow_right": float(lptc_out["flow_right"][0].item()),
            "flow_left": float(lptc_out["flow_left"][0].item()),
            "flow_up": float(lptc_out["flow_up"][0].item()),
            "flow_down": float(lptc_out["flow_down"][0].item()),
        }

        return embedding, self.last_activations
