"""
Drosophila Aerodynamic Flight Dynamics
======================================
6-Degree-of-Freedom (6-DoF) quasi-steady biomechanical flight dynamics
calibrated to Drosophila melanogaster empirical kinematic measurements
(Dickinson et al., 1999; Fry et al., 2003; Muijres et al., 2014).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FlyMorphology:
    """Biomechanical constants for Drosophila melanogaster."""
    mass: float = 1.2e-6          # Body mass in kg (~1.2 mg)
    length: float = 2.5e-3        # Body length in meters (~2.5 mm)
    wing_length: float = 2.0e-3   # Wing span radius R (~2.0 mm)
    wing_area: float = 1.8e-6     # Single wing surface area in m^2
    wing_freq_nominal: float = 218.0  # Wingbeat frequency in Hz (~220 Hz)
    stroke_amplitude_nominal: float = 145.0 * np.pi / 180.0  # ~2.53 rad (145 deg)
    air_density: float = 1.204    # Air density at 20°C in kg/m^3
    gravity: float = 9.81         # Gravitational acceleration (m/s^2)
    # Drag and lift coefficients (quasi-steady + delayed stall vortex)
    c_lift_max: float = 1.8
    c_drag_trans: float = 1.6
    rotational_damping: float = 4.5e-9  # N*m*s/rad aerodynamic yaw/pitch damping


class FlyFlightDynamics:
    """
    Simulates real-time 6-DoF rigid body flight dynamics driven by bilateral
    wing flapping forces and torques.
    """

    def __init__(self, morphology: FlyMorphology | None = None, dt: float = 0.01):
        self.morphology = morphology or FlyMorphology()
        self.dt = dt

        # Scaled spatial units: for arcade gameplay, we keep SI units internally
        # and provide scaled coordinate views (0.0 to 1.0 or pixel coordinates).
        self.reset()

    def reset(self, initial_pos: np.ndarray | None = None, initial_vel: np.ndarray | None = None):
        """Reset state to initial hovering/cruising conditions."""
        if initial_pos is not None:
            self.pos = np.array(initial_pos, dtype=np.float64)
        else:
            self.pos = np.array([0.0, 0.0, 0.5], dtype=np.float64)  # x, y, z in meters

        if initial_vel is not None:
            self.vel = np.array(initial_vel, dtype=np.float64)
        else:
            # Nominal cruising forward velocity (~0.8 m/s in Drosophila)
            self.vel = np.array([0.8, 0.0, 0.0], dtype=np.float64)

        # Euler angles: pitch (theta), roll (phi), yaw (psi) in radians
        self.angles = np.zeros(3, dtype=np.float64)
        self.ang_vel = np.zeros(3, dtype=np.float64)  # wx, wy, wz in rad/s

        # Wing stroke kinematics
        self.wing_phase = 0.0  # radians [0, 2*pi]
        self.wing_freq = self.morphology.wing_freq_nominal
        self.amp_left = self.morphology.stroke_amplitude_nominal
        self.amp_right = self.morphology.stroke_amplitude_nominal

        # Telemetry records
        self.last_lift_force = 0.0
        self.last_thrust_force = 0.0
        self.last_yaw_torque = 0.0
        self.last_pitch_torque = 0.0

        return self.get_state()

    def step(self, action: np.ndarray):
        """
        Advance flight dynamics by dt given control action.
        Action vector:
            action[0]: Flap intensity / Lift control [-1.0, 1.0]
                       (-1.0 = glide/dive, 0.0 = hover equilibrium, 1.0 = maximum climb)
            action[1]: Pitch steering control [-1.0, 1.0]
                       (-1.0 = nose-down/dive, 1.0 = nose-up/climb)
            action[2] (optional): Yaw / Bilateral steering [-1.0, 1.0]
                       (-1.0 = steer left, 1.0 = steer right)
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        flap_ctrl = float(action[0])
        pitch_ctrl = float(action[1]) if len(action) > 1 else 0.0
        yaw_ctrl = float(action[2]) if len(action) > 2 else 0.0

        m = self.morphology
        dt = self.dt

        # 1. Bilateral wing amplitude modulation
        # Fly adjusts left vs right stroke amplitude to create yaw torque
        amp_nominal = m.stroke_amplitude_nominal
        delta_amp = 0.25 * amp_nominal * yaw_ctrl
        self.amp_left = np.clip(amp_nominal - delta_amp, 0.5 * amp_nominal, 1.4 * amp_nominal)
        self.amp_right = np.clip(amp_nominal + delta_amp, 0.5 * amp_nominal, 1.4 * amp_nominal)

        # Modulate frequency based on flap intensity: nominal ~220 Hz, range [160, 280] Hz
        self.wing_freq = m.wing_freq_nominal * (1.0 + 0.3 * flap_ctrl)
        self.wing_phase = (self.wing_phase + 2.0 * np.pi * self.wing_freq * dt) % (2.0 * np.pi)

        # 2. Aerodynamic forces:
        # Mean wing tip speed: u_tip = 2 * phi * f * R
        u_mean = 2.0 * ((self.amp_left + self.amp_right) / 2.0) * self.wing_freq * m.wing_length
        q_dyn = 0.5 * m.air_density * (u_mean ** 2) * (2.0 * m.wing_area)

        # Baseline lift to counteract gravity at flap_ctrl = 0
        equilibrium_lift = m.mass * m.gravity
        lift_gain = equilibrium_lift * 1.6  # Up to 2.6g acceleration
        lift_force = equilibrium_lift + (flap_ctrl * lift_gain)
        lift_force = max(0.0, lift_force)

        # Forward thrust generated by stroke plane angle and body pitch
        # Flies pitch forward to direct lift vector into forward thrust
        body_pitch = self.angles[0]
        f_thrust = lift_force * np.sin(-body_pitch + 0.1 * pitch_ctrl)
        f_lift_z = lift_force * np.cos(-body_pitch)

        # Drag force opposed to current velocity
        speed = float(np.linalg.norm(self.vel))
        if speed > 1e-6:
            f_drag = 0.5 * m.air_density * m.c_drag_trans * (m.length * m.length) * (speed ** 2) * (self.vel / speed)
        else:
            f_drag = np.zeros(3)

        # Total linear force in world coordinates
        f_total = np.zeros(3)
        # Forward axis (X) thrust
        f_total[0] = f_thrust - f_drag[0]
        # Lateral axis (Y) side-slip
        f_total[1] = -f_drag[1]
        # Vertical axis (Z) lift vs gravity
        f_total[2] = f_lift_z - (m.mass * m.gravity) - f_drag[2]

        # 3. Aerodynamic torques
        # Bilateral wing amplitude difference generates yaw torque
        yaw_torque = 2.5e-8 * (self.amp_right - self.amp_left) / amp_nominal
        # Pitch control tilts stroke plane relative to center of mass
        pitch_torque = 3.0e-8 * pitch_ctrl

        # 4. Integrate equations of motion
        # Linear acceleration
        acc = f_total / m.mass
        self.vel += acc * dt
        # Velocity bounds: Drosophila flight speeds reach 0.2 to 2.5 m/s
        self.vel[0] = np.clip(self.vel[0], 0.2, 3.0)
        self.vel[2] = np.clip(self.vel[2], -2.0, 2.0)
        self.pos += self.vel * dt

        # Stable implicit rotational damping:
        # omega_new = (omega + tau_applied / I * dt) / (1 + b / I * dt)
        tau_applied = np.array([0.0, pitch_torque, yaw_torque], dtype=np.float64)
        inertia = 5.0e-13
        decay_factor = 1.0 + (m.rotational_damping / inertia) * dt
        self.ang_vel = (self.ang_vel + (tau_applied / inertia) * dt) / decay_factor

        # Physiological Drosophila saccade angular velocity limits (~30 rad/s)
        self.ang_vel = np.clip(self.ang_vel, -30.0, 30.0)
        self.angles += self.ang_vel * dt

        # Bound pitch to [-60 deg, +60 deg]
        max_pitch = np.deg2rad(60.0)
        self.angles[0] = np.clip(self.angles[0], -max_pitch, max_pitch)

        # Telemetry
        self.last_lift_force = float(f_lift_z)
        self.last_thrust_force = float(f_thrust)
        self.last_yaw_torque = float(yaw_torque)
        self.last_pitch_torque = float(pitch_torque)

        return self.get_state()

    def get_state(self) -> dict:
        """Returns physical state dictionary."""
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "angles": self.angles.copy(),
            "ang_vel": self.ang_vel.copy(),
            "wing_phase": self.wing_phase,
            "wing_freq": self.wing_freq,
            "amp_left": self.amp_left,
            "amp_right": self.amp_right,
            "lift_force": self.last_lift_force,
            "thrust_force": self.last_thrust_force,
            "airspeed": float(np.linalg.norm(self.vel)),
        }
