"""
MuJoCo Biomechanical Drosophila Fly Simulation
==============================================
Loads and manages a 3D biomechanical model of Drosophila melanogaster in MuJoCo.
Supports headless rendering (MUJOCO_GL=egl or osmesa), wing stroke actuation,
and compound eye vision synthesis.
"""

import os
import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


# Drosophila melanogaster MJCF (MuJoCo XML) Model
DROSOPHILA_MJCF = """
<mujoco model="drosophila_melanogaster">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.001">
    <flag contact="enable" energy="enable"/>
  </option>

  <default>
    <joint damping="0.0001" armature="0.00001"/>
    <geom condim="3" friction="1.0 0.1 0.1" density="1000"/>
  </default>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
  </visual>

  <asset>
    <material name="cuticle" rgba="0.45 0.25 0.12 1.0" shininess="0.5"/>
    <material name="eye" rgba="0.75 0.12 0.12 1.0" shininess="0.8"/>
    <material name="wing" rgba="0.85 0.90 0.95 0.4" shininess="0.9"/>
    <material name="ground" rgba="0.2 0.22 0.25 1.0"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="ground"/>

    <!-- Free-floating 6-DoF Drosophila Root Body -->
    <body name="fly_thorax" pos="0 0 1.0">
      <freejoint name="fly_free_joint"/>
      <!-- Thorax (central body capsule) -->
      <geom name="geom_thorax" type="capsule" fromto="-0.001 0 0 0.001 0 0" size="0.0008" material="cuticle"/>

      <!-- Head with bilateral compound eyes -->
      <body name="fly_head" pos="0.0015 0 0.0002">
        <geom name="geom_head" type="sphere" size="0.0006" material="cuticle"/>
        <!-- Left eye -->
        <geom name="eye_left" type="sphere" pos="0.0002 0.0005 0.0002" size="0.00035" material="eye"/>
        <!-- Right eye -->
        <geom name="eye_right" type="sphere" pos="0.0002 -0.0005 0.0002" size="0.00035" material="eye"/>
        <!-- Forward-facing compound eye camera -->
        <camera name="compound_eye_cam" pos="0.0005 0 0.0002" euler="90 0 -90" fovy="120"/>
      </body>

      <!-- Abdomen -->
      <body name="fly_abdomen" pos="-0.0018 0 -0.0002">
        <geom name="geom_abdomen" type="capsule" fromto="0 0 0 -0.002 0 -0.0004" size="0.0007" material="cuticle"/>
      </body>

      <!-- Left Wing with stroke and pitch hinge joints -->
      <body name="wing_left" pos="0 0.0007 0.0004">
        <joint name="wing_stroke_left" type="hinge" axis="0 0 1" range="-70 70"/>
        <joint name="wing_pitch_left" type="hinge" axis="0 1 0" range="-45 45"/>
        <geom name="geom_wing_left" type="box" pos="-0.0005 0.001 0" size="0.0008 0.0015 0.00002" material="wing"/>
      </body>

      <!-- Right Wing with stroke and pitch hinge joints -->
      <body name="wing_right" pos="0 -0.0007 0.0004">
        <joint name="wing_stroke_right" type="hinge" axis="0 0 1" range="-70 70"/>
        <joint name="wing_pitch_right" type="hinge" axis="0 1 0" range="-45 45"/>
        <geom name="geom_wing_right" type="box" pos="-0.0005 -0.001 0" size="0.0008 0.0015 0.00002" material="wing"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Wing flap stroke & pitch actuators -->
    <motor name="act_stroke_left" joint="wing_stroke_left" ctrlrange="-1 1" gear="1e-5"/>
    <motor name="act_pitch_left" joint="wing_pitch_left" ctrlrange="-1 1" gear="1e-5"/>
    <motor name="act_stroke_right" joint="wing_stroke_right" ctrlrange="-1 1" gear="1e-5"/>
    <motor name="act_pitch_right" joint="wing_pitch_right" ctrlrange="-1 1" gear="1e-5"/>
  </actuator>
</mujoco>
"""


class MuJoCoFly:
    """MuJoCo fruit fly environment wrapper with headless rendering support."""

    def __init__(self, headless: bool = True):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not installed or available.")

        if headless and "MUJOCO_GL" not in os.environ:
            # Set headless backend
            os.environ["MUJOCO_GL"] = "egl"

        self.model = mujoco.MjModel.from_xml_string(DROSOPHILA_MJCF)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "compound_eye_cam")

    def reset(self, initial_height: float = 1.0):
        mujoco.mj_resetData(self.model, self.data)
        # Set initial freejoint position
        self.data.qpos[2] = initial_height
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, ctrl: np.ndarray):
        """
        Apply actuator controls and step physics.
        ctrl: [stroke_left, pitch_left, stroke_right, pitch_right]
        """
        ctrl = np.clip(np.asarray(ctrl, dtype=np.float32), -1.0, 1.0)
        if len(ctrl) >= 4:
            self.data.ctrl[:4] = ctrl[:4]
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> dict:
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[0:3].copy()
        return {
            "position": pos,
            "quaternion": quat,
            "velocity": vel,
            "wing_left_pos": float(self.data.qpos[7]) if len(self.data.qpos) > 7 else 0.0,
            "wing_right_pos": float(self.data.qpos[9]) if len(self.data.qpos) > 9 else 0.0,
        }

    def render_eye_view(self, width: int = 32, height: int = 32) -> np.ndarray:
        """Render compound eye visual field from the fly head camera."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)
        self.renderer.update_scene(self.data, camera=self._cam_id)
        rgb = self.renderer.render()
        return rgb
