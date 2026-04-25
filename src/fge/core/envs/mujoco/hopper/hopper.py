import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import ipdb
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from fge.core.envs.mujoco.mj_plot_utils import CapsuleArtist
from fge.core.envs.mujoco.utils import invtrans2d, transform2d

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 10.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


@dataclass
class TaskCfg:
    # Mujoco configs
    frame_skip: int = 4
    default_camera_config: Dict[str, Union[float, int]] = field(
        default_factory=lambda: DEFAULT_CAMERA_CONFIG
    )
    terminate_when_unhealthy: bool = True

    px0_bounds: Tuple[float, float] = (-1.5, 2.5)

    # Force px to be the same.
    px0: float | None = None
    exclude_current_positions_from_observation: bool = (
        False  # True in original, but not implemented for True.
    )

    # Others
    add_cost_to_reward: bool = False
    max_steps: int = 1000

    eric_gear: bool = False


class Hopper(MujocoEnv, utils.EzPickle):
    """
    Custom Hopper environment based on Gymnasium Hopper-v5
    """

    Cfg = TaskCfg

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, task_cfg: TaskCfg, paths, render_mode=None, **kwargs):
        self.task_cfg = task_cfg
        self.paths = paths
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize MujocoEnv
        xml_file = str(pathlib.Path(__file__).parent.parent / "xmls/hopper.xml")
        self.model_path = xml_file
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=task_cfg.frame_skip,
            observation_space=None,
            default_camera_config=task_cfg.default_camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        self.timestep = 0

        # Observation space
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - self.task_cfg.exclude_current_positions_from_observation
            + 2
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.observation_structure = {
            "skipped_qpos": 1 * task_cfg.exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * task_cfg.exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        self.boundaries = [-1.5, -1.2, -0.1, 0.1, 1.25, 2.3, 2.5]

    @property
    def foot_pose0(self):
        # return np.array([0.0, 0.04, 0.0])

        r_foot = 0.04
        r_floor = 0.1
        r_tot = r_foot + r_floor
        theta = np.arctan(0.8)
        offset = np.array([-r_tot * np.sin(theta), r_tot * np.cos(theta), 0.0])
        return np.array([0.28 + 0.15, 0.15 * 0.8, -np.arctan(0.8)]) + offset

    @property
    def is_healthy(self):
        sdf_torso = 1.0
        sdf_thigh = 1.0

        # contacting floor is unsafe
        id_floor = self.model.geom("floor").id
        id_thigh = self.model.geom("thigh").id
        id_torso = self.model.geom("torso").id
        for contact in self.data.contact:
            if contact.geom1 == id_floor and contact.geom2 == id_thigh:
                sdf_thigh = min(sdf_thigh, contact.dist)
            if contact.geom1 == id_floor and contact.geom2 == id_torso:
                sdf_torso = min(sdf_torso, contact.dist)
        healthy_torso = sdf_torso > 0.0
        healthy_thigh = sdf_thigh > 0.0

        # x position shouldn't be too large
        pos_x = self.data.qpos[0]
        healthy_x = (
            self.task_cfg.healthy_x_range[0]
            <= pos_x
            <= self.task_cfg.healthy_x_range[1]
        )
        is_healthy = healthy_torso and healthy_thigh and healthy_x
        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self.task_cfg.exclude_current_positions_from_observation:
            position = position[1:]

        touch_toe = self.data.sensor("touch_toe").data
        touch_heel = self.data.sensor("touch_heel").data
        touch_obs = np.concatenate([touch_toe, touch_heel], axis=0)
        assert touch_obs.shape == (2,)
        touch_obs = np.log1p(touch_obs)

        observation = np.concatenate((position, velocity, touch_obs)).ravel()
        return observation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.timestep = 0

        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.timestep += 1

        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()

        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
        } | self._get_info()

        reward = 0
        if self.task_cfg.add_cost_to_reward:
            reward -= info["cost"]

        terminated = (not self.is_healthy) and self.task_cfg.terminate_when_unhealthy

        trunc = self.timestep >= self.task_cfg.max_steps

        return observation, reward, terminated, trunc, info

    def set_state_foot(self, foot_pose: np.ndarray, joints: np.ndarray, ankle: float):
        X_W_foot = foot_pose
        X_foot_root = get_X_foot_root(joints[0], joints[1], joints[2], ankle)
        X_W_root = transform2d(X_W_foot, X_foot_root)

        X_rootW_W = np.array([0.0, -1.0, 0.0])
        X_rootW_root = transform2d(X_rootW_W, X_W_root)
        qpos = np.array(
            [
                X_rootW_root[0],
                X_rootW_root[1],
                X_rootW_root[2],
                joints[0],
                joints[1],
                joints[2],
                joints[3] + ankle,
            ]
        )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = np.zeros(self.model.nv)

    def reset_model(self):
        if self.task_cfg.px0 is not None:
            px0 = self.task_cfg.px0
        else:
            px0_lo, px0_hi = self.task_cfg.px0_bounds
            px0 = self.np_random.uniform(px0_lo, px0_hi)

        r_foot = 0.04
        r_floor = 0.1
        height = 0.8 + r_foot + r_floor
        joints0 = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0]))
        self.set_state_foot(np.array([px0, height, 0.0]), joints0, 0.0)

        # Run fwd kinematics to get the pose of everything.
        mujoco.mj_kinematics(self.model, self.data)

        obs = self._get_obs()

        return obs

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "agent_qpos": self.data.qpos.flatten(),
            "agent_qvel": self.data.qvel.flatten(),
        }

    def _get_info(self):
        return {
            "agent_qpos": self.data.qpos.flatten(),
            "agent_qvel": self.data.qvel.flatten(),
            "cost": float(not self.is_healthy),
        }

    def label_ic(self, ax: plt.Axes):
        ax.set_xlabel("px0")

        # Label the sections.
        for x in self.boundaries:
            ax.axvline(x, color="C3", linestyle="--", alpha=0.5)

    def setup_trajplot(self, ax: plt.Axes):
        # Set the limits.
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(-1.8, 2.7)
        ax.set_aspect("equal")

        # Draw the two floor.
        floor_l = CapsuleArtist.fromto(
            0.1, offset=(-0.28, 0.0), fr=(0.0, 0.0), to=(-1.0, 0.8), facecolor="C3"
        )
        floor_r = CapsuleArtist.fromto(
            0.1, offset=(0.28, 0.0), fr=(0.0, 0.0), to=(1.0, 0.8), facecolor="C3"
        )
        floor_r2 = CapsuleArtist.fromto(
            0.1, offset=(0.28, 0.0), fr=(1.0, 0.8), to=(2.0, 0.8), facecolor="C3"
        )
        ax.add_artist(floor_l)
        ax.add_artist(floor_r)
        ax.add_artist(floor_r2)


def get_X_foot_root(
    pelvis: float | np.ndarray,
    thigh: float | np.ndarray,
    leg: float | np.ndarray,
    ankle: float | np.ndarray,
):
    torso_offset = 1.0
    X_W_torso = np.array([0.0, torso_offset, 0.0])

    pelvis_offset = -0.05
    X_torso_pelvis = np.array([0.0, pelvis_offset, pelvis])
    X_W_pelvis = transform2d(X_W_torso, X_torso_pelvis)

    thigh_offset = -0.2
    X_pelvis_thigh = np.array([0.0, thigh_offset, thigh])
    X_W_thigh = transform2d(X_W_pelvis, X_pelvis_thigh)

    calf_offset = -0.33
    X_thigh_calf = np.array([0.0, calf_offset, leg])
    X_W_calf = transform2d(X_W_thigh, X_thigh_calf)

    foot_offset = -0.32
    X_calf_foot = np.array([0.0, foot_offset, ankle])
    X_W_foot = transform2d(X_W_calf, X_calf_foot)

    X_foot_root = transform2d(invtrans2d(X_W_foot), X_W_torso)
    return X_foot_root
