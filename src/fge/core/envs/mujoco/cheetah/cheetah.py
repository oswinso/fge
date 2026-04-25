import pathlib
from dataclasses import dataclass, field
from typing import Dict, Union

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

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
    default_camera_config: Dict[str, Union[float, int]] = field(default_factory=lambda: DEFAULT_CAMERA_CONFIG)
    terminate_when_unhealthy: bool = True

    tgt_height: float = 1.0  # Target height for the cheetah's head

    # px0_bounds: Tuple[float, float] = (-1.5, 2.5)
    # height_bounds: Tuple[float, float] = (0.4, 0.6, 0.8, 1.0)
    # max_height_bounds: Tuple[float, float] = (0.5, 2.0)
    # tgt_height_bounds: Tuple[float, float] = (0.5, 2.0)  # target height: +- 0.1
    # min_height_bounds: Tuple[float, float] = (0.2, 1.5)

    # Force px to be the same.
    px0: float | None = None
    # min_height: float | None = None
    # max_height: float | None = None
    exclude_current_positions_from_observation: bool = True  # True in original, but not implemented for True.

    # Others
    add_cost_to_reward: bool = False
    max_steps: int = 1000


class Cheetah(MujocoEnv, utils.EzPickle):
    """ A task based on the HalfCheetah environment from the DM control suite, but with a different objective.

    The goal here is to keep the cheetah's head above some height (specified by a parameter) for all time.
    The friction coefficients are also specified by a parameter.

    The goal is to be robust to as large a range of heights and friction coefficients as possible.
    """
    Cfg = TaskCfg
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(self, task_cfg: TaskCfg, paths, render_mode=None, **kwargs):
        self.task_cfg = task_cfg
        self.paths = paths
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize MujocoEnv
        xml_file = str(pathlib.Path(__file__).parent.parent / "xmls/cheetah.xml")
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

        # Observation space (qpos + qvel)
        obs_size = self.data.qpos.size + self.data.qvel.size + 1
        # if self.task_cfg.exclude_current_positions_from_observation:
        #     obs_size -= 2  # Remove rootx and rootz
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        self.timestep = 0
        self.boundaries = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

    def reset_model(self):
        if self.task_cfg.height is not None:
            height0 = self.task_cfg.height
        else:
            height0 = self.np_random.uniform(*self.task_cfg.height_bounds)

        # Set initial state
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[1] = height0  # Set torso z-position

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self.task_cfg.exclude_current_positions_from_observation:
            position = position[2:]  # Keep rooty rotation but remove x/z

        return np.concatenate((position, velocity)).ravel()

    @property
    def is_healthy(self):
        head_z = self.data.geom_xpos[self.model.geom("head").id][2]
        healthy_height = head_z > self.task_cfg.min_head_height

        # Check ground contact for body parts
        floor_geom = self.model.geom("ground").id
        body_geoms = [
            self.model.geom("head").id,
            self.model.geom("torso").id,
            self.model.geom("bthigh").id,
            self.model.geom("bshin").id,
            self.model.geom("bfoot").id,
            self.model.geom("fthigh").id,
            self.model.geom("fshin").id,
            self.model.geom("ffoot").id
        ]

        safe_contact = True
        for contact in self.data.contact:
            if contact.geom1 == floor_geom and contact.geom2 in body_geoms:
                safe_contact = False
                break

        return healthy_height and safe_contact

    def step(self, action):
        self.timestep += 1
        x_position_before = self.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        # reward = x_velocity  # Reward is forward velocity
        reward = 0.0
        terminated = not self.is_healthy
        truncated = self.timestep >= self.task_cfg.max_steps

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "head_height": self.data.geom_xpos[self.model.geom("head").id][2],
            "agent_qpos": self.data.qpos.copy(),
            "agent_qvel": self.data.qvel.copy()
        }

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {
            "agent_qpos": self.data.qpos.copy(),
            "agent_qvel": self.data.qvel.copy(),
            "head_height": self.data.geom_xpos[self.model.geom("head").id][2],
            "cost": float(not self.is_healthy),
        }

    def label_ic(self, ax: plt.Axes):
        ax.set_xlabel("Max Force")
        for x in self.boundaries:
            ax.axvline(x, color="C3", linestyle="--", alpha=0.5)
        # for h in (0, 1):
        #     ax.axvline(h, color="C3", linestyle="--", alpha=0.5)

    def setup_trajplot(self, ax: plt.Axes):
        ax.set_ylim(*self.task_cfg.height_bounds)
        ax.set_xlim(-0.5, 2.0)
        ax.set_aspect("auto")

        # Draw ground plane
        ground = plt.Rectangle((-5, -0.5), 10, 0.5,
                               facecolor="C3", alpha=0.3)
        ax.add_patch(ground)
