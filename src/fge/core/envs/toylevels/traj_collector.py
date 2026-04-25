from copy import deepcopy

import gymnasium as gym
import numpy as np
from loguru import logger


class ToylevelsTrajectoryCollector(gym.Wrapper):
    def __init__(self, env, env_idx=None):
        super().__init__(env)
        self.env_idx = env_idx

        self._reset_poses = []
        self._curr_traj = np.empty(
            (
                self.unwrapped.task_cfg.max_steps + 1,
                self.unwrapped.observation_space.shape[0],
            )
        )
        # self._all_trajs = []
        self._recent_trajs = []

    def _get_obs(self):
        return self.env._get_obs()

    def _get_info(self):
        return self.env._get_info()

    @property
    def curr_traj(self):
        return self._curr_traj

    @property
    def recent_trajs(self):
        return self._recent_trajs

    # @property
    # def all_trajs(self):
    #     return self._all_trajs
    #
    # @all_trajs.setter
    # def all_trajs(self, value):
    #     if not isinstance(value, list):
    #         raise ValueError("all_trajs must be a list")
    #     self._all_trajs = value

    @recent_trajs.setter
    def recent_trajs(self, value):
        if not isinstance(value, list):
            raise ValueError("recent_trajs must be a list")
        self._recent_trajs = value

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._curr_traj[info["timestep"]] = info["agent_pos"]
        self._reset_poses.append(deepcopy(info["agent_pos"]))
        return obs, info

    def step(self, action):
        obs, reward, trunc, term, info = self.env.step(action)
        self._curr_traj[info["timestep"] - 1] = info["agent_pos"]
        if trunc or term:
            # self._all_trajs.append(deepcopy(self._curr_traj)[:info['timestep']])
            self._recent_trajs.append(deepcopy(self._curr_traj)[: info["timestep"]])
            self._curr_traj = np.empty(
                (
                    self.unwrapped.task_cfg.max_steps + 1,
                    self.unwrapped.observation_space.shape[0],
                )
            )

        return obs, reward, trunc, term, info

    def consume_recent_trajs(self):
        """
        Consume the recent trajectories and reset the recent trajectories list.
        """
        trajs = self._recent_trajs
        self._recent_trajs = []
        return trajs

    def close(self):
        super().close()
