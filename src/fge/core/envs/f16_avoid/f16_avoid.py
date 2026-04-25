from dataclasses import dataclass, field

import gymnasium as gym
import jax
import jax.random as jr
import numpy as np
from gymnasium.spaces import Box

from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax, TaskCfg


class F16Avoid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, task_cfg: TaskCfg, paths, render_mode=None, **kwargs):
        self.task_cfg = task_cfg
        self.paths = paths
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # self.task_jax = jax.device_put(F16AvoidJax(task_cfg), jax.devices("cpu")[0])
        self.task_jax = F16AvoidJax(task_cfg)
        self.step_jax_jit = jax.jit(self.task_jax.step)

        self.observation_space = Box(
            low=np.array([-np.inf] * 20), high=np.array([np.inf] * 20)
        )
        self.action_space = self.task_jax.action_space

        self.rng = np.random.default_rng()
        self.state = None
        self.timestep = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep = 0
        self.state = self.task_jax.reset(
            key=jr.PRNGKey(self.rng.integers(0, 2**32 - 1))
        )
        return self.task_jax.get_obs(self.state), self._get_info()

    def step(self, action):
        self.timestep += 1
        # step_out = self.task_jax.step(self.state, action)
        step_out = self.step_jax_jit(self.state, action)
        self.state = step_out.state

        return (
            step_out.obs,
            np.array(step_out.rew),
            step_out.term,
            self.timestep >= self.task_cfg.max_steps,
            self._get_info(),
        )

    def _get_info(self):
        return {"state": self.state}

    def _get_obs(self):
        return self.task_jax.get_obs(self.state)
