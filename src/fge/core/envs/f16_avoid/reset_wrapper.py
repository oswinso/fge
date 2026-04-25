from copy import deepcopy
from typing import Optional

import numpy as np

from fge.core.envs.mujoco.hopper.hopper import Hopper
from fge.core.envs.reset_wrapper import ResetWrapper


class F16AvoidResetWrapper(ResetWrapper):
    """
    A restart wrapper for the F16Avoid environment.
    """

    def __init__(self, env):
        super().__init__(env)

        self.all_reset_states = []  # For all time
        self.recent_reset_states = []  # Meant to be cleared after each plotting

    def get_state(self):
        return self.unwrapped._get_info()["state"]

    @staticmethod
    def get_state_given_info(info):
        return deepcopy(info["state"])

    def set_state(self, state):
        self.unwrapped.state = state

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = {}):
        self.unwrapped.timestep = 0
        obs = self.reset_model(reset_state=options.get("reset_state", None))
        return obs, self.unwrapped._get_info()

    def reset_model(self, reset_state=None):
        if reset_state is not None:
            self.set_state(reset_state)
        else:
            self.env.reset()

        obs = self.unwrapped._get_obs()
        info = self.unwrapped._get_info()
        return obs
