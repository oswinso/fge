from copy import deepcopy
from typing import Optional

import numpy as np
from loguru import logger
from shapely.geometry.point import Point

from fge.core.envs.reset_wrapper import ResetWrapper


class ToylevelsResetWrapper(ResetWrapper):
    """
    A restart wrapper.
    """

    def __init__(self, env):
        super().__init__(env)

    def get_state(self):
        return deepcopy(self.unwrapped._get_info()["agent_pos"])

    @staticmethod
    def get_state_given_info(info):
        return deepcopy(info["agent_pos"])

    def set_state(self, state):
        logger.info(f"Setting state to {state}")
        self.unwrapped.agent = Point(state).buffer(self.unwrapped.task_cfg.agent_radius)

    def _get_obs(self):
        return self.env._get_obs()

    def _get_info(self):
        return self.env._get_info()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = {},
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        if options is not None and "reset_state" in options:
            reset_x, reset_y = options["reset_state"]
            self.unwrapped.agent = Point([reset_x, reset_y]).buffer(
                self.unwrapped.task_cfg.agent_radius
            )
            # Detect what reset region it is in
            self.unwrapped.reset_region = self.unwrapped.which_reset_region(
                reset_x, reset_y
            )
        else:
            if options is not None and "reset_idx" in options:
                reset_idx = options["reset_idx"]
            else:
                # Do nothing
                return obs, info

            reset_options = [
                self.unwrapped.easy_reset_region,
                self.unwrapped.hard_reset_region,
                self.unwrapped.impossible_reset_region,
            ]
            reset_option_names = ["easy", "hard", "impossible"]
            reset_xlb, reset_xub = reset_options[reset_idx]
            reset_x = np.random.uniform(reset_xlb, reset_xub)
            reset_y = (
                self.unwrapped.task_cfg.env_yub
                - 2 * self.unwrapped.task_cfg.agent_radius
            )
            self.unwrapped.agent = Point([reset_x, reset_y]).buffer(
                self.unwrapped.task_cfg.agent_radius
            )

            # Logging
            self.unwrapped.reset_region = reset_option_names[reset_idx]

        self.unwrapped.old_dist = self.unwrapped.goal_box.distance(self.unwrapped.agent)

        return self._get_obs(), self._get_info()
