from typing import Optional

import gymnasium as gym


class ResetWrapper(gym.Wrapper):
    """
    A restart wrapper.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Custom resetting with the options.
        """
        ...

    def get_state(self): ...

    @staticmethod
    def get_state_given_info(info): ...

    def set_state(self, state): ...
