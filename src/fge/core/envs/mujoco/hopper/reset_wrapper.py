from copy import deepcopy
from typing import Optional

import mujoco
import numpy as np
from loguru import logger

from fge.core.envs.mujoco.hopper.hopper import Hopper
from fge.core.envs.reset_wrapper import ResetWrapper


class HopperResetWrapper(ResetWrapper):
    """
    A restart wrapper for the Hopper environment.
    """

    def __init__(self, env):
        super().__init__(env)

        self.all_reset_states = []  # For all time
        self.recent_reset_states = []  # Meant to be cleared after each plotting

    def get_state(self):
        agent_pos = deepcopy(self.unwrapped._get_info()["agent_qpos"])
        agent_vel = deepcopy(self.unwrapped._get_info()["agent_qvel"])
        return np.concatenate((agent_pos, agent_vel))

    @staticmethod
    def get_state_given_info(info):
        return deepcopy(np.concatenate([info["agent_qpos"], info["agent_qvel"]]))

    def set_state(self, state):
        qpos = state[: self.unwrapped.model.nq]
        qvel = state[
            self.unwrapped.model.nq : self.unwrapped.model.nq + self.unwrapped.model.nv
        ]
        self.unwrapped.set_state(qpos, qvel)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = {},
    ):
        self.unwrapped.timestep = 0
        self.env.reset(seed=seed, options=options)

        mujoco.mj_resetData(self.unwrapped.model, self.unwrapped.data)

        obs = self.reset_model(
            reset_state=options.get("reset_state", None),
            ankle=options.get("ankle", None),
        )
        info = self.unwrapped._get_reset_info()

        # Logging
        # logger.debug(f'{self.get_state()=}')
        # import pdb; pdb.set_trace()
        self.all_reset_states.append(deepcopy(self.get_state()))
        self.recent_reset_states.append(deepcopy(self.get_state()))

        return obs, info

    def consume_recent_reset_states(self):
        recent_reset_states = deepcopy(self.recent_reset_states)
        self.recent_reset_states = []
        return recent_reset_states

    def reset_model(self, reset_state=None, ankle=None):
        unwrapped: Hopper = self.unwrapped

        if reset_state is not None or ankle is not None:
            assert not (
                reset_state is not None and ankle is not None
            ), "Cannot set both reset_state and ankle"
            if reset_state is not None:
                self.set_state(reset_state)
            elif ankle is not None:
                joints0 = np.zeros(4)
                unwrapped.set_state_foot(unwrapped.foot_pose0, joints0, ankle)
                unwrapped.energy = unwrapped.task_cfg.energy0
        else:
            if unwrapped.task_cfg.ankle0 is not None:
                ankle = unwrapped.task_cfg.ankle0
            else:
                ankle_lo, ankle_hi = unwrapped.task_cfg.ankle_state_bounds
                ankle = self.np_random.normal(
                    loc=0.0, scale=np.clip(ankle_hi / 4, 0, 5)
                )

            joints0 = np.zeros(4)
            unwrapped.set_state_foot(unwrapped.foot_pose0, joints0, ankle)
            unwrapped.energy = unwrapped.task_cfg.energy0

        # Run fwd kinematics to get the pose of everything.
        mujoco.mj_kinematics(unwrapped.model, unwrapped.data)

        obs = self.env._get_obs()

        return obs
