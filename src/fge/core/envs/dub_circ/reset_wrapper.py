from typing import Optional

import numpy as np

from fge.core.envs.reset_wrapper import ResetWrapper


class DubCircResetWrapper(ResetWrapper):

    def __init__(self, env):
        super().__init__(env)

    def _get_obs(self):
        return self.env._get_obs()

    def _get_info(self):
        info = self.env._get_info()
        info["state"] = self.get_state()
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None and "reset_state" in options:
            self.env.reset(seed=seed, options=options)
            self.set_state(options["reset_state"])

            # Clip the velocities of each vehicle... weird issue?
            self.env.unwrapped.state[3] = np.clip(
                self.env.unwrapped.state[3],
                self.env.unwrapped.task_cfg.ego_min_vel,
                self.env.unwrapped.task_cfg.ego_max_vel,
            )
            for i in range(len(self.env.unwrapped.other_cars)):
                self.env.unwrapped.other_cars[i][3] = np.clip(
                    self.env.unwrapped.other_cars[i][3],
                    self.env.unwrapped.task_cfg.other_min_ang_vel,
                    self.env.unwrapped.task_cfg.other_max_ang_vel,
                )
        else:
            self.env.reset(seed=seed, options=options)
        return self._get_obs(), self._get_info()

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["state"] = self.get_state()
        return obs, reward, term, trunc, info

    def interpret_state(self, state):
        interp = {
            "ego": {
                "x": state[0],
                "y": state[1],
                "theta": state[2],
                "v": state[3],
            },
        }
        for i in range(len(self.env.task_cfg.num_vehicles)):
            interp[f"other_{i}"] = {
                "x": state[4 + i * 4],
                "y": state[5 + i * 4],
                "theta": state[6 + i * 4],
                "v": state[7 + i * 4],
            }
        return interp

    def get_state(self):
        ego_state = self.env.unwrapped.state.copy()
        other_state = []
        for car in self.env.unwrapped.other_cars:
            other_state.extend([car[0], car[1], car[2], car[3]])
        return np.concatenate([ego_state, np.array(other_state)], axis=0)

    @staticmethod
    def get_state_given_info(info):
        return np.array(info["state"])

    def set_state(self, state):
        reset_state = np.array(state)
        self.env.state = reset_state[:4]
        self.env.other_cars = []
        for i in range(0, len(reset_state[4:]), 4):
            self.env.other_cars.append(reset_state[4:][i : i + 4])
        if len(self.env.other_cars) == 0:
            self.env.other_cars = np.array([])
        else:
            self.env.other_cars = np.stack(self.env.other_cars)
