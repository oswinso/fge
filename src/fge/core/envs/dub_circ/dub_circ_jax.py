import functools as ft
import itertools
import math
from typing import Iterable

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from loguru import logger
from og.jax_types import BFloat
from og.rng import PRNGKey

from fge.core.bits.state_reset_id import Source
from fge.core.envs.dub_circ.car import Rectangle, inside_obstacles
from fge.core.envs.dub_circ.dub_circ import DubinsCircularTrackEnv, TaskCfg
from fge.core.envs.jax_task import EvalRegionInfo, EvalStateInfo, JaxTask, State_, StepOutput, TimedState


class DubinsJax(JaxTask):
    Cfg = TaskCfg

    @jdc.pytree_dataclass
    class State(TimedState):
        px0: jnp.ndarray  # Initial conditions for the vehicles
        ego_state: jnp.ndarray  # 4 dimensions: (x, y, theta, v, r)
        o_states: Iterable[jnp.ndarray]  # [(x, y, theta, v, length), ...]  for each other vehicle
        ic: jnp.ndarray

    def __init__(self, task_cfg: TaskCfg):
        # Action space is continuous with 2 dimensions: steering and acceleration
        action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(action_space)
        self.task_cfg = task_cfg
        self.task_cpu = DubinsCircularTrackEnv(task_cfg)

        # Eval info.
        self._eval_rollout_T = 1000

        # Function to create other vehicles
        self.create_rect_vehicle = jax.vmap(ft.partial(Rectangle.create, height=task_cfg.vehicle_radius * 2))

    @property
    def eval_rollout_T(self) -> int:
        return self._eval_rollout_T

    @eval_rollout_T.setter
    def eval_rollout_T(self, value: int):
        self._eval_rollout_T = value

    def _is_finished(self, state: State):
        # Define it as a time-based thing.
        return state.step >= self.task_cfg.max_timesteps

    def _oob(self, state: State):
        """
        Check out of bounds
        """
        x, y, theta, v, rad = state.ego_state
        r = jnp.sqrt(x**2 + y**2)
        inner_r = self.task_cpu.track_inner_radius
        outer_r = self.task_cpu.track_outer_radius
        # rad = self.task_cfg.vehicle_radius
        return jnp.logical_or(r < inner_r + rad, r > outer_r - rad)

    def _check_collide(self, state: State):
        """
        Check collision with other cars
        """
        if self.task_cfg.num_vehicles == 0:
            return False

        rad = state.ego_state[4]
        e_state = state.ego_state
        o_states = state.o_states

        # create rectangles for other vehicles
        o_vehicles = self.create_rect_vehicle(o_states[:, :2], width=o_states[:, 4], theta=o_states[:, 3])
        collision = inside_obstacles(e_state[:2], o_vehicles, rad)

        return collision

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        x, y, theta, v, r = state.ego_state
        steering, acc = action

        # Ego agent update using regular velocity
        turning_radius = (self.task_cpu.track_inner_radius + self.task_cpu.track_outer_radius) / 2
        max_turn_rate = v / turning_radius
        max_turn_rate *= 2  # Otherwise too small
        omega = steering * max_turn_rate
        v = jnp.clip(v + acc * self.task_cfg.dt, self.task_cfg.ego_min_vel, self.task_cfg.ego_max_vel)
        theta += omega * self.task_cfg.dt
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)  # Normalize to [0, 2π)
        x += v * jnp.cos(theta) * self.task_cfg.dt
        y += v * jnp.sin(theta) * self.task_cfg.dt
        e_state = (x, y, theta, v, r)
        assert len(state.o_states) == self.task_cfg.num_vehicles

        # Other cars update
        o_states = []
        for i, (x_o, y_o, theta_o, v_o, l_o) in enumerate(state.o_states):
            # Angular velocity
            r_o = jnp.sqrt(x_o**2 + y_o**2)
            angle_o = jnp.arctan2(y_o, x_o)
            angle_o = (angle_o + 2 * jnp.pi) % (2 * jnp.pi)
            angle_o = (angle_o + v_o * self.task_cfg.dt) % (2 * jnp.pi)
            x_o = r_o * jnp.cos(angle_o)
            y_o = r_o * jnp.sin(angle_o)
            theta_o = (angle_o + jnp.pi / 2) % (2 * jnp.pi)
            o_states.append((x_o, y_o, theta_o, v_o, l_o))

        o_states = jnp.array(o_states)
        assert o_states.shape == (self.task_cfg.num_vehicles, 5)

        state_new = jdc.replace(state, step=state.step + 1, ego_state=jnp.array(e_state), o_states=o_states)

        obs = self.get_obs(state_new)
        collided = self._check_collide(state_new)
        oob = self._oob(state_new)
        term = jnp.logical_or(collided, oob)
        trunc = self._is_finished(state_new)
        rew = jnp.where(term, -1.0, 0.0)
        info = {
            'collided': collided,
            'ic': state_new.ic,
        }

        return StepOutput(state=state_new, obs=obs, rew=rew, term=term, trunc=trunc, info=info)

    def get_obs(self, state: State_):
        obs = []
        x, y, theta, v, radius = state.ego_state

        r = jnp.sqrt(x**2 + y**2)
        # Get r in [0,1] where 0 is the inner track and 1 is the outer track
        norm_r = (r - self.task_cpu.track_inner_radius) / (
            self.task_cpu.track_outer_radius - self.task_cpu.track_inner_radius
        )

        # Compute ego's global angle on track
        angle = jnp.arctan2(y, x)
        angle = (angle + 2 * jnp.pi) % (2 * jnp.pi)

        # Normalize the velocity between [0, 1] where 1 is the max velocity and 0 is the min velocity
        v = self.task_cpu.reg_to_ang_v(v, r)
        e_ang_lo, e_ang_hi = self.task_cpu.reg_to_ang_v(self.task_cfg.ego_min_vel, r), self.task_cpu.reg_to_ang_v(
            self.task_cfg.ego_max_vel, r
        )
        # norm_v = (v - self.task_cfg.ego_min_vel) / (self.task_cfg.ego_max_vel - self.task_cfg.ego_min_vel)
        norm_v = (v - e_ang_lo) / (e_ang_hi - e_ang_lo)

        # Get the theta relative to the tangent of the track
        tangent_vector = jnp.array([-y, x])
        norm_tangent_vector = tangent_vector / jnp.linalg.norm(tangent_vector)
        tangent_angle = jnp.arctan2(norm_tangent_vector[1], norm_tangent_vector[0])
        tangent_angle = (tangent_angle + 2 * jnp.pi) % (2 * jnp.pi)

        # Get relative theta based on cos and sin
        rel_cos_theta = jnp.cos(tangent_angle - theta)
        rel_sin_theta = jnp.sin(tangent_angle - theta)

        obs.append(rel_cos_theta)
        obs.append(rel_sin_theta)
        obs.append(norm_r)
        obs.append(norm_v)

        for o_state in state.o_states:
            x_o, y_o, theta_o, v_o, l_o = o_state
            r_o = jnp.sqrt(x_o**2 + y_o**2)
            norm_r_o = (r_o - self.task_cpu.track_inner_radius) / (
                self.task_cpu.track_outer_radius - self.task_cpu.track_inner_radius
            )

            angle_o = jnp.arctan2(y_o, x_o)
            angle_o = (angle_o + 2 * jnp.pi) % (2 * jnp.pi)

            rel_angle = angle_o - angle
            cos_o = jnp.cos(rel_angle)
            sin_o = jnp.sin(rel_angle)

            norm_v_o = (v_o - self.task_cfg.other_min_ang_vel) / (
                self.task_cfg.other_max_ang_vel - self.task_cfg.other_min_ang_vel
            )

            rel_norm_r_o = norm_r_o - norm_r
            rel_norm_v = norm_v_o - norm_v  # Slightly inaccurate, since norm_v is in regular velocity.

            obs.append(cos_o)
            obs.append(sin_o)
            obs.append(rel_norm_r_o)
            obs.append(rel_norm_v)
            obs.append(l_o)  # Length of the other vehicle

        return jnp.array(obs)

    @ft.cache
    def get_dummy_obs(self):
        state = self.reset(jr.PRNGKey(1234))
        obs = self.get_obs(state)
        return np.array(obs)

    def _ego_s0(self):
        angle = 0  # Ego starts on right side
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        lane = 0  # Ego starts in the inner lane
        x = self.task_cpu.lane_radii[lane] * np.cos(angle)
        y = self.task_cpu.lane_radii[lane] * np.sin(angle)

        theta = (angle + np.pi / 2) % (2 * np.pi)
        v = (self.task_cfg.ego_min_vel + self.task_cfg.ego_max_vel) / 2  # Midpoint
        return jnp.array([x, y, theta, v, self.task_cfg.vehicle_radius])

    def _o_s0(self, pv0: Iterable[float]):
        assert len(pv0) == self.task_cfg.num_vehicles * 2  # velocity + length
        cars = []
        for i in range(self.task_cfg.num_vehicles):
            lane = i % self.task_cfg.num_lanes  # Each additional car is one lane over
            angle = jnp.pi + i * (jnp.pi / 8)
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            x = self.task_cpu.lane_radii[lane] * np.cos(angle)
            y = self.task_cpu.lane_radii[lane] * np.sin(angle)
            theta = (angle + np.pi / 2) % (2 * np.pi)
            v = pv0[i]
            ll = pv0[i + self.task_cfg.num_vehicles]  # Length of the vehicle
            cars.append([x, y, theta, v, ll])
        out = jnp.array(cars)
        assert out.shape == (self.task_cfg.num_vehicles, 5)
        return out

    def state0_from_pv(self, pv: Iterable[float]):
        ego_state = self._ego_s0()
        o_states = self._o_s0(pv0=pv)

        state = DubinsJax.State(step=0, source=0, ego_state=ego_state, o_states=o_states, px0=pv, ic=pv.copy())
        return state

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State:
        del b_key

        def f(pv):
            assert pv.shape == (4,)
            state = self.state0_from_pv(pv)
            state = jdc.replace(state, source=Source.BASE)
            return state

        key = jr.PRNGKey(54321)
        pv0_lo, pv0_hi = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
        l_lo, l_hi = self.task_cfg.o_vehicle_len_range
        lo_lim = jnp.array([pv0_lo, pv0_lo, l_lo, l_lo])
        hi_lim = jnp.array([pv0_hi, pv0_hi, l_hi, l_hi])
        b_pv = jr.uniform(key, minval=lo_lim, maxval=hi_lim, shape=(num, self.task_cfg.num_vehicles * 2))
        return jax.vmap(f)(b_pv)

    def reset(self, key: jr.PRNGKey) -> State:
        if self.task_cfg.pv0 is None:
            pv0_lo, pv0_hi = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
            l_lo, l_hi = self.task_cfg.o_vehicle_len_range
            lo_lim = jnp.array([pv0_lo, pv0_lo, l_lo, l_lo])
            hi_lim = jnp.array([pv0_hi, pv0_hi, l_hi, l_hi])
            match self.task_cfg.fix_other_vel:
                case "normal":
                    # pv0 = jr.uniform(key, minval=pv0_lo, maxval=pv0_hi, shape=(self.task_cfg.num_vehicles,))
                    pv0 = jr.uniform(key, minval=lo_lim, maxval=hi_lim, shape=(self.task_cfg.num_vehicles * 2,))
                case "min":
                    pv0 = jnp.array([lo_lim for _ in range(self.task_cfg.num_vehicles)])
                case "max":
                    pv0 = jnp.array([hi_lim for _ in range(self.task_cfg.num_vehicles)])
                case "mid":
                    pv0 = jnp.array([(lo_lim + hi_lim) / 2 for _ in range(self.task_cfg.num_vehicles)])
                case _:
                    raise NotImplementedError(f"Unknown fix_other_vel: {self.task_cfg.fix_other_vel}")
        else:
            pv0 = self.task_cfg.pv0

        self.task_cpu.timestep = 0
        state = self.state0_from_pv(pv0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def x0_from_box(self, uniform: jnp.ndarray) -> State:
        pv0 = self.pv0_from_box(uniform)
        state = self.state0_from_pv(pv0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def box_from_x0(self, x0: State):
        pv0 = x0.px0
        return self.box_from_pv0(pv0)

    def box_from_pv0(self, x0: jnp.ndarray):
        # Map from [x0_lo, x0_hi] to [-1, 1]^n_vehicle
        # x0.shape = (batch_size, num_vehicles)

        pv0_lo, pv0_hi = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
        l_lo, l_hi = self.task_cfg.o_vehicle_len_range
        lo_lim, hi_lim = jnp.array([pv0_lo, pv0_lo, l_lo, l_lo]), jnp.array([pv0_hi, pv0_hi, l_hi, l_hi])

        assert x0.shape[-1] == 2 * self.task_cfg.num_vehicles
        uniform = 2 * (x0 - hi_lim) / (hi_lim - lo_lim) - 1
        assert uniform.shape == x0.shape
        return uniform

    def pv0_from_box(self, uniform: jnp.ndarray) -> jnp.ndarray:
        # Map from [-1, 1]^n_vehicle to [pv0_lo, pv0_hi]
        assert uniform.shape == (self.task_cfg.num_vehicles * 2,)
        pv0_lo, pv0_hi = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
        l_lo, l_hi = self.task_cfg.o_vehicle_len_range
        lo_lim, hi_lim = jnp.array([pv0_lo, pv0_lo, l_lo, l_lo]), jnp.array([pv0_hi, pv0_hi, l_hi, l_hi])
        # pv0 = (uniform + 1) / 2 * (pv0_hi - pv0_lo) + pv0_lo
        pv0 = (uniform + 1) / 2 * (hi_lim - lo_lim) + lo_lim
        return pv0

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        """Interface for allowing adversarial optimization over the initial state. [-1, 1]^n to state."""
        pv0 = self.pv0_from_box(uniform)
        state = self.state0_from_pv(pv0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def box_from_reset(self, state: State):
        return self.box_from_x0(state)

    def to_icval(self, state: State) -> np.ndarray:
        """Project the state to the line / plane / surface for visualizing the initial conditions."""
        return state.px0

    def icval_bins(self, n_bins: int = 31) -> np.ndarray:
        ...

    def get_eval_contour(self):
        n_per_region = 2
        b_state, _, _ = self.get_eval_states(n_per_region)

        b_px0 = b_state.px0
        b_obs = jax.vmap(self.get_obs)(b_state)

        return b_px0, b_obs, b_state

    def generate_initial_conditions(
        self, o_minv: float, o_maxv: float, o_minl: float, o_maxl: float, n_per_region: int, num_cars: int
    ) -> tuple[list[EvalRegionInfo], list[list[float]]]:
        linspaces = [np.linspace(o_minv, o_maxv, n_per_region) for _ in range(num_cars)] + [
            np.linspace(o_minl, o_maxl, n_per_region) for _ in range(num_cars)
        ]

        regions: list[EvalRegionInfo] = []
        b_px0 = []

        for fixed_vals in itertools.product(*linspaces[:-1]):
            # This iterates through linspaces[0]

            for last_val in linspaces[-1]:
                # This iterates through linspaces[1]
                full_comb = list(fixed_vals) + [last_val]
                b_px0.append(full_comb)
            label = "|".join([f"Car{i + 1}={v:.4f}" for i, v in enumerate(fixed_vals)])
            regions.append(EvalRegionInfo(label, n_per_region))

        return regions, b_px0

    @property
    def region_names(self) -> list[str]:
        """Names of the regions for the eval states."""
        o_minv, o_maxv = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
        o_minl, o_maxl = self.task_cfg.o_vehicle_len_range
        n_vehicles = self.task_cfg.num_vehicles
        regions, _ = self.generate_initial_conditions(
            o_minv=o_minv, o_maxv=o_maxv, o_minl=o_minl, o_maxl=o_maxl, n_per_region=2, num_cars=n_vehicles
        )
        regions = [region.name for region in regions]
        return regions

    def eval_xs_(self, n_per_region: int = 2) -> tuple[BFloat, list[EvalRegionInfo]]:
        o_minv, o_maxv = self.task_cfg.other_min_ang_vel, self.task_cfg.other_max_ang_vel
        o_minl, o_maxl = self.task_cfg.o_vehicle_len_range
        e_minv = self.task_cfg.ego_min_vel

        n_vehicles = self.task_cfg.num_vehicles

        if n_vehicles == 0:
            regions = [("EgoCar", 1)]
            b_px0 = np.array([e_minv])
        else:
            regions, b_px0 = self.generate_initial_conditions(
                o_minv=o_minv,
                o_maxv=o_maxv,
                o_minl=o_minl,
                o_maxl=o_maxl,
                n_per_region=n_per_region,
                num_cars=n_vehicles,
            )
            b_px0 = np.stack(b_px0)
            assert b_px0.shape == (n_per_region ** (n_vehicles * 2), n_vehicles * 2)

        return b_px0, regions

    def eval_xs(self, n_per_region: int = 2):
        """Some 1D representation of the eval states."""
        b_px0, regions = self.eval_xs_(n_per_region)
        return b_px0

    def get_eval_states(self, n_per_region: int = 2) -> EvalStateInfo:
        b_pv0, regions = self.eval_xs_(n_per_region)
        n_reset = n_per_region ** (self.task_cfg.num_vehicles * 2)
        b_state = jax.vmap(self.state0_from_pv)(b_pv0)
        return EvalStateInfo(b_state, n_reset, regions)

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        """Shape of uniform in reset_from_box."""
        return (4,)

    def get_contour_grid(self, vals: np.ndarray):
        _b_px0, _, _ = self.get_eval_contour()
        x_unique = np.unique(_b_px0[:, 0])
        y_unique = np.unique(_b_px0[:, 1])
        nx = len(x_unique)
        ny = len(y_unique)
        assert nx * ny == len(vals)
        vals_grid = vals.reshape((nx, ny))
        return vals_grid

    def _draw_track(self, ax: plt.Axes):
        track_center = self.task_cfg.track_center
        opts = dict(color="black", fill=False, linewidth=2, zorder=1)
        inner = plt.Circle(track_center, self.task_cfg.track_inner_radius, **opts)
        outer = plt.Circle(track_center, self.task_cfg.track_outer_radius, **opts)
        ax.add_patch(inner)
        ax.add_patch(outer)

        for i in range(1, self.task_cfg.num_lanes):
            radius = self.task_cfg.track_radius + i * self.task_cfg.lane_width
            circ = plt.Circle(track_center, radius, color="gray", ls="--", linewidth=1, zorder=1, fill=False)
            ax.add_patch(circ)

    def setup_trajplot(self, ax: plt.Axes):
        ax.set_aspect("equal")
        lim = self.task_cfg.track_outer_radius + self.task_cfg.lane_width
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        self._draw_track(ax)
