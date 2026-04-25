import functools as ft
import itertools
import pathlib
from typing import NamedTuple

import ipdb
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_dataclasses as jdc
import mujoco
import numpy as np
from gymnasium import spaces
from jax import random as jr
from loguru import logger
from matplotlib import pyplot as plt
from mujoco import mjx
from mujoco.mjx._src.collision_primitive import capsule_capsule
from og.dyn_types import QpFloat, QvFloat
from og.jax_types import BBool, BFloat, FloatScalar, IntScalar
from og.jax_utils import jax_vmap

from fge.core.bits.state_reset_id import Source
from fge.core.envs.jax_task import JaxTask, State_, StepOutput, TimedState
from fge.core.envs.mujoco.cheetah.cheetah import Cheetah, TaskCfg
from fge.core.envs.mujoco.utils import invtrans2d_jax, transform2d, transform2d_jax


class CheetahJax(JaxTask):
    # Cfg = TaskCfg

    @jdc.pytree_dataclass
    class State(TimedState):
        px0: FloatScalar  # (0, 1), force on the floating base
        reset_region: IntScalar
        data: mjx.Data

    @jdc.pytree_dataclass
    class MinState(TimedState):
        px0: FloatScalar
        reset_region: IntScalar
        qpos: jnp.ndarray
        qvel: jnp.ndarray

    def __init__(self, task_cfg: TaskCfg):
        self.task_cfg = task_cfg
        self.task_cpu = Cheetah(task_cfg, paths=None)

        self.model_path = pathlib.Path(__file__).parent.parent / "xmls/cheetah.xml"
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(self.model_path))

        if jax.config.jax_default_matmul_precision != "highest":
            raise ValueError("Jax matmul precision must be set to highest, or we get NaNs...")

        bounds = self._model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        # remove the ground height control
        low = low[1:]
        high = high[1:]
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        super().__init__(action_space)

        cpu_device = jax.devices("cpu")[0]
        self.mjx_model = mjx.put_model(self.model, device=cpu_device)

        # Eval info
        self.n_eval_regions = 6
        separators = np.linspace(0, 1, num=self.n_eval_regions + 1)
        self.eval_regions = np.stack([separators[:-1], separators[1:]], axis=1)

    @property
    def model(self):
        return self._model

    @property
    def eval_rollout_T(self) -> int:
        return 1000

    # @eval_rollout_T.setter
    # def eval_rollout_T(self, T: int):
    #     assert T > 0
    #     self._eval_rollout_T = T

    def get_obs(self, state: State):
        data = state.data
        qpos = data.qpos
        qvel = data.qvel
        # jax.debug.breakpoint()
        return jnp.concatenate([qpos, qvel, self.step2ground_action(state.step, state.px0)[None]])

    def _get_collided(self, state: State):
        data = state.data
        # Height-based collision
        head_geom_id = self.model.geom("head").id
        head_z = data.geom_xpos[head_geom_id][2]
        height_healthy = (head_z >= self.task_cfg.tgt_height - 0.42) & (head_z <= self.task_cfg.tgt_height + 0.42)

        # Floor contact check
        floor_geom_id = self.model.geom("ground").id
        body_geoms = [
            self.model.geom("head").id,
            self.model.geom("torso").id,
            self.model.geom("bthigh").id,
            # self.model.geom("bshin").id,
            # self.model.geom("bfoot").id,
            self.model.geom("fthigh").id,
            # self.model.geom("fshin").id,
            # self.model.geom("ffoot").id
        ]
        # Check contacts between body parts and ground
        geoms = jnp.array([[floor_geom_id, gid] for gid in body_geoms])
        dists, _, _ = capsule_capsule(self.mjx_model, data, None, geoms)
        contact_healthy = jnp.all(dists > 0.0)

        is_healthy = height_healthy & contact_healthy

        return ~is_healthy

    def _is_finished(self, state: State):
        return state.step >= self.task_cfg.max_steps

    def step_mujoco_simulation(self, state: State, action: jnp.ndarray):
        assert action.shape == (self.model.nu,)
        assert state.data.ctrl.shape == action.shape
        data_new = state.data.replace(ctrl=action)
        data_new = self.mj_step(data_new, n_frames=self.task_cpu.frame_skip)
        return data_new

    def mj_step(self, data: mjx.Data, n_frames: int):
        def f(state: mjx.Data, _):
            return mjx.step(self.mjx_model, state), None
        data_new, _ = lax.scan(f, data, (), length=n_frames)
        return data_new

    def step2ground_action(self, step: IntScalar, px0) -> jnp.ndarray:
        """
        Convert step to ground action
        interval 0, 3, 5, 6 -> -px0
        interval 1, 2, 4, 7 -> px0
        """
        interval = self.eval_rollout_T // 8
        i_interval = step // interval
        i_interval = jnp.clip(i_interval, 0, 7)

        ground_action = jnp.where(jnp.isin(i_interval, jnp.array([0, 3, 5, 6])), -px0, px0) + 0.11
        # ground_action = jnp.zeros(())
        return ground_action

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        action_all = jnp.concatenate([action, self.step2ground_action(state.step, state.px0)[None]])
        data_new = self.step_mujoco_simulation(state, action_all)
        collided = self._get_collided(state)
        reward = jnp.where(collided, -1.0, 0.0)
        terminal = collided
        truncated = self._is_finished(state)
        info = {}
        state_new = jdc.replace(state, step=state.step + 1, data=data_new)
        obs = self.get_obs(state_new)
        # jax.debug.breakpoint()
        return StepOutput(state_new, obs, reward, terminal, truncated, info)

    def get_minstate(self, state: State) -> MinState:
        qpos = state.data.qpos
        qvel = state.data.qvel
        return CheetahJax.MinState(
            step=state.step,
            source=state.source,
            px0=state.px0,
            reset_region=state.reset_region,
            qpos=qpos,
            qvel=qvel
        )

    def from_minstate(self, minstate: MinState) -> State:
        data = self.pipeline_init(minstate.qpos, minstate.qvel)
        return CheetahJax.State(
            step=minstate.step,
            source=minstate.source,
            px0=minstate.px0,
            reset_region=minstate.reset_region,
            data=data
        )

    @ft.cache
    def _minstate_treedef(self) -> jtu.PyTreeDef:
        minstate = CheetahJax.MinState(0, 0, 0, 0, 0, 0)
        return jtu.tree_structure(minstate)

    @property
    def minstate_treedef(self):
        return self._minstate_treedef()

    def pipeline_init(self, q: QpFloat, qd: QvFloat) -> mjx.Data:
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=q, qvel=qd)
        data = mjx.forward(self.mjx_model, data)
        data = data.replace(qacc_warmstart=data.qacc_warmstart.copy())
        return data

    # def get_reset_region(self, height: float):
    #     #Analogous to Hopper's px-based regions but using height
    #     boundaries = self.task_cfg.tgt_height_bounds
    #     idx = jnp.searchsorted(jnp.array(boundaries), height, side="right") - 1
    #     return jnp.clip(idx, 0, len(boundaries) - 2)

    # def state0_from_height(self, height: float):
    #     qpos = jnp.zeros(self.model.nq).at[1].set(height)
    #     qvel = jnp.zeros(self.model.nv)
    #     data = self.pipeline_init(qpos, qvel)
    #     reset_region = self.get_reset_region(height)
    #     return CheetahJax.State(0, Source.BASE, height, reset_region, data)

    def state0_from_px0(self, px0: float):
        qpos = jnp.array(self.task_cpu.init_qpos.copy())
        qvel = jnp.array(self.task_cpu.init_qvel.copy())
        data = self.pipeline_init(qpos, qvel)
        state = CheetahJax.State(0, 0, px0, 0, data)
        return state

    def reset(self, key: jr.PRNGKey) -> State:
        # if self.task_cfg.height is None:
        #     lo, hi = self.task_cfg.height_bounds[0], self.task_cfg.height_bounds[-1]
        #     height = jr.uniform(key, minval=lo, maxval=hi)
        # else:
        #     height = self.task_cfg.height
        if self.task_cfg.px0 is None:
            px0 = jr.uniform(key, minval=0.0, maxval=1.0)
        else:
            px0 = self.task_cfg.px0

        state = self.state0_from_px0(px0)
        state = jdc.replace(state, source=Source.BASE)
        # jax.debug.breakpoint()
        return state

    # @property
    # def n_eval_regions(self):
    # #     return len(self.task_cfg.height_bounds) - 1
    #
    # @property
    # def region_names(self):
    #     return ["Low", "Medium", "High"]  # Simple height-based regions

    @property
    def region_names(self):
        furthest_bound = [b[0] if b[0] < 0 else b[1] for b in self.eval_regions]
        return ["f{:.2f}".format(b) for b in furthest_bound]

    def get_eval_states(self, n_per_region: int = 10) -> tuple[State, int, list[tuple[str, int]]]:
        regions: list[tuple[str, int]] = []

        bounds = [0, 1]
        # Have ankle0 be evenly spaced in ankle_state_bounds.
        n_reset = n_per_region * self.n_eval_regions
        b_f0 = np.linspace(bounds[0], bounds[1], num=n_reset)

        for name, region in zip(self.region_names, self.eval_regions):
            regions.append((name, n_per_region))
            # Get the ankle0 within the region.
            b_f0_region = b_f0[(b_f0 >= region[0]) & (b_f0 <= region[1])]
            assert len(b_f0_region) == n_per_region, f"Expected {n_per_region} but got {len(b_f0_region)}"

        b_state = jax.vmap(self.state0_from_px0)(b_f0)
        return b_state, n_reset, regions

        #
        # b_height, regions = self.eval_xs_(n_per_region)
        # n_reset = n_per_region * self.n_eval_regions
        # b_state = jax.vmap(self.state0_from_height)(b_height)
        #
        # return b_state, n_reset, regions

    def get_eval_contour(self):
        n_per_region = 40
        b_state, _, _ = self.get_eval_states(n_per_region)

        b_f = b_state.px0
        b_obs = jax.vmap(self.get_obs)(b_state)

        return b_f, b_obs, b_state
    #
    # def eval_xs_(self, n_per_region: int = 10):
    #     """Generate height samples within each region"""
    #     regions: list[tuple[str, int]] = []
    #     b_height_all = []
    #
    #     boundaries = self.task_cfg.height_bounds
    #     for ii, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    #         region_name = self.region_names[ii]
    #         regions.append((region_name, n_per_region))
    #
    #         # Include edges for first/last regions
    #         if ii == 0 or ii == len(boundaries) - 2:
    #             b_height = np.linspace(lo, hi, n_per_region)
    #         else:
    #             b_height = np.linspace(lo, hi, n_per_region + 2)[1:-1]
    #
    #         b_height_all.append(b_height)
    #
    #     return np.concatenate(b_height_all), regions

    def eval_xs(self, n_per_region: int = 10):
        bounds = [0, 1]
        n_reset = n_per_region * self.n_eval_regions
        b_px0 = np.linspace(bounds[0], bounds[1], num=n_reset)

        # b_height, _ = self.eval_xs_(n_per_region)
        return b_px0

    def to_icval(self, state: State | MinState) -> np.ndarray:
        return state.px0

    def eval_ics(self, n_per_region: int = 10) -> BFloat:
        b_px0 = self.eval_xs(n_per_region)
        return b_px0

    def label_ic(self, ax: plt.Axes):
        return self.task_cpu.label_ic(ax)

    @property
    def icval_bounds(self):
        # return self.task_cfg.height_bounds
        return (0., 1.)

    def icval_bins(self, n_bins: int = 31):
        """The boundaries to use for a histogram. Try and have equal bin widths, but make sure the bin edges are also the boundaries."""
        # Try to get this bin width in each region.
        bin_width_tgt = (self.icval_bounds[1] - self.icval_bounds[0]) / n_bins

        bin_edges = [self.icval_bounds[0]]
        for region in self.eval_regions:
            # Figure out how many bins to reach bin_width_tgt. Err on the side of more bins.
            n_bins_region = int(np.ceil((region[1] - region[0]) / bin_width_tgt))

            # Make sure we have at least 1 bin.
            n_bins_region = max(n_bins_region, 1)

            edges = np.linspace(region[0], region[1], n_bins_region + 1)[1:]
            bin_edges.extend(edges)

        bin_edges = np.array(bin_edges)
        return bin_edges

    def box_from_reset(self, state: State):
        px0 = state.px0
        # Change [px0_lo, px0_hi] -> [-1, 1]
        px0_lo, px0_hi = self.icval_bounds
        box = (px0 - px0_lo) / (px0_hi - px0_lo) * 2.0 - 1.0
        return jnp.array([box])

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        return (1,)

    def reset_from_box(self, uniform: jnp.ndarray) -> State_:
        assert uniform.shape == (1,)
        box = uniform[0]
        # Change [-1, 1] -> [px0_lo, px0_hi]
        px0_lo, px0_hi = (0, 1)  # self.task_cfg.px0_bounds
        px0 = (box + 1.0) / 2.0 * (px0_hi - px0_lo) + px0_lo
        state = self.state0_from_px0(px0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def reset_papereval(self, b_key: jr.PRNGKey, num: int) -> State:
        """Reset uniformly to remove any randomness."""

        def f(px0):
            state = self.state0_from_px0(px0)
            state = jdc.replace(state, source=Source.BASE)
            return state

        px0_lo, px0_hi = (0, 1)
        b_px0 = jnp.linspace(px0_lo, px0_hi, num=num)
        return jax.vmap(f)(b_px0)