import functools as ft
import itertools
import pathlib
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import spaces
from jax import random as jr
from loguru import logger
from mujoco import mjx
from mujoco.mjx._src.collision_primitive import capsule_capsule
from og.dyn_types import QpFloat, QvFloat
from og.jax_types import BBool, BFloat, FloatScalar, IntScalar
from og.jax_utils import jax_vmap
from og.rng import PRNGKey

from fge.core.bits.state_reset_id import Source
from fge.core.envs.jax_task import (
    EvalRegionInfo,
    EvalStateInfo,
    JaxTask,
    State_,
    StepOutput,
    TimedState,
)
from fge.core.envs.mujoco.hopper.hopper import Hopper, TaskCfg
from fge.core.envs.mujoco.utils import invtrans2d_jax, transform2d, transform2d_jax


class HopperJax(JaxTask):
    Cfg = TaskCfg

    @jdc.pytree_dataclass
    class State(TimedState):
        px0: FloatScalar
        reset_region: IntScalar
        data: mjx.Data
        ic: FloatScalar

    @jdc.pytree_dataclass
    class MinState(TimedState):
        px0: FloatScalar
        reset_region: IntScalar
        qpos: jnp.ndarray
        qvel: jnp.ndarray
        ic: FloatScalar

    @staticmethod
    def set_precision():
        jax.config.update("jax_default_matmul_precision", "highest")

    def __init__(self, task_cfg: TaskCfg):
        self.task_cfg = task_cfg
        self.task_cpu = Hopper(task_cfg, paths=None)

        # Use a modified version of the model for mujoco-mjx to speedup.
        if task_cfg.eric_gear:
            self.model_path = pathlib.Path(__file__).parent.parent / "xmls/hopper_eric.xml"
        else:
            self.model_path = pathlib.Path(__file__).parent.parent / "xmls/hopper.xml"

        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(self.model_path))

        if jax.config.jax_default_matmul_precision != "highest":
            raise ValueError("Jax matmul precision must be set to highest, or we get NaNs...")

        bounds = self._model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        super().__init__(action_space)

        # mjx setup.
        # cpu_device = jax.devices("cpu")[0]
        self.mjx_model = mjx.put_model(self.model)

        # Eval info.
        self.boundaries = self.task_cpu.boundaries
        self._region_names = [
            "LeftEdge",
            "LeftSlope",
            "Gap",
            "RightSlope",
            "RightFlat",
            "RightEdge",
        ]
        self.n_eval_regions = len(self.boundaries) - 1
        self.eval_regions = np.stack([self.boundaries[:-1], self.boundaries[1:]], axis=1)
        assert len(self._region_names) == self.n_eval_regions

        self._eval_rollout_T = 1000

        self.use_ff = False
        rng = np.random.default_rng(seed=12345)
        self.n_ff_feats = 16
        scale = 0.05
        self.ff_W = rng.uniform(low=-1 / scale, high=1 / scale, size=self.n_ff_feats)
        self.ff_b = rng.uniform(low=0, high=2 * np.pi, size=self.n_ff_feats)

    @property
    def model(self):
        return self._model

    @property
    def eval_rollout_T(self) -> int:
        return self._eval_rollout_T

    @eval_rollout_T.setter
    def eval_rollout_T(self, T: int):
        assert T > 0
        self._eval_rollout_T = T

    def get_obs(self, state: State):
        data = state.data
        qpos = data.qpos
        qvel = data.qvel
        obs = jnp.concatenate([qpos, qvel])

        if self.use_ff:
            ankle0 = qpos[-1]
            ankle0_obs = jnp.concatenate(
                [
                    jnp.sin(self.ff_W * ankle0 + self.ff_b),
                    jnp.cos(self.ff_W * ankle0 + self.ff_b),
                ]
            )
            obs = jnp.concatenate([obs, ankle0_obs])

        return obs

    @staticmethod
    def masked_min(b_val: BFloat, b_valid: BBool, invalid_val: float):
        b_val = jnp.where(b_valid, b_val, invalid_val)
        return b_val.min()

    def _get_collided(self, state: State):
        data = state.data
        contact = data.contact

        # contacting floor is unsafe
        id_floorl = self.model.geom("floor_l").id
        id_floorr = self.model.geom("floor_r").id
        id_floorr2 = self.model.geom("floor_r2").id

        id_thigh = self.model.geom("thigh").id
        id_pelvis = self.model.geom("pelvis").id
        id_torso = self.model.geom("torso").id
        id_nose = self.model.geom("nose").id

        # g1, g2 = geom.T => geom.shape = (n_pairs, 2)
        # Neither of [id_thigh, id_pelvis, id_torso, id_nose] should be in contact with [id_floorl, id_floorr].

        # n_pairs = 4 * 3 = 12
        ids_floor = [id_floorl, id_floorr, id_floorr2]
        ids_body = [id_thigh, id_pelvis, id_torso, id_nose]
        geoms = np.array(list(itertools.product(ids_floor, ids_body)))
        assert geoms.shape == (12, 2)
        b_dist, _, _ = capsule_capsule(self.mjx_model, state.data, None, geoms)
        assert b_dist.shape == (12,)

        sdf_body = b_dist.min()
        healthy_body = sdf_body > 0.0

        # don't fall down through the gap.
        pos_z = data.qpos[1]
        healthy_z = pos_z >= -0.9

        is_healthy = healthy_body & healthy_z

        return ~is_healthy

    def _is_finished(self, state: State):
        # Define it as a time-based thing.
        return state.step >= self.task_cfg.max_steps

    def step_mujoco_simulation(self, state: State, action: jnp.ndarray):
        assert action.shape == (self.model.nu,)
        assert state.data.ctrl.shape == action.shape
        data_new = state.data.replace(ctrl=action)

        # mj_step, step=n_frames.
        data_new = self.mj_step(data_new, n_frames=self.task_cpu.frame_skip)
        return data_new

    def mj_step(self, data: mjx.Data, n_frames: int):
        def f(state: mjx.Data, _):
            data_new = mjx.step(self.mjx_model, state)
            return data_new, None

        pipeline_state_new, _ = lax.scan(f, data, (), length=n_frames)
        return pipeline_state_new

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        data_new = self.step_mujoco_simulation(state, action)
        state_new = jdc.replace(state, step=state.step + 1, data=data_new)
        obs = self.get_obs(state_new)

        # Terminated if collided, or survive for enough time.
        collided = self._get_collided(state)
        rew = jnp.where(collided, -1.0, 0.0)

        term = collided

        # Never truncated.
        trunc = self._is_finished(state)

        info = {
            'ic': state_new.ic,
        }

        return StepOutput(state_new, obs, rew, term, trunc, info)

    def get_minstate(self, state: State) -> MinState:
        qpos = state.data.qpos
        qvel = state.data.qvel
        return HopperJax.MinState(state.step, state.source, state.px0, state.reset_region, qpos, qvel, state.px0.copy())

    def from_minstate(self, minstate: MinState) -> State:
        data = self.pipeline_init(minstate.qpos, minstate.qvel)
        return HopperJax.State(minstate.step, minstate.source, minstate.px0, minstate.reset_region, data, minstate.px0.copy())

    @ft.cache
    def _minstate_treedef(self) -> jtu.PyTreeDef:
        minstate = HopperJax.MinState(0, 0, 0, 0, 0, 0, 0)
        treedef = jtu.tree_structure(minstate)
        return treedef

    @property
    def minstate_treedef(self):
        return self._minstate_treedef()

    def pipeline_init(self, q: QpFloat, qd: QvFloat) -> mjx.Data:
        """Initializes the pipeline state."""
        # Use the mjx init directly.
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=q, qvel=qd)
        data = mjx.forward(self.mjx_model, data)
        # The qacc and qacc_warmstart also share the same buffer....
        data = data.replace(qacc_warmstart=data.qacc_warmstart.copy())
        return data

    def get_data_from_px(self, px: float):
        X_W_foot = jnp.array([px, 1.0, 0.0])

        joints = np.zeros(4)
        X_foot_root = self.get_X_foot_root(joints[0], joints[1], joints[2], joints[3])
        X_W_root = transform2d_jax(X_W_foot, X_foot_root)

        X_rootW_W = np.array([0.0, -1.0, 0.0])
        X_rootW_root = transform2d_jax(X_rootW_W, X_W_root)

        qpos = jnp.array(
            [
                X_rootW_root[0],
                X_rootW_root[1],
                X_rootW_root[2],
                joints[0],
                joints[1],
                joints[2],
                joints[3],
            ]
        )
        qvel = jnp.zeros(self.model.nv)

        data = self.pipeline_init(qpos, qvel)
        return data

    @staticmethod
    def get_X_foot_root(
        pelvis: float | jnp.ndarray,
        thigh: float | jnp.ndarray,
        leg: float | jnp.ndarray,
        ankle: float | jnp.ndarray,
    ):
        torso_offset = 1.0
        X_W_torso = np.array([0.0, torso_offset, 0.0])

        pelvis_offset = -0.05
        X_torso_pelvis = np.array([0.0, pelvis_offset, pelvis])
        X_W_pelvis = transform2d(X_W_torso, X_torso_pelvis)

        thigh_offset = -0.2
        X_pelvis_thigh = np.array([0.0, thigh_offset, thigh])
        X_W_thigh = transform2d(X_W_pelvis, X_pelvis_thigh)

        calf_offset = -0.33
        X_thigh_calf = np.array([0.0, calf_offset, leg])
        X_W_calf = transform2d(X_W_thigh, X_thigh_calf)

        foot_offset = -0.32
        X_calf_foot = jnp.array([0.0, foot_offset, ankle])
        X_W_foot = transform2d_jax(X_W_calf, X_calf_foot)

        X_foot_root = transform2d_jax(invtrans2d_jax(X_W_foot), X_W_torso)
        return X_foot_root

    def get_reset_region(self, px0: float):
        """Compute which reset_region the ankle0 is in."""
        # Find the first index i such that boundaries[i] <= px0 < boundaries[i+1]
        idx = jnp.searchsorted(jnp.array(self.boundaries), px0, side="right") - 1
        return jnp.clip(idx, 0, self.n_eval_regions - 1)

    def state0_from_px(self, px: float):
        data = self.get_data_from_px(px)
        reset_region = self.get_reset_region(px)
        state = HopperJax.State(0, 0, px, reset_region, data, px.copy())
        return state

    def reset(self, key: jr.PRNGKey) -> State:
        if self.task_cfg.px0 is None:
            px0_lo, px0_hi = self.task_cfg.px0_bounds
            px0 = jr.uniform(key, minval=px0_lo, maxval=px0_hi)
        else:
            px0 = self.task_cfg.px0
        state = self.state0_from_px(px0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def get_uniform_obs0_ic(self, num: int = 256):
        def f(px0):
            state = self.state0_from_px(px0)
            state = jdc.replace(state, source=Source.BASE)
            return state

        px0_lo, px0_hi = self.task_cfg.px0_bounds
        b_px0 = jnp.linspace(px0_lo, px0_hi, num=num)
        b_state = jax.vmap(f)(b_px0)
        b_obs = jax.vmap(self.get_obs)(b_state)
        return b_px0, b_obs

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State:
        """Reset uniformly to remove any randomness."""

        def f(px0):
            state = self.state0_from_px(px0)
            state = jdc.replace(state, source=Source.BASE)
            return state

        px0_lo, px0_hi = self.task_cfg.px0_bounds
        b_px0 = jnp.linspace(px0_lo, px0_hi, num=num)
        return jax.vmap(f)(b_px0)

    def reset_from_box(self, uniform: jnp.ndarray) -> State:
        assert uniform.shape == (1,)
        box = uniform[0]
        # Change [-1, 1] -> [px0_lo, px0_hi]
        px0_lo, px0_hi = self.task_cfg.px0_bounds
        px0 = (box + 1.0) / 2.0 * (px0_hi - px0_lo) + px0_lo
        state = self.state0_from_px(px0)
        state = jdc.replace(state, source=Source.BASE)
        return state

    def box_from_reset(self, state: State):
        px0 = state.px0
        # Change [px0_lo, px0_hi] -> [-1, 1]
        px0_lo, px0_hi = self.task_cfg.px0_bounds
        box = (px0 - px0_lo) / (px0_hi - px0_lo) * 2.0 - 1.0
        return jnp.array([box])

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def region_names(self):
        return self._region_names

    def get_eval_states(self, n_per_region: int = 10) -> EvalStateInfo:
        b_px0, regions = self.eval_xs_(n_per_region)
        n_reset = n_per_region * self.n_eval_regions
        b_state = jax.vmap(self.state0_from_px)(b_px0)

        return EvalStateInfo(b_state, n_reset, regions)

    def get_eval_contour(self):
        n_per_region = 40
        b_state, _, _ = self.get_eval_states(n_per_region)

        b_px0 = b_state.px0
        b_obs = jax.vmap(self.get_obs)(b_state)

        return b_px0, b_obs, b_state

    def eval_xs_(self, n_per_region: int = 10):
        """Some 1D representation of the eval states."""
        regions: list = []
        b_px0_all = []

        for ii, (name, region) in enumerate(zip(self.region_names, self.eval_regions)):
            regions.append(EvalRegionInfo(name, n_per_region))
            # Include the edges for the first and last regions. Otherwise, only use the middle points.
            is_first_or_last = (ii == 0) or (ii == self.n_eval_regions - 1)
            if is_first_or_last:
                b_px0 = np.linspace(region[0], region[1], n_per_region)
            else:
                b_px0 = np.linspace(region[0], region[1], n_per_region + 2)[1:-1]

            b_px0_all.append(b_px0)

        b_px0 = np.concatenate(b_px0_all, axis=0)

        return b_px0, regions

    def eval_ics(self, n_per_region: int = 10):
        """Some 1D representation of the eval states."""
        b_px0, regions = self.eval_xs_(n_per_region)
        return b_px0

    def to_icval(self, state: State | MinState) -> np.ndarray:
        return state.px0

    @property
    def icval_bounds(self):
        return self.task_cfg.px0_bounds

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

    def label_ic(self, ax: plt.Axes):
        return self.task_cpu.label_ic(ax)

    def setup_trajplot(self, ax: plt.Axes):
        return self.task_cpu.setup_trajplot(ax)
