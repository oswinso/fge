from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from gymnasium.spaces import Discrete
from jax import random as jr
from matplotlib import pyplot as plt
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
from fge.core.envs.toylevels.toylevels import TaskCfg, ToyLevels


class ToyLevelsJax(JaxTask):
    @jdc.pytree_dataclass
    class State(TimedState):
        pos: jnp.ndarray
        reset_region: jnp.ndarray

        # the x,y coordinate of the position at time 0
        ic: jnp.ndarray

    def __init__(self, task_cfg: TaskCfg):
        # left, right, nothing
        action_space = Discrete(n=3)
        super().__init__(action_space)
        self.task_cfg = task_cfg
        self.task_cpu = ToyLevels(task_cfg, None)

        rng = np.random.default_rng(seed=12345)
        self.n_ff_feats = 16
        scale = 2
        n_pos = 2
        self.ff_W = rng.uniform(low=-1 / scale, high=1 / scale, size=(self.n_ff_feats, n_pos))
        self.ff_b = rng.uniform(low=0, high=2 * np.pi, size=(self.n_ff_feats,))

    @property
    def eval_rollout_T(self) -> int:
        steps = int(np.ceil((self.reset_y - self.task_cfg.env_ylb) / self.task_cfg.agent_drop_rate))
        return steps + 1

    def _convert_action(self, action: jnp.ndarray):
        assert action.shape == tuple() and action.dtype == jnp.int32
        outputs = jnp.array(
            [
                [0.0, 0.0],
                [-self.task_cpu.agent_stepsize, 0.0],
                [self.task_cpu.agent_stepsize, 0.0],
            ]
        )
        dstate = outputs[action]
        assert dstate.shape == (2,)
        return dstate

    def _in_disturb_box(self, state: State) -> bool:
        assert state.pos.shape == (2,)
        x, y = state.pos
        x_lb = self.task_cfg.disturb_x_lb - 1
        x_ub = self.task_cfg.disturb_x_ub + 1
        y_lb = self.task_cfg.disturb_y_lb
        y_ub = self.task_cfg.disturb_y_ub
        return (x_lb <= x) & (x <= x_ub) & (y_lb <= y) & (y < y_ub)

    def _is_finished(self, state: State):
        x, y = state.pos

        # y outside the bounds.
        y_lb = self.task_cfg.env_ylb
        is_done = y <= y_lb
        return is_done

    def _get_collided(self, state: State):
        x, y = state.pos

        # Agent out of bounds. Only look at x, since y is automatic.
        x_lb = self.task_cfg.env_xlb
        x_ub = self.task_cfg.env_xub

        in_bounds = (x_lb <= x) & (x <= x_ub)
        oob = ~in_bounds

        # Agent in wall1.
        wall_x = self.task_cfg.easy_xub
        hit_wall1 = jnp.abs(x - wall_x) <= self.task_cfg.agent_radius

        # Agent in wall2.
        wall_X = self.task_cfg.hard_xub
        hit_wall2 = jnp.abs(x - wall_X) <= self.task_cfg.agent_radius

        # Agent hit impossible obstacle.
        x_lb = self.task_cfg.obstacle_x_lb
        x_ub = self.task_cfg.obstacle_x_ub
        y_lb = self.task_cfg.obstacle_y_lb
        y_ub = self.task_cfg.obstacle_y_ub
        hit_obs = (x_lb <= x) & (x <= x_ub) & (y_lb <= y) & (y < y_ub)

        # Reset region has changed.
        reset_region = self.which_reset_reegion(state.pos)
        changed_region = state.reset_region != reset_region

        return oob | hit_wall1 | hit_wall2 | hit_obs | changed_region

    def which_reset_reegion(self, pos: jnp.ndarray):
        # return 0: x in [easy_xlb, easy_xub]
        # return 1: x in [hard_xlb, hard_xub]
        # return 2: x in [impossible_xlb, impossible_xub]
        x = pos[0]
        x_easy_lb = self.task_cfg.easy_xlb
        x_easy_ub = self.task_cfg.easy_xub
        x_hard_lb = self.task_cfg.hard_xlb
        x_hard_ub = self.task_cfg.hard_xub
        x_impossible_lb = self.task_cfg.impossible_xlb
        x_impossible_ub = self.task_cfg.impossible_xub
        in_easy = (x_easy_lb <= x) & (x <= x_easy_ub)
        in_hard = (x_hard_lb <= x) & (x <= x_hard_ub)
        in_impossible = (x_impossible_lb <= x) & (x <= x_impossible_ub)

        out = jnp.where(in_easy, 0, jnp.where(in_hard, 1, jnp.where(in_impossible, 2, -1)))
        return out

    def step(self, state: State, action: jnp.ndarray) -> StepOutput:
        dstate = self._convert_action(action)
        pos_new = state.pos + dstate
        assert pos_new.shape == (2,)

        # If agent is in disturb region, push it right by a fixed amount
        in_disturb_box = self._in_disturb_box(state)
        dx = jnp.where(in_disturb_box, 2 * self.task_cpu.agent_stepsize, 0.0)
        pos_new = pos_new.at[0].add(dx)

        # Agent always goes downward
        pos_new = pos_new.at[1].add(-self.task_cfg.agent_drop_rate)

        collided = self._get_collided(state)
        rew = jnp.where(collided, -1.0, 0.0)

        # Terminated if
        term = collided | self._is_finished(state)

        # Never truncated.
        trunc = False

        info = {
            "collided": collided,
            "ic": state.ic[..., 0],
        }
        state_new = jdc.replace(state, step=state.step + 1, pos=pos_new)
        obs = self.get_obs(state_new)

        return StepOutput(state_new, obs, rew, term, trunc, info)

    @property
    def reset_y(self):
        # 49.5
        return self.task_cfg.env_yub - 2 * self.task_cfg.agent_radius

    def state_from_px(self, px: float):
        pos = jnp.array([px, self.reset_y])
        assert pos.shape == (2,)
        reset_region = self.which_reset_reegion(pos)
        return self.state0_from_px(pos, reset_region)

    def reset(self, key: jr.PRNGKey, options: dict | None = None) -> State:
        key_region, key_pos = jr.split(key, 2)
        reset_region = jr.choice(
            key_region,
            3,
            p=np.array(
                [
                    self.task_cfg.easy_prob,
                    self.task_cfg.hard_prob,
                    self.task_cfg.impossible_prob,
                ]
            ),
        )
        task_cpu = self.task_cpu
        reset_xlb = jnp.array(
            [
                task_cpu.easy_reset_region[0],
                task_cpu.hard_reset_region[0],
                task_cpu.impossible_reset_region[0],
            ]
        )[reset_region]
        reset_xub = jnp.array(
            [
                task_cpu.easy_reset_region[1],
                task_cpu.hard_reset_region[1],
                task_cpu.impossible_reset_region[1],
            ]
        )[reset_region]

        x = jr.uniform(key_pos, minval=reset_xlb, maxval=reset_xub)
        state = self.state_from_px(x)

        return state

    def get_dummy_obs(self):
        if not hasattr(JaxTask, "obs_shape"):
            state = jax.jit(self.reset)(jr.PRNGKey(1234))
            obs = self.get_obs(state)
            JaxTask.obs_shape = obs.shape

        return np.zeros(JaxTask.obs_shape)

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        return (1,)

    def get_dummy_obs(self):
        if not hasattr(JaxTask, "obs_shape"):
            state = jax.jit(self.reset)(jr.PRNGKey(1234))
            obs = self.get_obs(state)
            JaxTask.obs_shape = obs.shape

        return np.zeros(JaxTask.obs_shape)

    def get_reset_bounds(self, reset_region: int):
        task_cpu = self.task_cpu

        reset_xlb = jnp.array(
            [
                task_cpu.easy_reset_region[0],
                task_cpu.hard_reset_region[0],
                task_cpu.impossible_reset_region[0],
            ]
        )[reset_region]
        reset_xub = jnp.array(
            [
                task_cpu.easy_reset_region[1],
                task_cpu.hard_reset_region[1],
                task_cpu.impossible_reset_region[1],
            ]
        )[reset_region]

        return reset_xlb, reset_xub

    def reset_from_box(self, uniform: jnp.ndarray) -> State_:
        assert uniform.shape == (1,)
        (px_box,) = uniform

        # Map from [-1, 1] to [0, 1].
        px_box = (px_box + 1) / 2

        # Map from [0, 1] to the reset region with uniform probability.
        is_easy = (0.0 <= px_box) & (px_box < 1 / 3)
        is_hard = (1 / 3 <= px_box) & (px_box < 2 / 3)
        reset_region = jnp.where(is_easy, 0, jnp.where(is_hard, 1, 2))

        reset_xlb, reset_xub = self.get_reset_bounds(reset_region)

        # Map px_box back to [0, 1].
        x_lo = jnp.array([0, 1 / 3, 2 / 3])[reset_region]
        x_hi = jnp.array([1 / 3, 2 / 3, 1.0])[reset_region]

        box01 = (px_box - x_lo) / (x_hi - x_lo)

        # Map [0, 1] to reset_xlb, reset_xub
        px = reset_xlb + box01 * (reset_xub - reset_xlb)

        return self.state_from_px(px)

    def box_from_reset(self, state: State) -> jnp.ndarray:
        task_cpu = self.task_cpu
        assert state.pos.shape == (2,)
        px = state.pos[0]

        # Determine reset region based on px (in original coordinate space)
        is_easy = (task_cpu.easy_reset_region[0] <= px) & (px < task_cpu.easy_reset_region[1])
        is_hard = (task_cpu.hard_reset_region[0] <= px) & (px < task_cpu.hard_reset_region[1])
        reset_region = jnp.where(is_easy, 0, jnp.where(is_hard, 1, 2))

        reset_xlb, reset_xub = self.get_reset_bounds(reset_region)

        # Step 3: box01 in [0,1]
        box01 = (px - reset_xlb) / (reset_xub - reset_xlb)

        x_lo = jnp.array([0, 1 / 3, 2 / 3])[reset_region]
        x_hi = jnp.array([1 / 3, 2 / 3, 1.0])[reset_region]
        px_box = x_lo + box01 * (x_hi - x_lo)

        # Step 5: map back to uniform in [-1,1]
        uniform = 2 * px_box - 1
        return uniform.reshape((1,))

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State_:
        del b_key

        task_cpu = self.task_cpu
        reset_xlb = np.array(
            [
                task_cpu.easy_reset_region[0],
                task_cpu.hard_reset_region[0],
                task_cpu.impossible_reset_region[0],
            ]
        )
        reset_xub = np.array(
            [
                task_cpu.easy_reset_region[1],
                task_cpu.hard_reset_region[1],
                task_cpu.impossible_reset_region[1],
            ]
        )

        # Uniformly sample.
        reset_widths = reset_xub - reset_xlb

        # Distribute num proportional to reset_widths.
        reset_widths = reset_widths / np.sum(reset_widths)

        n_easy = int(np.round(num * reset_widths[0]))
        n_hard = int(np.round(num * reset_widths[1]))
        n_impossible = num - (n_easy + n_hard)

        b_px_easy = np.linspace(reset_xlb[0], reset_xub[0], n_easy)
        b_px_hard = np.linspace(reset_xlb[1], reset_xub[1], n_hard)
        b_px_impossible = np.linspace(reset_xlb[2], reset_xub[2], n_impossible)

        b_px = np.concatenate([b_px_easy, b_px_hard, b_px_impossible], axis=0)
        return jax.vmap(self.state_from_px)(b_px)

    def get_obs(self, state: State):
        # [0, 50]^2
        pos = state.pos

        pos_centered = (pos - 25) / 25

        # Encode position using fourier features.
        pos_proj = jnp.sum(pos * self.ff_W, axis=1) + self.ff_b
        assert pos_proj.shape == (self.n_ff_feats,)
        obs_pos = jnp.concatenate([pos_centered, jnp.cos(pos_proj), jnp.sin(pos_proj)], axis=0)

        # one-hot encode the region.
        obs_region = jnp.zeros((3,))
        obs_region = obs_region.at[state.reset_region].set(1.0)
        obs = jnp.concatenate([obs_pos, obs_region])
        return obs

    def state0_from_px(self, pos: jnp.ndarray, region: jnp.ndarray):
        state = ToyLevelsJax.State(step=0, source=Source.BASE, pos=pos, reset_region=region, ic=pos.copy())
        return state

    def get_eval_states(self, n_per_region: int = 20) -> EvalStateInfo:
        env = self.task_cpu

        regions: list = []

        b_pos_list = []
        for region_name, region in [
            ("easy", env.easy_reset_region),
            ("hard", env.hard_reset_region),
            ("impossible", env.impossible_reset_region),
        ]:
            regions.append(EvalRegionInfo(region_name, n_per_region))

            b_x = np.linspace(region[0], region[1], n_per_region)
            b_y = np.full(n_per_region, self.reset_y)
            b_pos = np.stack([b_x, b_y], axis=1)

            b_pos_list.append(b_pos)

        b_pos = np.concatenate(b_pos_list, axis=0)
        n_reset = len(b_pos)

        b_region = jax.vmap(self.which_reset_reegion)(b_pos)
        b_zero = jnp.zeros(n_reset, dtype=jnp.int32)
        # b_state = ToyLevelsJax.State(b_zero, b_pos, b_region)
        b_state = jax.vmap(self.state0_from_px)(b_pos, b_region)
        return EvalStateInfo(b_state, n_reset, regions)

    @property
    def region_names(self):
        return ["easy", "hard", "impossible"]

    def get_contour_grid(self):
        nx = 256
        ny = 256
        b_x = np.linspace(self.task_cfg.env_xlb, self.task_cfg.env_xub, nx)
        b_y = np.linspace(self.task_cfg.env_ylb, self.task_cfg.env_yub, ny)
        bb_X, bb_Y = np.meshgrid(b_x, b_y)
        bb_pos = np.stack([bb_X, bb_Y], axis=-1)
        bb_region = jax_vmap(self.which_reset_reegion, rep=2)(bb_pos)
        bb_zero = jnp.zeros((nx, ny), dtype=jnp.int32)
        bb_state = ToyLevelsJax.State(bb_zero, bb_zero, bb_pos, bb_region, bb_pos.copy())
        bb_obs = jax_vmap(self.get_obs, rep=2)(bb_state)
        return bb_X, bb_Y, bb_obs, bb_state

    def get_eval_contour(self):
        nx_per_region = 64
        b_state, _, _ = self.get_eval_states(nx_per_region)
        b_x = b_state.pos[:, 0]
        b_obs = jax.vmap(self.get_obs)(b_state)
        return b_x, b_obs, b_state

    def eval_ics(self, n_per_region: int = 20):
        """Some 1D representation of the eval states from get_eval_states."""
        b_state, _, _ = self.get_eval_states(n_per_region)
        b_x = b_state.pos[:, 0]
        return b_x

    def to_icval(self, state: State) -> np.ndarray:
        pos = state.pos
        # We are varying the x-position for the initial conditions.
        return pos[..., 0]

    def icval_bins(self, n_bins: int = 31) -> np.ndarray:
        return np.linspace(self.task_cfg.env_xlb, self.task_cfg.env_xub, n_bins)

    def setup_trajplot(self, ax: plt.Axes):
        self.task_cpu._fig = None
        self.task_cpu._get_fig_and_ax(draw_agent=False, ax=ax)

