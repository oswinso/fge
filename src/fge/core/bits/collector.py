import copy
import functools as ft
from types import ModuleType
from typing import Any, NamedTuple, Protocol, Self

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from attrs import define
from flax import struct
from jax_tqdm import scan_tqdm
from og.cfg_utils import Cfg
from og.dyn_types import Control, Obs, TBool, TControl, TFloat, TObs
from og.jax_types import BBool, FloatScalar, IntScalar
from og.jax_utils import concat_at_front
from og.rng import PRNGKey
from og.tree_utils import tree_where_dim0

from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import StateResetId
from fge.core.envs.jax_task import (
    JaxTask,
    StepOutput,
    TimedState,
    TreeLeaves,
    leaf_concat,
    leaf_index,
    leaf_stack,
)


# type StateResetId = tuple[TimedState, IntScalar]
# type ResetBuf = CircLIFOJax[StateResetId]


class ResetBufLike(Protocol):
    def sample(self, key: PRNGKey, n_samples: int, b_valid: BBool) -> tuple[Self, StateResetId]: ...


@define(eq=False)
class CollectorCfg(Cfg):
    # Batch size when collecting data.
    n_envs: int
    pbar: bool = False


# Somehow if CollectorCfg doesn't have a hash function, then manually add it.
if CollectorCfg.__hash__ == None:
    def _hash(self):
        return hash((self.n_envs, self.pbar))


    CollectorCfg.__hash__ = _hash


class CollectorState(NamedTuple):
    steps: IntScalar
    reset_id: IntScalar
    state: TimedState


@struct.dataclass(slots=True, frozen=False)
class RolloutOutput:
    T_reset_id: IntScalar
    """Identifier (based on x0) for the current state (state_now). -1 if it's randomly sampled."""

    T_state_now: TreeLeaves
    """Current state. Note: This is a list of arrays, output of tree_leaves."""

    T_state_nxt: TreeLeaves
    """Next state. Note: This is a list of arrays, output of tree_leaves."""

    T_obs_now: TObs | TreeLeaves
    """Observations corresponding to the current state."""

    T_obs_nxt: TObs | TreeLeaves
    """Observations corresponding to the next state."""

    T_control: TControl
    """Control inputs applied to transition from the current state to the next state."""

    T_logprob: TFloat
    """Logarithm of the probability of the chosen control action."""

    T_rew: TFloat
    """Reward received for the transition from the current state to the next state."""

    T_trunc: TBool
    """Boolean flag indicating whether the episode was truncated."""

    T_term: TBool
    """Boolean flag indicating whether the episode was terminated."""

    T_info: dict
    """Additional metadata or information associated with the rollout."""

    x0_lazy: np.ndarray | None = None

    @property
    def x0(self):
        if self.x0_lazy is not None:
            return self.x0_lazy
        return jtu.tree_map(lambda x: x[0], self.T_state_now)

    @property
    def Tp1_state(self):
        state0 = jtu.tree_map(lambda x: x[0], self.T_state_now)
        Tp1_state = jtu.tree_map(concat_at_front, state0, self.T_state_nxt)
        return Tp1_state

    def tree_index(self, idx: Any):
        reset_id = self.T_reset_id[idx]
        state_now = leaf_index(self.T_state_now, idx)
        state_nxt = leaf_index(self.T_state_nxt, idx)

        if isinstance(self.T_obs_now, (np.ndarray, jnp.ndarray)):
            obs_now = self.T_obs_now[idx]
            obs_nxt = self.T_obs_nxt[idx]
        else:
            obs_now = leaf_index(self.T_obs_now, idx)
            obs_nxt = leaf_index(self.T_obs_nxt, idx)

        control = self.T_control[idx]
        logprob = self.T_logprob[idx]
        rew = self.T_rew[idx]
        trunc = self.T_trunc[idx]
        term = self.T_term[idx]
        info = {k: v[idx] for k, v in self.T_info.items()}
        return RolloutOutput(
            reset_id,
            state_now,
            state_nxt,
            obs_now,
            obs_nxt,
            control,
            logprob,
            rew,
            trunc,
            term,
            info,
        )

    @classmethod
    def tree_stack_lazy(cls, rollouts: list[Self], axis: int, which: ModuleType = np) -> Self:
        rollouts: list[RolloutOutput]

        b_reset = which.stack([r.T_reset_id for r in rollouts], axis=axis)
        b_rew = which.stack([r.T_rew for r in rollouts], axis=axis)
        b_trunc = which.stack([r.T_trunc for r in rollouts], axis=axis)
        b_term = which.stack([r.T_term for r in rollouts], axis=axis)

        x0 = rollouts[0].T_state_now
        return RolloutOutput(
            b_reset,
            [r.T_state_now for r in rollouts],
            [r.T_state_nxt for r in rollouts],
            [r.T_obs_now for r in rollouts],
            [r.T_obs_nxt for r in rollouts],
            [r.T_control for r in rollouts],
            [r.T_logprob for r in rollouts],
            b_rew,
            b_trunc,
            b_term,
            {k: [r.T_info[k] for r in rollouts] for k in rollouts[0].T_info.keys()},
            x0_lazy=x0,
        )

    @classmethod
    def tree_stack(cls, rollouts: list[Self], axis: int, which: ModuleType = np) -> Self:
        rollouts: list[RolloutOutput]
        b_reset = which.stack([r.T_reset_id for r in rollouts], axis=axis)
        b_state_now = leaf_stack([r.T_state_now for r in rollouts], axis=axis, which=which)
        b_state_nxt = leaf_stack([r.T_state_nxt for r in rollouts], axis=axis, which=which)

        is_obs_arr = isinstance(rollouts[0].T_obs_now, (np.ndarray, jnp.ndarray))
        if is_obs_arr:
            b_obs_now = which.stack([r.T_obs_now for r in rollouts], axis=axis)
            b_obs_nxt = which.stack([r.T_obs_nxt for r in rollouts], axis=axis)
        else:
            b_obs_now = leaf_stack([r.T_obs_now for r in rollouts], axis=axis, which=which)
            b_obs_nxt = leaf_stack([r.T_obs_nxt for r in rollouts], axis=axis, which=which)

        b_control = which.stack([r.T_control for r in rollouts], axis=axis)
        b_logprob = which.stack([r.T_logprob for r in rollouts], axis=axis)
        b_rew = which.stack([r.T_rew for r in rollouts], axis=axis)
        b_trunc = which.stack([r.T_trunc for r in rollouts], axis=axis)
        b_term = which.stack([r.T_term for r in rollouts], axis=axis)

        info_keys = rollouts[0].T_info.keys()
        b_info = {k: which.stack([r.T_info[k] for r in rollouts], axis=axis) for k in info_keys}

        return RolloutOutput(
            b_reset,
            b_state_now,
            b_state_nxt,
            b_obs_now,
            b_obs_nxt,
            b_control,
            b_logprob,
            b_rew,
            b_trunc,
            b_term,
            b_info,
        )

    @classmethod
    def tree_concat(cls, rollouts: list[Self], axis: int, which: ModuleType = np) -> Self:
        is_obs_arr = isinstance(rollouts[0].T_obs_now, (np.ndarray, jnp.ndarray))

        rollouts: list[RolloutOutput]
        b_reset = which.concatenate([r.T_reset_id for r in rollouts], axis=axis)
        b_state_now = leaf_concat([r.T_state_now for r in rollouts], axis=axis, which=which)
        b_state_nxt = leaf_concat([r.T_state_nxt for r in rollouts], axis=axis, which=which)
        if is_obs_arr:
            b_obs_now = which.concatenate([r.T_obs_now for r in rollouts], axis=axis)
            b_obs_nxt = which.concatenate([r.T_obs_nxt for r in rollouts], axis=axis)
        else:
            b_obs_now = leaf_concat([r.T_obs_now for r in rollouts], axis=axis, which=which)
            b_obs_nxt = leaf_concat([r.T_obs_nxt for r in rollouts], axis=axis, which=which)

        b_control = which.concatenate([r.T_control for r in rollouts], axis=axis)
        b_logprob = which.concatenate([r.T_logprob for r in rollouts], axis=axis)
        b_rew = which.concatenate([r.T_rew for r in rollouts], axis=axis)
        b_trunc = which.concatenate([r.T_trunc for r in rollouts], axis=axis)
        b_term = which.concatenate([r.T_term for r in rollouts], axis=axis)

        info_keys = rollouts[0].T_info.keys()
        b_info = {k: which.concatenate([r.T_info[k] for r in rollouts], axis=axis) for k in info_keys}

        return RolloutOutput(
            b_reset,
            b_state_now,
            b_state_nxt,
            b_obs_now,
            b_obs_nxt,
            b_control,
            b_logprob,
            b_rew,
            b_trunc,
            b_term,
            b_info,
        )


class DistributionLike(Protocol):
    def experimental_sample_and_log_prob(self, seed: PRNGKey) -> tuple[Control, FloatScalar]: ...

    def log_prob(self, a: Control) -> FloatScalar: ...

    def mode(self) -> Control: ...


class GetPolFn(Protocol):
    def __call__(self, obs: Obs) -> DistributionLike | tuple[DistributionLike, dict]: ...


class ResetFn(Protocol):
    def __call__(self, b_key: PRNGKey, num: int) -> TimedState: ...


class EnvResetFn:
    def __init__(self, task: JaxTask):
        self.task = task

    def __call__(self, b_key: PRNGKey, num: int) -> TimedState:
        return jax.vmap(self.task.reset)(b_key)


class Collector(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    collect_state: CollectorState
    task: JaxTask = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    Cfg = CollectorCfg
    Rollout = RolloutOutput

    @classmethod
    def create(
            cls,
            key: PRNGKey,
            task: JaxTask,
            cfg: CollectorCfg,
            id_provider: ResetIDProvider | None = None,
    ):
        key, key_init = jr.split(key)
        b_key_init = jr.split(key_init, cfg.n_envs)
        b_state = jax.vmap(task.reset)(b_key_init)
        b_steps = jnp.zeros(cfg.n_envs, dtype=jnp.int32)

        # If no ResetIDProvider is provided, use a default reset ID value (-42069).
        if id_provider is None:
            b_reset_id = jnp.full(cfg.n_envs, -42069, dtype=jnp.int32)
        else:
            # Otherwise, use the ResetIDProvider to generate reset IDs.
            id_provider, b_reset_id = id_provider.get_ids(cfg.n_envs)

        collector_state = CollectorState(b_steps, b_reset_id, b_state)
        return Collector(0, key, collector_state, task, cfg), id_provider

    def reset_all(self) -> Self:
        n_envs = self.cfg.n_envs

        key0 = jr.fold_in(self.key, self.collect_idx)
        b_key_init = jr.split(key0, n_envs)

        b_steps = jnp.zeros(n_envs, dtype=jnp.int32)
        b_reset_id = jnp.full(n_envs, -42069, dtype=jnp.int32)
        b_state = jax.vmap(self.task.reset)(b_key_init)
        collect_state_new = CollectorState(b_steps, b_reset_id, b_state)

        # Increment collect_idx to prevent key reuse.
        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state_new)

        return self_new

    def reset_with_state(self, b_state: TimedState) -> Self:
        n_envs = self.cfg.n_envs
        b_steps = jnp.zeros(n_envs, dtype=jnp.int32)
        b_reset_id = jnp.full(n_envs, -42069, dtype=jnp.int32)
        collect_state_new = CollectorState(b_steps, b_reset_id, b_state)
        return self.replace(collect_state=collect_state_new)

    @classmethod
    def create_resetfn(cls, key: PRNGKey, task: JaxTask, reset_fn: ResetFn, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_key_init = jr.split(key_init, cfg.n_envs)
        b_state = reset_fn(b_key_init, cfg.n_envs)
        b_steps = jnp.zeros(cfg.n_envs, dtype=jnp.int32)
        b_reset_id = jnp.arange(cfg.n_envs)
        collector_state = CollectorState(b_steps, b_reset_id, b_state)
        return Collector(0, key, collector_state, task, cfg)

    def split(self, num: int) -> list[Self]:
        assert self.cfg.n_envs % num == 0
        n_env_new = self.cfg.n_envs // num
        cfg_new = copy.deepcopy(self.cfg)
        cfg_new.n_envs = n_env_new

        collectors: list[Collector] = []

        keys = jr.split(self.key, num)

        for ii in range(num):
            start_idx = ii * n_env_new
            end_idx = (ii + 1) * n_env_new

            colstate_new = CollectorState(
                steps=self.collect_state.steps[start_idx:end_idx],
                reset_id=self.collect_state.reset_id[start_idx:end_idx],
                state=jtu.tree_map(lambda x: x[start_idx:end_idx], self.collect_state.state),
            )

            collector_new = Collector(
                collect_idx=self.collect_idx,
                key=keys[ii],
                collect_state=colstate_new,
                task=self.task,
                cfg=cfg_new,
            )
            collectors.append(collector_new)

        return collectors

    def _step_single(
            self, get_pol: GetPolFn, state: CollectorState, key: PRNGKey
    ) -> tuple[TObs, StepOutput, TBool, dict, Control, FloatScalar]:
        # Split the random key into two parts: one for the policy and one for the environment step.
        key_pol, key_step = jr.split(key, 2)

        # Get the observations from the current state of the environment.
        obs_pol = self.task.get_obs(state.state)

        # Call the policy function to get the action distribution or additional information.
        a_pol: DistributionLike
        out = get_pol(obs_pol)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            # If the policy returns a tuple, extract the distribution and additional info.
            a_pol, pol_info = out
            pol_info = {f"Policy/{k}": v for k, v in pol_info.items()}
        else:
            # Otherwise, assume the policy only returns the distribution.
            a_pol = out
            pol_info = {}

        # Sample a control action and its log probability from the policy distribution.
        control, logprob = a_pol.experimental_sample_and_log_prob(seed=key_pol)

        # Step the environment using the sampled control action.
        output: StepOutput
        output = self.task.step(state.state, control)

        # Determine if the environment should reset (due to truncation or termination).
        should_reset = output.trunc | output.term

        # TODO: pol_info is replaced with the info from the environment
        pol_info = output.info

        # Return the observations, step output, reset flag, policy info, control action, and log probability.
        return obs_pol, output, should_reset, pol_info, control, logprob

    def collect_batch_with_buf(
            self, get_pol: GetPolFn, reset_buf: ResetBufLike, rollout_T: int
    ) -> tuple[Self, ResetBufLike, RolloutOutput, dict]:
        def _body(scan_state: tuple[CollectorState, ResetBufLike], key: PRNGKey):
            # Split the key into two parts: one for the policy and one for the reset buffer.
            key_rollout, key_reset = jr.split(key)

            # Split the rollout key into a batch of keys for each environment.
            b_key = jr.split(key_rollout, self.cfg.n_envs)

            # Unpack the current collector state and reset buffer.
            b_colstate, reset_buf_ = scan_state

            # Perform a single step for all environments using the policy function.
            b_output: StepOutput
            b_obs_now, b_output, b_shouldreset, b_info, b_control, b_logprob = jax.vmap(step_single_fn)(
                b_colstate, b_key
            )
            b_obs_next = b_output.obs  # Observations after taking the action.
            b_state_next = b_output.state  # Next state after taking the action.
            b_rew = b_output.rew  # R(s, a, s')
            b_trunc, b_term = (
                b_output.trunc,
                b_output.term,
            )  # Truncation and termination flags.

            # Sample new states from the reset buffer for environments that need to reset.
            reset_buf_, b_statereset_sampled = reset_buf_.sample(key_reset, self.cfg.n_envs, b_valid=b_shouldreset)

            b_state_reset, b_reset_id = (
                b_statereset_sampled.state,
                b_statereset_sampled.reset_id,
            )

            # Update the step counters, states, and reset IDs for all environments.
            # Step counter is reset to 0 for environments that should reset, otherwise incremented.
            b_steps_new = jnp.where(b_shouldreset, 0, b_colstate.steps + 1)
            b_state_new = tree_where_dim0(b_shouldreset, b_state_reset, b_output.state, which=jnp)
            b_reset_id_new = jnp.where(b_shouldreset, b_reset_id, b_colstate.reset_id)

            # Create the updated collector state.
            b_colstate_new = b_colstate._replace(steps=b_steps_new, reset_id=b_reset_id_new, state=b_state_new)

            # Ensure the shapes and data types of the states are consistent.
            leaves1 = jtu.tree_leaves_with_path(b_colstate.state)
            leaves2 = jtu.tree_leaves_with_path(b_state_reset)

            assert len(leaves1) == len(leaves2)
            for (path, leaf1), (path, leaf2) in zip(leaves1, leaves2):
                assert leaf1.shape == leaf2.shape
                assert leaf1.dtype == leaf2.dtype
            # ------------------------------------------

            # Prepare the output for the current timestep.
            b_reset_id = b_colstate.reset_id
            b_state_now = b_colstate.state

            scan_state_new = (b_colstate_new, reset_buf_)
            out = RolloutOutput(
                T_reset_id=b_reset_id,
                T_state_now=self.task.minify(b_state_now),
                T_state_nxt=self.task.minify(b_state_next),
                T_obs_now=b_obs_now,
                T_obs_nxt=b_obs_next,
                T_control=b_control,
                T_logprob=b_logprob,
                T_rew=b_rew,
                T_trunc=b_trunc,
                T_term=b_term,
                T_info=b_info | b_output.info,
            )
            return scan_state_new, out

        # Partially apply the `_step_single` method with the policy function.
        step_single_fn = ft.partial(self._step_single, get_pol)

        # Generate a sequence of random keys for the rollout.
        key0 = jr.fold_in(self.key, self.collect_idx)
        T_key = jr.split(key0, rollout_T)

        # Initialize the scan state with the current collector state and reset buffer.
        scan_state0 = (self.collect_state, reset_buf)

        # Perform the batched rollout using `lax.scan`.
        (collect_state_new, reset_buf_new), Tb_rollout_out = lax.scan(_body, scan_state0, T_key, length=rollout_T)

        # Update the collector instance with the new state.
        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state_new)

        # Rearrange the rollout output from (T, b) to (b, T).
        Tb_output = Tb_rollout_out
        # Convert from (T, b) to (b, T).
        bT_output = jtu.tree_map(lambda x: ei.rearrange(x, "T b ... -> b T ..."), Tb_output)

        # Return the updated collector, reset buffer, rollout data, and additional info.
        collect_info = {}
        return self_new, reset_buf_new, bT_output, collect_info

    def collect_batch(self, get_pol: GetPolFn, rollout_T: int) -> tuple["Collector", RolloutOutput, dict]:
        return self.collect_batch_with_fn(get_pol, EnvResetFn(self.task), rollout_T)

    def minify(self, state_):
        minstate_ = self.task.get_minstate(state_)
        # Go one step further and convert it to a list of leaves, so we can save on the tree construction cost.
        return jtu.tree_leaves(minstate_)

    def collect_batch_with_fn(
            self, get_pol: GetPolFn, reset_fn: ResetFn, rollout_T: int
    ) -> tuple["Collector", RolloutOutput, dict]:
        def split_key(key: PRNGKey):
            return jr.split(key, 2)

        def _body(scan_state: tuple[CollectorState], inp_):
            if self.cfg.pbar:
                _, b_key = inp_
            else:
                b_key = inp_

            (b_colstate,) = scan_state
            # Split the batch key into policy and reset keys for each environment.
            b2_key = jax.vmap(split_key)(b_key)
            b_key_pol, b_key_reset = b2_key[:, 0], b2_key[:, 1]

            # Perform a single step for all environments using the policy function.
            b_output: StepOutput
            b_obs_now, b_output, b_shouldreset, b_info, b_control, b_logprob = jax.vmap(step_single_fn)(
                b_colstate, b_key_pol
            )
            b_obs_next = b_output.obs  # Observations after taking the action.
            b_state_next = b_output.state  # Next state after taking the action.
            b_rew = b_output.rew  # Reward for the transition.
            b_trunc, b_term = (
                b_output.trunc,
                b_output.term,
            )  # Truncation and termination flags.

            # Sample new states from the environment for resets.
            b_state_reset = reset_fn(b_key_reset, self.cfg.n_envs)
            b_reset_id_reset = -42069  # Default reset ID for new states.

            # Update the step counters, states, and reset IDs for all environments.
            b_steps_new = jnp.where(b_shouldreset, 0, b_colstate.steps + 1)
            b_state_new = tree_where_dim0(b_shouldreset, b_state_reset, b_output.state, which=jnp)
            b_reset_id_new = jnp.where(b_shouldreset, b_reset_id_reset, b_colstate.reset_id)

            # Create the updated collector state.
            b_colstate_new = b_colstate._replace(steps=b_steps_new, reset_id=b_reset_id_new, state=b_state_new)

            # Prepare the output for the current timestep.
            b_reset_id = b_colstate.reset_id
            b_state_now = b_colstate.state

            is_obs_arr = isinstance(b_obs_now, (np.ndarray, jnp.ndarray))
            if not is_obs_arr:
                b_obs_now = tuple(jtu.tree_leaves(b_obs_now))
                b_obs_next = tuple(jtu.tree_leaves(b_obs_next))

            scan_state_new = (b_colstate_new,)
            out = RolloutOutput(
                b_reset_id,
                self.task.minify(b_state_now),
                self.task.minify(b_state_next),
                b_obs_now,
                b_obs_next,
                b_control,
                b_logprob,
                b_rew,
                b_trunc,
                b_term,
                b_info | b_output.info,
            )
            return scan_state_new, out

        # Partially apply the `_step_single` method with the policy function.
        step_single_fn = ft.partial(self._step_single, get_pol)

        # Generate a sequence of random keys for the rollout.
        key0 = jr.fold_in(self.key, self.collect_idx)
        Tb_key = ei.rearrange(
            jr.split(key0, rollout_T * self.cfg.n_envs),
            "(T b) ... -> T b ...",
            T=rollout_T,
            b=self.cfg.n_envs,
        )

        # Initialize the scan state with the current collector state.
        scan_state0 = (self.collect_state,)
        T_inp = Tb_key
        if self.cfg.pbar:
            # Wrap the body function with a progress bar if enabled.
            _body = scan_tqdm(rollout_T)(_body)
            T_inp = (np.arange(rollout_T), Tb_key)

        # Perform the batched rollout using `lax.scan`.
        (collect_state_new,), Tb_rollout_out = lax.scan(_body, scan_state0, T_inp, length=rollout_T)
        # (collect_state_new,), Tb_rollout_out = python_scan(_body, scan_state0, T_inp, length=rollout_T)

        # Update the collector instance with the new state.
        self_new = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state_new)

        # Rearrange the rollout output from (T, b) to (b, T).
        Tb_output = Tb_rollout_out
        # Convert from (T, b) to (b, T).
        bT_output = jtu.tree_map(lambda x: ei.rearrange(x, "T b ... -> b T ..."), Tb_output)

        # Return the updated collector, rollout data, and additional info.
        collect_info = {}
        return self_new, bT_output, collect_info
