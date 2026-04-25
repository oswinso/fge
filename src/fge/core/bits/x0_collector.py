import functools as ft
from typing import Protocol

import einops as ei
import equinox as eqx
import jax
from equinox.internal._loop.common import _Buffer
from flax import struct
from jax import dtypes
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from og.dyn_types import BControl, BObs, TBBool, TBFloat, TBool, TFloat
from og.jax_types import BFloat, IntScalar
from og.rng import PRNGKey
from og.tree_utils import make_batch_pytree

from fge.core.bits.collector import (
    CollectorCfg,
    CollectorState,
    DistributionLike,
    GetPolFn,
)
from fge.core.envs.jax_task import JaxTask, StepOutput, TimedState, TreeLeaves


@struct.dataclass(slots=True, frozen=False)
class X0RolloutOutput:
    """
    Data structure to store the output of a rollout.

    Attributes:
        T_state_now (TreeLeaves): Current states at each timestep.
        T_state_nxt (TreeLeaves): Next states at each timestep.
        T_rew (TFloat): Rewards at each timestep.
        T_trunc (TBool): Truncation flags at each timestep.
        T_term (TBool): Termination flags at each timestep.
        T_info (dict): Additional information about the rollout.
    """

    T_state_now: TreeLeaves
    T_state_nxt: TreeLeaves
    T_rew: TFloat
    T_trunc: TBool
    T_term: TBool
    T_info: dict


@struct.dataclass(slots=True, frozen=False)
class X0Data:
    """
    Data structure to summarize the initial state and rollout.

    Attributes:
        b_obs (BObs): Batch of observations.
        b_control (BControl): Batch of controls.
        b_logprob (BFloat): Log probabilities of the controls.
        b_rew (BFloat): Batch of rewards.
    """

    b_obs: BObs
    b_control: BControl
    b_logprob: BFloat
    b_rew: BFloat


class LoopState(struct.PyTreeNode):
    """
    Represents the state of the rollout loop.

    Attributes:
        steps (IntScalar): Current step count.
        state (TimedState): Current environment state.
        b_valid (TBool): Validity flags for each environment.
        Tb_state_now (TreeLeaves): Buffer for current states.
        Tb_state_nxt (TreeLeaves): Buffer for next states.
        Tb_rew (TBFloat): Buffer for rewards.
        Tb_trunc (TBBool): Buffer for truncation flags.
        Tb_term (TBBool): Buffer for termination flags.
    """

    steps: IntScalar
    state: TimedState
    b_valid: TBool
    Tb_state_now: TreeLeaves
    Tb_state_nxt: TreeLeaves
    Tb_rew: TBFloat
    Tb_trunc: TBBool
    Tb_term: TBBool


class GetX0ControlFn(Protocol):
    """
    Protocol for a function that generates control distributions for initial states.

    Returns:
        DistributionLike | tuple[DistributionLike, dict]: Control distribution or a tuple with additional info.
    """

    def __call__(self) -> DistributionLike | tuple[DistributionLike, dict]: ...


class X0Collector(struct.PyTreeNode):
    """
    Collector for generating rollouts starting from adversarially sampled initial states.

    Attributes:
        collect_idx (int): Index of the current collection.
        key (PRNGKey): Random number generator key.
        discount_gamma (float): Discount factor for rewards.
        task (JaxTask): Task/environment to interact with.
        cfg (CollectorCfg): Configuration for the collector.
    """

    Cfg = CollectorCfg

    collect_idx: int
    key: PRNGKey
    discount_gamma: float
    task: JaxTask = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, key: PRNGKey, task: JaxTask, cfg: CollectorCfg, discount_gamma: float
    ):
        """
        Factory method to create an X0Collector instance.

        Args:
            key (PRNGKey): Random number generator key.
            task (JaxTask): Task/environment to interact with.
            cfg (CollectorCfg): Configuration for the collector.
            discount_gamma (float): Discount factor for rewards.

        Returns:
            X0Collector: A new instance of the collector.
        """
        if dtypes.issubdtype(key.dtype, dtypes.prng_key):
            key = jr.key_data(key)

        return X0Collector(0, key, discount_gamma, task, cfg)

    def _step_single(self, get_pol: GetPolFn, state: CollectorState, key: PRNGKey):
        """
        Perform a single step in the environment.

        Args:
            get_pol (GetPolFn): Function to get the policy distribution.
            state (CollectorState): Current state of the collector.
            key (PRNGKey): Random number generator key.

        Returns:
            tuple: Observations, step output, reset flags, policy info, control, and log probabilities.
        """
        key_pol, key_step = jr.split(key, 2)
        obs_pol = self.task.get_obs(state.state)
        a_pol: DistributionLike
        out = get_pol(obs_pol)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            a_pol, pol_info = out
            pol_info = {f"Policy/{k}": v for k, v in pol_info.items()}
        else:
            a_pol = out
            pol_info = {}

        control, logprob = a_pol.experimental_sample_and_log_prob(seed=key_pol)

        output: StepOutput
        output = self.task.step(state.state, control)

        should_reset = output.trunc | output.term

        return obs_pol, output, should_reset, pol_info, control, logprob

    def sample_state0_logprob(self, key: PRNGKey, get_x0_control_dist):
        """
        Sample initial states and their log probabilities.

        Args:
            key (PRNGKey): Random number generator key.
            get_x0_control_dist (GetX0ControlFn): Function to get control distributions.

        Returns:
            tuple: Initial states, controls, and log probabilities.
        """
        b_key_x0 = jr.split(key, self.cfg.n_envs)
        dist = get_x0_control_dist()

        def sample_fn(key_):
            return dist.experimental_sample_and_log_prob(seed=key_)

        b_x0_control, b_logprob = jax.vmap(sample_fn)(b_key_x0)
        assert b_x0_control.shape[0] == self.cfg.n_envs

        # x0 controls -> x0
        b_state0 = jax.vmap(self.task.reset_from_box)(b_x0_control)

        return b_state0, b_x0_control, b_logprob

    def collect_batch(
        self, get_pol: GetPolFn, get_x0_control_dist: GetX0ControlFn, rollout_T: int
    ):
        """
        Collect a batch of rollouts starting from adversarially sampled initial states.

        Args:
            get_pol (GetPolFn): Function to get the policy distribution.
            get_x0_control_dist (GetX0ControlFn): Function to get control distributions.
            rollout_T (int): Maximum rollout length.

        Returns:
            tuple: Updated collector, summarized data, and collection info.
        """

        def _cond_fn(st: LoopState):
            # Keep going if there's at least one valid environment.
            return jnp.any(st.b_valid[:])

        def _body_fn(st: LoopState) -> LoopState:
            """
            Perform a single step in the rollout loop.

            Args:
                st (LoopState): Current loop state.

            Returns:
                LoopState: Updated loop state.
            """
            key_thisstep = jr.fold_in(key_step, st.steps)
            b_key = jr.split(key_thisstep, self.cfg.n_envs)
            b_output: StepOutput
            b_reset_id = jnp.zeros(self.cfg.n_envs, dtype=jnp.int32)
            b_step = jnp.full(self.cfg.n_envs, st.steps, dtype=jnp.int32)
            b_colstate = CollectorState(b_step, b_reset_id, st.state)
            b_obs_now, b_output, b_shouldreset, b_info, b_control, _ = jax.vmap(
                step_single_fn
            )(b_colstate, b_key)
            b_state_next_ = b_output.state
            b_rew = b_output.rew
            b_trunc, b_term = b_output.trunc, b_output.term
            b_nextvalid = ~(b_trunc | b_term)

            b_valid_next = st.b_valid & b_nextvalid

            def is_leaf(val):
                return isinstance(val, _Buffer)

            def tree_set(tree_, treeval):
                return jtu.tree_map(
                    lambda x, y: x.at[st.steps].set(y), tree_, treeval, is_leaf=is_leaf
                )

            b_state_now_ = b_colstate.state
            Tb_state_now = tree_set(st.Tb_state_now, self.task.minify(b_state_now_))
            Tb_state_nxt = tree_set(st.Tb_state_nxt, self.task.minify(b_state_next_))
            Tb_rew = tree_set(st.Tb_rew, b_rew)
            Tb_trunc = tree_set(st.Tb_trunc, b_trunc)
            Tb_term = tree_set(st.Tb_term, b_term)

            steps_new = st.steps + 1

            return LoopState(
                steps_new,
                b_state_next_,
                b_valid_next,
                Tb_state_now,
                Tb_state_nxt,
                Tb_rew,
                Tb_trunc,
                Tb_term,
            )

        def _buffers(st: LoopState):
            """
            Extract buffers from the loop state.

            Args:
                st (LoopState): Current loop state.

            Returns:
                tuple: Buffers for states, rewards, and flags.
            """
            return st.Tb_state_now, st.Tb_state_nxt, st.Tb_rew, st.Tb_trunc, st.Tb_term

        step_single_fn = ft.partial(self._step_single, get_pol)

        key0 = jr.fold_in(self.key, self.collect_idx)
        key_x0, key_step = jr.split(key0, 2)
        b_state0, b_x0_control, b_logprob = self.sample_state0_logprob(
            key_x0, get_x0_control_dist
        )

        b_state0_leaf = self.task.minify(b_state0)

        b_valid0 = jnp.full(self.cfg.n_envs, True, dtype=bool)

        Tb_state_now = make_batch_pytree(b_state0_leaf, self.cfg.n_envs, fill_value=0)
        Tb_state_nxt = make_batch_pytree(b_state0_leaf, self.cfg.n_envs, fill_value=0)

        Tb_rew = jnp.zeros((rollout_T, self.cfg.n_envs), dtype=jnp.float32)
        Tb_trunc = jnp.zeros((rollout_T, self.cfg.n_envs), dtype=bool)
        Tb_term = jnp.zeros((rollout_T, self.cfg.n_envs), dtype=bool)
        loopstate0 = LoopState(
            0, b_state0, b_valid0, Tb_state_now, Tb_state_nxt, Tb_rew, Tb_trunc, Tb_term
        )
        loopstate = eqx.internal.while_loop(
            _cond_fn,
            _body_fn,
            loopstate0,
            max_steps=self.cfg.n_envs,
            buffers=_buffers,
            kind="lax",
        )

        Tb_output = X0RolloutOutput(
            loopstate.Tb_state_now,
            loopstate.Tb_state_nxt,
            loopstate.Tb_rew,
            loopstate.Tb_trunc,
            loopstate.Tb_term,
            None,
        )
        bT_output = jtu.tree_map(
            lambda x: ei.rearrange(x, "T b ... -> b T ..."), Tb_output
        )

        summarize_fn = ft.partial(summarize_rollout, discount_gamma=self.discount_gamma)
        b_x0_data, info = jax.vmap(summarize_fn)(
            b_x0_control, b_logprob, b_state0, bT_output
        )

        self_new = self.replace(collect_idx=self.collect_idx + 1)
        collect_info = info | {"steps": loopstate.steps}
        return self_new, b_x0_data, collect_info


def get_discounted_reward(T_rew, T_done, discount_gamma: float):
    """
    Compute the discounted reward for a trajectory.

    Args:
        T_rew (array): Rewards at each timestep.
        T_done (array): Done flags at each timestep.
        discount_gamma (float): Discount factor.

    Returns:
        float: Discounted reward sum.
    """
    (T,) = T_rew.shape
    first_done = jnp.argmax(T_done)
    has_done = jnp.any(T_done)
    T_valid = jnp.arange(T) <= first_done
    T_valid = jnp.where(has_done, T_valid, True)

    T_discount = discount_gamma ** jnp.arange(T)
    rew_sum = jnp.sum(T_discount * T_rew * T_valid)

    return rew_sum


def summarize_rollout(
    x0_control, logprob, state0, T_data: X0RolloutOutput, discount_gamma: float
) -> tuple[X0Data, dict]:
    """
    Summarize the rollout data.

    Args:
        x0_control: Initial control inputs.
        logprob: Log probabilities of the controls.
        state0: Initial state.
        T_data (X0RolloutOutput): Rollout data.
        discount_gamma (float): Discount factor.

    Returns:
        tuple: Summarized data and additional info.
    """
    obs = jnp.zeros(1)

    # Compute the discounted reward. Mask out anything after the first trunc/term.
    T_rew = T_data.T_rew
    T_done = T_data.T_trunc | T_data.T_term
    rew_sum = get_discounted_reward(T_rew, T_done, discount_gamma)

    x0_data = X0Data(obs, x0_control, logprob, rew_sum)
    info = {
        "x0": state0,
        "T_state_now": T_data.T_state_now,
        "T_trunc": T_data.T_trunc,
        "T_term": T_data.T_term,
        "T_info": T_data.T_info,
    }

    return x0_data, info
