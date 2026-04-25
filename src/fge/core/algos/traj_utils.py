import functools as ft

import jax
import numpy as np
from jax import numpy as jnp
from og.jax_utils import jax2np
from og.tree_utils import tree_cat, tree_stack

from fge.core.bits.collector import Collector, RolloutOutput
from fge.core.bits.collector_savemem import collect_eval_savemem
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.ppo_core import SumPPO
from fge.core.utils.jax_util import myjit


def split_trajs(b_rollout: RolloutOutput) -> list[RolloutOutput]:
    """Given a batch of rollouts, split them into individual rollouts that terminate or run the entire length."""
    if isinstance(b_rollout.T_rew, jnp.ndarray):
        # Put it on CPU.
        b_rollout = jax2np(b_rollout)

    b, T = b_rollout.T_rew.shape
    bT_isfinal = b_rollout.T_trunc | b_rollout.T_term

    rollouts: list[RolloutOutput] = []
    for bb in range(b):
        T_isfinal = bT_isfinal[bb]

        # rollout: RolloutOutput = jtu.tree_map(lambda x: x[bb], b_rollout)
        rollout: RolloutOutput = b_rollout.tree_index(bb)

        if not np.any(T_isfinal):
            # No terminate, just add the full rollout.
            rollouts.append(rollout)
            continue

        # There is a terminate.
        idxs_done = np.where(T_isfinal)[0]
        assert len(idxs_done) > 0
        idx_end = idxs_done[0]
        sl = slice(0, idx_end + 1)
        # rollout_segment: RolloutOutput = jtu.tree_map(lambda x: x[: idx_end + 1], rollout)
        rollout_segment: RolloutOutput = rollout.tree_index(sl)
        rollouts.append(rollout_segment)

    return rollouts


def get_rollout_summary(rollout: RolloutOutput):
    T_rew = rollout.T_rew
    T_done = rollout.T_trunc | rollout.T_term

    (T,) = T_rew.shape
    first_done = jnp.argmax(T_done)
    has_done = jnp.any(T_done)
    T_valid = jnp.arange(T) <= first_done
    T_valid = jnp.where(has_done, T_valid, True)

    rew_sum = jnp.sum(T_rew * T_valid)
    steps_alive = jnp.where(has_done, first_done, T)

    return rew_sum, steps_alive


def get_rollout_summary_np(b_rollout: RolloutOutput):
    bT_rew = b_rollout.T_rew
    bT_done = b_rollout.T_trunc | b_rollout.T_term

    (b, T) = bT_rew.shape
    b_first_done = np.argmax(bT_done, axis=1)
    b_has_done = np.any(bT_done, axis=1)
    bT_valid = np.arange(T) <= b_first_done[:, None]
    bT_valid = np.where(b_has_done[:, None], bT_valid, True)

    b_rew_sum = np.sum(bT_rew * bT_valid, axis=1)
    b_steps_alive = np.where(b_has_done, b_first_done, T)

    assert b_rew_sum.shape == b_steps_alive.shape == (b,)

    return b_rew_sum, b_steps_alive


@ft.partial(myjit, static_argnames=("ret_rollout",))
def rollout_and_get_summary(ppo, collector, ret_rollout: bool = False):
    b_rollouts: RolloutOutput
    b_rollouts, _ = ppo.collect_eval_det(collector)
    b_rew_sum, b_steps_alive = jax.vmap(get_rollout_summary)(b_rollouts)

    if ret_rollout:
        return b_rew_sum, b_steps_alive, b_rollouts.T_state_now
    else:
        return b_rew_sum, b_steps_alive


def rollout_and_get_summary_savemem(
        ppo: SumPPO | IntrinsicSumPPO, collector: Collector, small_T: int, n_outer: int = 1, n_vmap_split: int = 1,
        ret_rollout: bool = False
):
    if isinstance(ppo, IntrinsicSumPPO):
        ppo = ppo.ppo

    if n_outer == 1:
        b_rollouts: RolloutOutput
        b_rollouts, _ = collect_eval_savemem(ppo, collector, small_T, n_vmap_split=n_vmap_split, rng=False)
        b_rew_sum, b_steps_alive = get_rollout_summary_np(b_rollouts)

        if ret_rollout:
            return b_rew_sum, b_steps_alive, b_rollouts.T_state_now
        else:
            return b_rew_sum, b_steps_alive

    else:
        # Split up collector into n chunks, run collect_eval_savemem on each chunk, then aggregate results.
        c_collectors = collector.split(n_outer)

        outs = []
        T_state_nows = []
        for ii, collector in enumerate(c_collectors):
            b_rollouts: RolloutOutput
            b_rollouts, _ = collect_eval_savemem(
                ppo, collector, small_T, n_vmap_split=n_vmap_split, rng=False, drop_obs=True, compress_state=True
            )
            b_rew_sum, b_steps_alive = get_rollout_summary_np(b_rollouts)

            if ret_rollout:
                outs.append((b_rew_sum, b_steps_alive))
                T_state_nows.append(b_rollouts.T_state_now)
            else:
                outs.append((b_rew_sum, b_steps_alive))

            del b_rollouts

        b_rew_sums, b_steps_alives = zip(*outs)
        b_rew_sum = np.concatenate(b_rew_sums, axis=0)
        b_steps_alive = np.concatenate(b_steps_alives, axis=0)

        if ret_rollout:
            T_state_nows = tree_cat(T_state_nows, axis=0)
            return b_rew_sum, b_steps_alive, T_state_nows
        else:
            return b_rew_sum, b_steps_alive
