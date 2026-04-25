import functools as ft

import einops as ei
import ipdb
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm
from og.rng import PRNGKey
from og.tree_utils import tree_where_dim0

from fge.core.bits.collector import Collector, CollectorState, EnvResetFn, RolloutOutput
from fge.core.bits.ppo_core import SumPPO
from fge.core.envs.jax_task import (
    StepOutput,
)


def collect_eval_savemem(
    ppo: SumPPO,
    collector: Collector,
    small_T: int,
    n_vmap_split: int = 1,
    rng: bool = True,
    drop_obs: bool = False,
    compress_state: bool = False,
) -> tuple[Collector.Rollout, dict]:
    """Collect evaluation data, but save memory by not doing the entire rollout all at once on the GPU.

    Instead, split up the rollout into smaller chunks, moving data back to the CPU as we go.
    """
    self = collector
    assert small_T > 0, "small_T must be positive"

    rollout_T = ppo.task.eval_rollout_T

    # ---- Generate all keys for the full rollout ----
    key0 = jr.fold_in(self.key, self.collect_idx)
    Tb_key_all = ei.rearrange(
        jr.split(key0, rollout_T * self.cfg.n_envs),
        "(T b) ... -> T b ...",
        T=rollout_T,
        b=self.cfg.n_envs,
    )

    # ---- Python loop over chunks, JIT inside ----
    n_chunks = (rollout_T + small_T - 1) // small_T

    collect_state = self.collect_state
    Tb_chunks_host = []  # store chunk outputs (each already moved to host)

    pbar = tqdm.trange(n_chunks, desc="Collecting chunked eval rollout")
    for chunk_idx in pbar:
        start = chunk_idx * small_T
        end = min((chunk_idx + 1) * small_T, rollout_T)
        cur_T = end - start
        if cur_T <= 0:
            break

        Tb_key_chunk = Tb_key_all[start:end]  # shape (cur_T, n_envs, ...)

        # Call the compiled scan helper. First call for a given cur_T will compile.
        collect_state, Tb_rollout_out = collect_eval_savemem_inner(
            ppo,
            collector,
            collect_state,
            Tb_key_chunk,
            chunk_T=cur_T,
            n_vmap_split=n_vmap_split,
            drop_obs=drop_obs,
            compress_state=compress_state,
            rng=rng,
        )

        # Move this chunk's outputs to CPU and forget them on device.
        Tb_rollout_out_host: RolloutOutput = jax.device_get(Tb_rollout_out)
        del Tb_rollout_out

        Tb_chunks_host.append(Tb_rollout_out_host)

    # ---- Concatenate all chunks along the T dimension on CPU ----
    Tb_output_host = jtu.tree_map(
        lambda *xs: np.concatenate(xs, axis=0),
        *Tb_chunks_host,
    )

    # Convert from (T, b, ...) to (b, T, ...)
    bT_output_host = jtu.tree_map(
        lambda x: ei.rearrange(x, "T b ... -> b T ..."),
        Tb_output_host,
    )

    # If you prefer JAX arrays on CPU:
    # bT_output = jax.tree_map(jax.device_put, bT_output_host)
    # else just keep NumPy:
    bT_output = bT_output_host

    collect_info = {}
    assert isinstance(bT_output, Collector.Rollout)
    return bT_output, collect_info


@ft.partial(jax.jit, static_argnames=("chunk_T", "rng", "n_vmap_split", "drop_obs", "compress_state"))
def collect_eval_savemem_inner(
    ppo: SumPPO,
    collector: Collector,
    collect_state: CollectorState,
    Tb_key: PRNGKey,
    chunk_T: int,
    n_vmap_split: int,
    drop_obs: bool,
    compress_state: bool,
    rng: bool,
):
    self = collector

    if rng:
        get_pol = ppo.policy.apply
    else:
        get_pol = ppo.policy_det

    reset_fn = EnvResetFn(ppo.task)

    # Partially apply the `_step_single` method with the policy function.
    step_single_fn = ft.partial(self._step_single, get_pol)

    def split_key(key: PRNGKey):
        return jr.split(key, 2)

    def _body(scan_state: tuple[CollectorState], inp_):
        b_key = inp_

        (b_colstate,) = scan_state
        # Split the batch key into policy and reset keys for each environment.
        b2_key = jax.vmap(split_key)(b_key)
        b_key_pol, b_key_reset = b2_key[:, 0], b2_key[:, 1]

        # Perform a single step for all environments using the policy function.
        b_output: StepOutput
        if n_vmap_split == 1:
            b_obs_now, b_output, b_shouldreset, b_info, b_control, b_logprob = jax.vmap(step_single_fn)(
                b_colstate, b_key_pol
            )
        else:

            def wrapper(inp):
                b_colstate_, b_key_pol_ = inp
                return step_single_fn(b_colstate_, b_key_pol_)

            b = len(b_key)
            map_batch_size = b // n_vmap_split
            assert n_vmap_split * map_batch_size == b, "n_vmap_split must evenly divide batch size"

            b_inp = b_colstate, b_key_pol
            b_obs_now, b_output, b_shouldreset, b_info, b_control, b_logprob = jax.lax.map(
                wrapper, b_inp, batch_size=map_batch_size
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

        n_env = len(b_term)

        if drop_obs:
            b_obs_now = b_obs_next = jnp.zeros((n_env, 0))

        if compress_state:
            b_state_now = jax.vmap(self.task.compress_state)(b_state_now)
            b_state_next = jax.vmap(self.task.compress_state)(b_state_next)

        b_state_now_out = self.task.minify(b_state_now)
        b_state_next_out = self.task.minify(b_state_next)

        scan_state_new = (b_colstate_new,)
        out = RolloutOutput(
            b_reset_id,
            b_state_now_out,
            b_state_next_out,
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

    scan_state0 = (collect_state,)
    (collect_state_new,), Tb_rollout_out = lax.scan(_body, scan_state0, Tb_key, length=chunk_T)

    return collect_state_new, Tb_rollout_out
