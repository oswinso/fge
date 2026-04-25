import functools as ft
from collections import defaultdict
from typing import Self

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from attrs import define
from flax import struct
from og.jax_types import BBool, BFloat, BInt
from og.jax_utils import jax2np, jax_vmap
from og.rng import PRNGKey
from og.tree_utils import make_batch_pytree, tree_where_dim0

from fge.core.bits.collector import RolloutOutput
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import StateResetId
from fge.core.envs.jax_task import JaxTask, TreeLeaves, leaf_index, leaf_set
from fge.core.utils.jax_util import myjit


@define
class PLRSamplerCfg:
    score_ema_alpha: float = 1.0

    plr_buf_size: int = 8192
    """How large the buffer (on GPU) should be."""

    score_temperature: float = 0.1

    staleness_coef: float = 0.1
    staleness_temperature: float = 1.0

    p_sample_base: float = 0.1


class PLRBuf(struct.PyTreeNode):
    b_data: StateResetId
    b_score_weight: BFloat
    b_staleness: BInt
    size: int

    n_sampled_buf: int
    """Keep track of how many items were sampled, so we can increment the staleness of everything not in the buffer."""

    p_sample_base: float

    id_provider: ResetIDProvider

    capacity: int = struct.field(pytree_node=False)
    task: JaxTask = struct.field(pytree_node=False)
    cfg: PLRSamplerCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        capacity: int,
        item: StateResetId,
        task: JaxTask,
        cfg: PLRSamplerCfg,
        id_provider: ResetIDProvider,
    ) -> Self:
        b_data = make_batch_pytree(item, capacity, fill_value=0, whichnp=jnp)
        b_score_weight = jnp.zeros(capacity, dtype=jnp.float32)
        b_staleness = jnp.zeros(capacity, dtype=jnp.int32)

        return cls(
            b_data=b_data,
            b_score_weight=b_score_weight,
            b_staleness=b_staleness,
            size=0,
            p_sample_base=cfg.p_sample_base,
            n_sampled_buf=0,
            id_provider=id_provider,
            capacity=capacity,
            task=task,
            cfg=cfg,
        )

    def sample(self, key: PRNGKey, num: int, b_valid: BBool | None = None):
        key_bernoulli, key_sample_base, key_buffer = jr.split(key, 3)

        b_sample_base = jr.bernoulli(key_bernoulli, self.p_sample_base, shape=(num,))

        # If the buffer is empty (first iteration), then sample everything from base.
        size_zero = self.size == 0
        b_sample_base = jnp.where(size_zero, True, b_sample_base)
        b_sample_buf = ~b_sample_base

        b_key_sample = jr.split(key_sample_base, num)
        b_state_ = jax.vmap(self.task.reset)(b_key_sample)

        b_base_valid = b_sample_base & b_valid
        id_provider, b_reset_id_ = self.id_provider.get_masked_ids(b_base_valid)
        b_statetup_base = StateResetId(b_state_, b_reset_id_)

        # 1: Compute the staleness weight.
        eps = 1e-3
        b_staleness_weights = (self.b_staleness + eps) ** (
            1.0 / self.cfg.staleness_temperature
        )
        b_staleness_weights = b_staleness_weights / jnp.sum(b_staleness_weights)

        # 2: Compute the total weight.
        staleness_coef = self.cfg.staleness_coef
        b_weights = (
            1 - staleness_coef
        ) * self.b_score_weight + staleness_coef * b_staleness_weights

        # Make sure that b_weights is zeroed out for the invalid items.

        b_sizevalid_mask = jnp.arange(self.capacity) < self.size
        b_weights = jnp.where(b_sizevalid_mask, b_weights, 0.0)
        weights_sum = jnp.sum(b_weights)
        weights_sum = jnp.where(weights_sum < 1e-12, 1.0, weights_sum)
        b_weights = b_weights / weights_sum

        # 3: Sample from the buffer.
        b_idxs = jr.choice(
            key_buffer, self.capacity, shape=(num,), replace=True, p=b_weights
        )

        # The state here is a leaf. Convert back to the full state.
        b_statetup_buf: StateResetId = jtu.tree_map(lambda x: x[b_idxs], self.b_data)
        # b_state_buf = jax.vmap(self.task.leaf_to_state)(b_statetup_buf.state)
        # b_statetup_buf = b_statetup_buf._replace(state=b_state_buf)

        b_statetup: StateResetId = tree_where_dim0(
            b_sample_base, b_statetup_base, b_statetup_buf
        )

        b_issampled = jnp.full(self.capacity, fill_value=False, dtype=bool)
        b_issampled = b_issampled.at[b_idxs].set(b_valid)

        # 4: Increase the staleness of all items by n_sampled. Set the sampled items to 0.
        n_sampled_buf = (b_sample_buf & b_valid).sum()
        b_staleness_new = jnp.where(b_issampled, 0, self.b_staleness + n_sampled_buf)
        n_sampled_buf_total = self.n_sampled_buf + n_sampled_buf

        self_new = self.replace(
            b_staleness=b_staleness_new,
            n_sampled_buf=n_sampled_buf_total,
            id_provider=id_provider,
        )
        return self_new, b_statetup

    def set_data(
        self, b_data: StateResetId, b_score_weight: BFloat, b_staleness: BInt, size: int
    ):
        assert b_score_weight.shape == self.b_score_weight.shape
        assert b_staleness.shape == self.b_staleness.shape
        assert not isinstance(b_data.state, tuple)
        return self.replace(
            b_data=b_data,
            b_score_weight=b_score_weight,
            b_staleness=b_staleness,
            size=size,
            n_sampled_buf=0,
        )


class PLRSampler:
    """A sampler that uses Prioritized Level Replay (PLR)"""

    Cfg = PLRSamplerCfg

    def __init__(
        self,
        task: JaxTask,
        cfg: PLRSamplerCfg,
        reset_id_provider: ResetIDProvider,
        seed: int = 1234567,
    ):
        self.task = task
        self.cfg = cfg

        # n_init = 1 << 16
        n_init = self.cfg.plr_buf_size

        state = task.reset(jr.PRNGKey(0))
        statetup = StateResetId(state, np.array(-1, dtype=np.int32))
        buffer = PLRBuf.create(
            self.cfg.plr_buf_size, statetup, task, cfg, reset_id_provider
        )
        gpu_device = jax.devices("gpu")[0]

        self._buffer: PLRBuf = jax.device_put(buffer, gpu_device)

        leaf = task.minify(state)

        self.b_scores: np.ndarray = np.zeros(n_init, dtype=np.float32)
        """Stores the score for each seed = reset_idx"""
        self.b_staleness: np.ndarray = np.zeros(n_init, np.int32)
        self.b_valid = np.zeros(n_init, bool)
        self.b_x0 = make_batch_pytree(leaf, n_init, fill_value=0, whichnp=np)
        # b_px0, _, _ = task.get_eval_contour()
        # b_ic_shape = (n_init, b_px0.shape[-1])
        # self.b_ic = np.zeros(b_ic_shape, dtype=np.float32)

        ####### initialization of b_ic #########
        sample_ic = np.atleast_1d(self.task.to_icval(self.task.leaf_to_minstate(leaf)))
        ic_dim = sample_ic.shape[0]
        self.b_ic = np.zeros((n_init, ic_dim), dtype=np.float32)
        ######################################
        self.n_valid_x0 = 0

        self.rng = np.random.default_rng(seed=seed)

        self.buf_reset_id: np.ndarray = np.array([], dtype=np.int32)
        self.buf_size = 0

        self.argmax_ics = []

    @property
    def buffer(self) -> PLRBuf:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: PLRBuf):
        self._buffer = buffer

    @ft.partial(myjit, static_argnums=0)
    def get_bT_score(self, ppo: SumPPO, b_rollout: RolloutOutput):
        bT_batch, _ = ppo.make_bT_dset(b_rollout)

        bT_Vl_now = jax_vmap(ppo.Vl_ext.apply, rep=2)(b_rollout.T_obs_now).squeeze(-1)
        bT_Q_gae = bT_batch.b_Ql

        bT_score = jnp.abs(bT_Q_gae - bT_Vl_now)

        return bT_score

    def process_rollout(self, ppo: SumPPO, b_rollout: RolloutOutput):
        """Called after rollout."""
        nbatch, T = b_rollout.T_rew.shape

        # 1: Compute the score.
        bT_score = np.array(self.get_bT_score(ppo, b_rollout))

        # 2: Compute the mean of the score for each reset_id.
        bT_isfinal = np.array(b_rollout.T_trunc) | np.array(b_rollout.T_term)

        bT_reset_id = np.array(b_rollout.T_reset_id)
        bT_state_now = jax2np(b_rollout.T_state_now)

        scores_dict: dict[int, np.floatingi] = defaultdict(float)
        len_dict: dict[int, int] = defaultdict(int)

        reset_id_max = np.max(bT_reset_id)
        self.maybe_resize_scores(reset_id_max)

        info = {"len_b_scores": len(self.b_scores)}

        for bb in range(nbatch):
            T_isfinal = bT_isfinal[bb]
            idxs_done = np.where(T_isfinal)[0]

            # Get the start and end indices.
            idx_start = 0
            for idx_end in idxs_done:
                reset_id: int = bT_reset_id[bb, idx_start]

                score_mean = np.mean(bT_score[bb, idx_start : idx_end + 1])
                traj_len = idx_end - idx_start + 1
                len_total = len_dict[reset_id] + traj_len

                # Combine the mean. There may be multiple trajectories with the same reset_id.
                scores_dict[reset_id] += (
                    (score_mean - scores_dict[reset_id]) * traj_len / len_total
                )
                len_dict[reset_id] = len_total

                # If we haven't seen this before, then set staleness to 0.
                if not self.b_valid[reset_id]:
                    index_start = (bb, idx_start)
                    x0_leaf = leaf_index(bT_state_now, index_start)
                    self.b_valid[reset_id] = True
                    self.b_staleness[reset_id] = 0
                    leaf_set(self.b_x0, reset_id, x0_leaf)
                    ic = self.task.to_icval(self.task.leaf_to_minstate(x0_leaf))
                    self.b_ic[reset_id, :] = ic
                    self.n_valid_x0 += 1

                idx_start = idx_end + 1

            # Handle the end.
            if idx_start < T:
                reset_id: int = bT_reset_id[bb, idx_start]

                score_mean = np.mean(bT_score[bb, idx_start:])
                traj_len = T - idx_start
                len_total = len_dict[reset_id] + traj_len

                scores_dict[reset_id] += (
                    (score_mean - scores_dict[reset_id]) * traj_len / len_total
                )
                len_dict[reset_id] = len_total

                if not self.b_valid[reset_id]:
                    index_start = (bb, idx_start)
                    x0_leaf = leaf_index(bT_state_now, index_start)
                    self.b_valid[reset_id] = True
                    self.b_staleness[reset_id] = 0
                    leaf_set(self.b_x0, reset_id, x0_leaf)
                    ic = self.task.to_icval(self.task.leaf_to_minstate(x0_leaf))
                    self.b_ic[reset_id] = ic
                    self.n_valid_x0 += 1

        # Update the staleness.
        self.update_staleness()

        # Updated the scores with EMA.
        self.update_scores_with_ema(scores_dict)
        self.update_sampling_buf(self.b_scores, self.b_staleness)

        return info

    def update_staleness(self):
        if self.buf_size == 0:
            return

        n_sampled_buf = int(self.buffer.n_sampled_buf)
        b_staleness = np.array(self.buffer.b_staleness)

        # The staleness of everything increseas by n_sampled_buf
        self.b_staleness[:] += n_sampled_buf
        # Update the staleness with the reset_ids on the buffer.
        self.b_staleness[self.buf_reset_id] = b_staleness[: self.buf_size]

    def maybe_resize_scores(self, reset_id_max: int):
        size_needed = reset_id_max + 10
        if len(self.b_scores) >= size_needed:
            return

        # Keep doubling the array size until it is large enough
        cur_len = len(self.b_scores)
        # Smallest power of 2 greater than reset_id_max.
        tgt_len = 1 << int(np.ceil(np.log2(size_needed)))
        len_to_add = tgt_len - cur_len
        self.b_scores = np.concatenate([self.b_scores, np.zeros(len_to_add)])
        self.b_staleness = np.concatenate(
            [self.b_staleness, np.zeros(len_to_add, dtype=np.int32)]
        )
        self.b_valid = np.concatenate(
            [self.b_valid, np.full(len_to_add, False, dtype=bool)]
        )
        self.b_ic = np.concatenate(
            [self.b_ic, np.zeros((len_to_add, self.b_ic.shape[1]), dtype=np.float32)]
        )

        # Resize the x0 buffer.
        arrs = []
        for ii, arr in enumerate(self.b_x0):
            shape = (len_to_add,) + arr.shape[1:]
            arr_new = np.concatenate([arr, np.zeros(shape, dtype=arr.dtype)])
            arrs.append(arr_new)
        self.b_x0 = tuple(arrs)

        assert len(self.b_scores) >= reset_id_max

    def update_scores_with_ema(self, scores_dict: dict[int, np.floating]):
        reset_id_max = max(scores_dict.keys())
        self.maybe_resize_scores(reset_id_max)

        alpha = self.cfg.score_ema_alpha

        for reset_id, score in scores_dict.items():
            self.b_scores[reset_id] = (1 - alpha) * self.b_scores[
                reset_id
            ] + alpha * score

    def _score_weights(self, b_seed_scores: np.ndarray):
        # # lmao the original code was doing something very weird...
        # temp = np.flip(np.argsort(b_seed_scores))
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(len(temp)) + 1

        order = np.argsort(b_seed_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(ranks), 0, -1)

        weights = 1 / ranks ** (1.0 / self.cfg.score_temperature)
        return weights

    def _staleness_weights(self, b_staleness: np.ndarray):
        eps = 1e-3
        weights = (b_staleness + eps) ** (1.0 / self.cfg.staleness_temperature)

        # Give everything that is invalid a weight of zero.
        weights = np.where(self.b_valid, weights, 0.0)

        return weights

    def get_weights(self, b_seed_scores: np.ndarray, b_staleness: np.ndarray):
        b_score_weights = self._score_weights(b_seed_scores)

        z = np.sum(b_score_weights)
        if z > 0:
            b_score_weights /= z

        b_staleness_weights = self._staleness_weights(b_staleness)
        z = np.sum(b_staleness_weights)
        if z > 0:
            b_staleness_weights /= z

        staleness_coef = self.cfg.staleness_coef
        b_weights = (
            1 - staleness_coef
        ) * b_score_weights + staleness_coef * b_staleness_weights
        return b_weights, b_score_weights

    def update_sampling_buf(self, b_seed_scores: np.ndarray, b_staleness: np.ndarray):
        b_weights, b_score_weights = self.get_weights(b_seed_scores, b_staleness)
        n_resetid = self.n_valid_x0

        # Note: it's possible that b_valid is False in between Trues. We don't want to sample those.

        # Store the ic of the highest score for viz.
        ii_argmax = np.argmax(self.b_scores)
        self.argmax_ics.append(self.b_ic[ii_argmax])

        def mychoice(rng: np.random.Generator, a, size: int, replace: bool, p):
            return rng.choice(a, size, replace=replace, p=p)

        if n_resetid < self.cfg.plr_buf_size:
            # Do a weighted subsample on CPU to put on GPU.
            # b_reset_ids_orig = self.rng.choice(len(b_weights), size=n_resetid, replace=False, p=b_weights)
            b_reset_ids_orig = mychoice(
                self.rng, len(b_weights), size=n_resetid, replace=False, p=b_weights
            )

            # Pad b_reset_ids, but have the weights zeroed out, so its of length plr_buf_size.
            n_pad = self.cfg.plr_buf_size - n_resetid
            b_reset_ids = np.concatenate(
                [b_reset_ids_orig, np.full(n_pad, 0, dtype=np.int32)]
            )

            b_x0_sample = leaf_index(self.b_x0, b_reset_ids)

            b_score_weights_sampled = b_score_weights[b_reset_ids_orig]
            b_staleness_sampled = b_staleness[b_reset_ids_orig]

            # Pad score_weights and staleness.
            b_score_weights_sampled = np.concatenate(
                [b_score_weights_sampled, np.zeros(n_pad)]
            )
            b_staleness_sampled = np.concatenate(
                [b_staleness_sampled, np.zeros(n_pad, dtype=np.int32)]
            )
        else:
            # Do a weighted subsample on CPU to put on GPU.
            # b_reset_ids_orig = self.rng.choice(len(b_weights), size=self.cfg.plr_buf_size, replace=False, p=b_weights)
            b_reset_ids_orig = mychoice(
                self.rng,
                len(b_weights),
                size=self.cfg.plr_buf_size,
                replace=False,
                p=b_weights,
            )
            b_reset_ids = b_reset_ids_orig

            b_x0_sample = leaf_index(self.b_x0, b_reset_ids)

            b_score_weights_sampled = b_score_weights[b_reset_ids]
            b_staleness_sampled = b_staleness[b_reset_ids]

        assert self.b_valid[b_reset_ids].all()

        size = min(n_resetid, self.cfg.plr_buf_size)
        assert (
            b_score_weights_sampled.shape
            == b_staleness_sampled.shape
            == (self.cfg.plr_buf_size,)
        )
        self._buffer = self.set_buffer_data(
            self._buffer,
            b_x0_sample,
            b_reset_ids,
            b_score_weights_sampled,
            b_staleness_sampled,
            size,
        )

        self.buf_reset_id = b_reset_ids_orig
        self.buf_size = size

    @ft.partial(myjit, static_argnums=0)
    def set_buffer_data(
        self,
        buffer: PLRBuf,
        b_x0_leaf: TreeLeaves,
        b_reset_ids: jnp.ndarray,
        b_score_weights_sampled: jnp.ndarray,
        b_staleness_sampled: jnp.ndarray,
        size: int,
    ):
        b_x0 = jax.vmap(self.task.leaf_to_state)(b_x0_leaf)
        b_data = StateResetId(b_x0, b_reset_ids)
        return buffer.set_data(
            b_data, b_score_weights_sampled, b_staleness_sampled, size
        )
