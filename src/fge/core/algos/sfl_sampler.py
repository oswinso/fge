from typing import Self

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from attrs import define
from flax import struct
from og.jax_types import BBool
from og.rng import PRNGKey
from og.tree_utils import make_batch_pytree, tree_where_dim0

from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import StateResetId
from fge.core.envs.jax_task import JaxTask


@define
class SFLSamplerCfg:
    update_buf_every: int = 50  # Update period T. How often to call "collect_learnable_levels".
    p_sample_base: float = 0.5  # 1 - rho, probability of sampling from the base distribution.
    init_sample_size: int = 5_000  # N. How many levels to sample before rolling out and getting top K
    sfl_buf_size: int = 1_000  # The K in Top K levels in B ranked by learnability
    T_coef: int = 5  # How many times to multiply with rollout_eval_T to get a non-0/1 success rate
    """How large the buffer (on GPU) should be."""


class SFLBuf(struct.PyTreeNode):
    b_data: StateResetId
    size: int

    n_sampled_buf: int
    """Keep track of how many items were sampled, so we can increment the staleness of everything not in the buffer."""

    p_sample_base: float

    id_provider: ResetIDProvider

    capacity: int = struct.field(pytree_node=False)
    task: JaxTask = struct.field(pytree_node=False)
    cfg: SFLSamplerCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            capacity: int,
            item: StateResetId,
            task: JaxTask,
            cfg: SFLSamplerCfg,
            id_provider: ResetIDProvider,
    ) -> Self:
        b_data = make_batch_pytree(item, capacity, fill_value=0, whichnp=jnp)

        return cls(
            b_data=b_data,
            size=0,
            p_sample_base=cfg.p_sample_base,
            n_sampled_buf=0,
            id_provider=id_provider,
            capacity=capacity,
            task=task,
            cfg=cfg,
        )

    def sample(self, key: PRNGKey, num: int, b_valid: BBool | None = None):
        key_bernoulli, key_sample_base, key_buffer, key_sfl = jr.split(key, 4)

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

        b_weights = jnp.ones(self.capacity, dtype=jnp.float32)

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

        b_statetup: StateResetId = tree_where_dim0(
            b_sample_base, b_statetup_base, b_statetup_buf
        )

        n_sampled_buf = (b_sample_buf & b_valid).sum()
        n_sampled_buf_total = self.n_sampled_buf + n_sampled_buf

        self_new = self.replace(
            n_sampled_buf=n_sampled_buf_total,
            id_provider=id_provider,
        )
        return self_new, b_statetup

    def set_data(
            self, b_data: StateResetId, size: int
    ):
        assert not isinstance(b_data.state, tuple)
        return self.replace(
            b_data=b_data,
            size=size,
            n_sampled_buf=0,
        )


class SFLSampler:
    """A sampler that uses Prioritized Level Replay (SFL)"""

    Cfg = SFLSamplerCfg

    def __init__(
            self,
            task: JaxTask,
            cfg: SFLSamplerCfg,
            reset_id_provider: ResetIDProvider,
            seed: int = 1234567,
    ):
        self.task = task
        self.cfg = cfg

        # n_init = 1 << 16
        n_init = self.cfg.sfl_buf_size

        state = task.reset(jr.PRNGKey(0))
        statetup = StateResetId(state, np.array(-1, dtype=np.int32))
        buffer = SFLBuf.create(
            self.cfg.sfl_buf_size, statetup, task, cfg, reset_id_provider
        )
        gpu_device = jax.devices("gpu")[0]

        self._buffer: SFLBuf = jax.device_put(buffer, gpu_device)

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
    def buffer(self) -> SFLBuf:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: SFLBuf):
        self._buffer = buffer
