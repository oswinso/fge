import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from cyclopts import Parameter
from flax import struct
from loguru import logger
from og.jax_types import BBool, IntScalar
from og.rng import PRNGKey
from og.tree_utils import tree_where_dim0

from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import StateResetId
from fge.core.common.circ_buf_jax import CircOnlyInJax
from fge.core.envs.jax_task import JaxTask
from fge.core.utils.jax_util import myjit


@Parameter(name="*", group="resetbuf")
@define
class ResetBufCfg:
    capacity_ci: int = 100_000
    capacity_explore: int = 512
    capacity_predci: int = 512

    p_explore: float = 0.4
    """Fraction of samples to use for exploration outside CI."""

    p_base: float = 0.1
    """Fraction of samples that are sampled from the base distribution."""

    p_predci: float = 0.0
    """Fraction of samples that are sampled from the predicted CI distribution."""

    # Fraction of samples when sampling from CI to dedicate to the most recent sample.
    frac_latest_ci: float = 0.2


type ResetBufJax = CircOnlyInJax[StateResetId]


class ResetBuf(struct.PyTreeNode):
    Cfg = ResetBufCfg

    buf_ci: ResetBufJax
    """Store the previous argmax of worst case in CI. TODO: Reservoir buffer."""

    buf_explore: ResetBufJax
    """Store a buffer of the current "explore distribution"."""

    buf_predci: ResetBufJax
    """Store a buffer of p_base(x | pred in ci) sampled using rejection sampling."""

    id_provider: ResetIDProvider

    n_sample_base: IntScalar
    n_sample_ci: IntScalar
    n_sample_explore: IntScalar
    n_sample_predci: IntScalar

    p_explore: float
    p_base: float
    p_predci: float

    task: JaxTask = struct.field(pytree_node=False)
    cfg: ResetBufCfg = struct.field(pytree_node=False)

    @staticmethod
    def create(statetup, task: JaxTask, cfg: ResetBufCfg, id_provider: ResetIDProvider):
        buf_ci = CircOnlyInJax.create(cfg.capacity_ci, statetup)
        buf_explore = CircOnlyInJax.create(cfg.capacity_explore, statetup)
        buf_predci = CircOnlyInJax.create(cfg.capacity_predci, statetup)
        p_explore = cfg.p_explore
        p_base = cfg.p_base
        p_predci = cfg.p_predci
        logger.critical(
            "p_explore: {:.3f} | p_base: {:.3f} | p_predci: {:.3f} | p_ci: {:.3f}".format(
                p_explore, p_base, p_predci, 1 - (p_explore + p_base + p_predci)
            )
        )
        return ResetBuf(
            buf_ci,
            buf_explore,
            buf_predci,
            id_provider,
            0,
            0,
            0,
            0,
            p_explore=p_explore,
            p_base=p_base,
            p_predci=p_predci,
            task=task,
            cfg=cfg,
        )

    @ft.partial(myjit, donate_argnums=0)
    def add_to_ci(self, statetup: StateResetId):
        """Add a state to the rehearsal buffer."""
        buf_ci = self.buf_ci.add(statetup)
        return self.replace(buf_ci=buf_ci)

    def add_batch_ci(self, b_item: StateResetId, num: int):
        buf_ci = self.buf_ci.add_batch(b_item, num)
        assert np.all(buf_ci.data.reset_id != -1)
        return self.replace(buf_ci=buf_ci)

    @ft.partial(myjit, donate_argnums=0)
    def add_to_explore(self, statetup: StateResetId):
        """Add a state to the rehearsal buffer."""
        buf_explore = self.buf_explore.add(statetup)
        return self.replace(buf_explore=buf_explore)

    def add_batch_explore(self, b_item: StateResetId, num: int):
        buf_explore = self.buf_explore.add_batch(b_item, num)
        assert np.all(buf_explore.data.reset_id != -1)
        return self.replace(buf=buf_explore)

    def set_predci(self, b_item: StateResetId, num: int):
        buf_predci = self.buf_predci.add_all(b_item, num)
        return self.replace(buf_predci=buf_predci)

    def get_prob_sample(self):
        p_base = self.p_base

        predci_empty = self.buf_predci.size == 0
        p_predci = jnp.where(predci_empty, 0.0, self.p_predci)

        p_ci = 1.0 - (self.p_explore + p_base + p_predci)

        # If buf_ci is empty (at the start), then p_ci is 0, and redistribute the probability to explore.
        ci_empty = self.buf_ci.size == 0
        p_ci = jnp.where(ci_empty, 0.0, p_ci)
        p_explore = 1.0 - (p_base + p_ci + p_predci)

        # If buf_explore is empty (at the start), then p_explore is 0.
        explore_empty = self.buf_explore.size == 0
        p_explore = jnp.where(explore_empty, 0.0, p_explore)

        # Set p_base so that it is the remaining probability.
        p_base = 1.0 - (p_ci + p_explore + p_predci)

        return p_base, p_ci, p_explore, p_predci

    def sample(self, key: PRNGKey, num: int, b_valid: BBool | None = None):
        (
            key_bernoulli,
            key_sample_ci,
            key_sample_explore,
            key_sample_base,
            key_sample_predci,
        ) = jr.split(key, 5)

        p_base, p_ci, p_explore, p_predci = self.get_prob_sample()
        p = jnp.array([p_base, p_ci, p_explore, p_predci])

        b_sample_type = jr.choice(
            key_bernoulli, len(p), shape=(num,), p=p, replace=True
        )

        b_base = b_sample_type == 0
        b_ci = b_sample_type == 1
        b_explore = b_sample_type == 2
        b_predci = b_sample_type == 3

        # Sample from CI buffer.
        b_statetup_ci = self.buf_ci.sample(
            key_sample_ci, num, replace=True, frac_latest=self.cfg.frac_latest_ci
        )

        # Sample from explore buffer.
        b_statetup_explore = self.buf_explore.sample(
            key_sample_explore, num, replace=True
        )

        # Sample from predci buffer.
        b_statetup_predci = self.buf_predci.sample(key_sample_predci, num, replace=True)

        b_base_valid = b_base
        if b_valid is not None:
            b_base_valid = b_base & b_valid
        b_key_sample = jr.split(key_sample_base, num)
        b_state_ = jax.vmap(self.task.reset)(b_key_sample)
        id_provider, b_reset_id_ = self.id_provider.get_masked_ids(b_base_valid)
        b_statetup_base = StateResetId(b_state_, b_reset_id_)
        b_statetup = tree_where_dim0(b_base, b_statetup_base, b_statetup_ci)
        b_statetup = tree_where_dim0(b_explore, b_statetup_explore, b_statetup)
        b_statetup = tree_where_dim0(b_predci, b_statetup_predci, b_statetup)

        # Count how many times we sampled argmax.
        if b_valid is None:
            # Sum b_useargmax.
            n_sample_base = jnp.sum(b_base)
            n_sample_ci = jnp.sum(b_ci)
            n_sample_explore = jnp.sum(b_explore)
            n_sample_predci = jnp.sum(b_predci)
        else:
            # We only use argmax if b_useargmax AND b_valid.
            n_sample_base = jnp.sum(b_base & b_valid)
            n_sample_ci = jnp.sum(b_ci & b_valid)
            n_sample_explore = jnp.sum(b_explore & b_valid)
            n_sample_predci = jnp.sum(b_predci & b_valid)

        n_sample_base_new = self.n_sample_base + n_sample_base
        n_sample_ci_new = self.n_sample_ci + n_sample_ci
        n_sample_explore_new = self.n_sample_explore + n_sample_explore
        n_sample_predci_new = self.n_sample_predci + n_sample_predci
        self_new = self.replace(
            id_provider=id_provider,
            n_sample_base=n_sample_base_new,
            n_sample_ci=n_sample_ci_new,
            n_sample_explore=n_sample_explore_new,
            n_sample_predci=n_sample_predci_new,
        )
        return self_new, b_statetup

    def set_probs_inplace(
            self, p_explore: float, p_base: float, p_predci: float = None
    ):
        """Modify the exploration probabilities in place."""

        # Do this so that the type is the same.
        p_explore = 0 * self.p_explore + p_explore
        p_base = 0 * self.p_base + p_base
        p_predci = 0 * self.p_predci + p_predci

        logger.info(
            "Setting p_explore: {}, p_base: {}, p_predci: {}",
            p_explore,
            p_base,
            p_predci,
        )

        object.__setattr__(self, "p_explore", p_explore)
        object.__setattr__(self, "p_base", p_base)
        object.__setattr__(self, "p_predci", p_predci)
