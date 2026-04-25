import functools as ft
from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from attrs import define
from flax import struct
from jax import lax
from og.dyn_types import BObs, Obs
from og.jax_types import BBool, FloatScalar
from og.networks.network_utils import get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.train_state import TrainState
from og.tree_utils import tree_split_dims

from fge.core.common.merge_cnn import MergeCNN
from fge.core.common.networks import encoder_modules
from fge.core.common.ppo_nets import MLP, ScalarValueNet
from fge.core.common.update_fn import update_bce, update_mse
from fge.core.utils.jax_util import get_leading_dim_fast


@define
class CIClassifierCfg:
    lr: float = 1e-3
    hids: Sequence[int] = (256, 256)

    epochs: int = 1

    smooth: float | None = 0.05
    """If not None, apply label smoothing."""

    n_sample: int = 1024
    """How many samples to sample from the buffer for training each iteration."""

    n_batches: int = 4
    """How many minibatches to split the data into for training."""


class CIClassifier(struct.PyTreeNode):
    Cfg = CIClassifierCfg

    key: PRNGKey
    logits: TrainState[FloatScalar]

    cfg: CIClassifierCfg = struct.field(pytree_node=False)
    name: str = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKey, obs: Obs, cfg: CIClassifierCfg, name: str):
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jr.key_data(key)
        key0, key = jr.split(key)

        is_obs_array = isinstance(obs, (jnp.ndarray, np.ndarray))

        # -----------------------
        if is_obs_array:
            base_cls = ft.partial(MLP, hid_sizes=cfg.hids, act=get_act_from_str("tanh"))
        else:
            # Image-based observations!
            from fge.core.envs.kinetix import lander

            assert isinstance(obs, lander.LanderObs)

            encoder_module = encoder_modules["impala_small"]

            base_cls = ft.partial(MergeCNN, encoder=encoder_module(), hid_sizes=cfg.hids)

        logits_def = ScalarValueNet(base_cls, lims=None)

        tx = get_default_tx(cfg.lr)
        logits = TrainState.create_from_def(key0, logits_def, (obs,), tx=tx)

        return CIClassifier(key0, logits, cfg, name)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_obs: BObs, b_inci: BBool):
        def updates_body(alg_: CIClassifier, b_batch: tuple[BObs, BBool]):
            b_obs_, b_inci_ = b_batch
            alg_, info = alg_.step(b_obs_, b_inci_)
            return alg_, info

        dset_size = get_leading_dim_fast(b_obs)
        assert b_inci.shape == (dset_size,)

        n_batches = self.cfg.n_batches
        batch_size = dset_size // n_batches

        b_dset = (b_obs, b_inci)

        new_self = self
        key_shuffle = jr.fold_in(self.key, self.logits.step)

        for ii in range(self.cfg.epochs):
            # 1: Shuffle and reshape
            key_shuffle_ = jr.fold_in(key_shuffle, ii)
            rand_idxs = jr.permutation(key_shuffle_, jnp.arange(dset_size))
            b_dset = jtu.tree_map(lambda x: x[rand_idxs], b_dset)
            mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))
            new_self, info = lax.scan(updates_body, new_self, mb_dset, length=n_batches)
            # Take the mean.
            info = jtu.tree_map(jnp.mean, info)

        return new_self, info

    def get_probs(self, obs: Obs) -> FloatScalar:
        logits = self.logits.apply(obs)
        probs = jnn.sigmoid(logits)
        return probs

    def step(self, b_obs: BObs, b_inci: BBool):
        logits, info = update_bce(self.logits, b_obs, b_inci, smooth=self.cfg.smooth, name=self.name)
        return self.replace(logits=logits), info
