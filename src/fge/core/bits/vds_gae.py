import functools as ft
from typing import NamedTuple, Self, Sequence

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from attrs import define
from flax import struct
from jax import lax
from loguru import logger
from og.dyn_types import BObs, Obs
from og.grad_utils import compute_norm
from og.jax_types import BFloat, FloatDict
from og.jax_utils import jax_vmap, merge01
from og.networks.ensemble import Ensemble
from og.networks.network_utils import get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.train_state import TrainState
from og.tree_utils import tree_split_dims

from fge.core.bits.collector import Collector, RolloutOutput
from fge.core.bits.gae import compute_gae
from fge.core.common.merge_cnn import MergeCNN, MergeCNN2
from fge.core.common.networks import encoder_modules
from fge.core.common.ppo_nets import MLP, ScalarNet, ScalarValueNet
from fge.core.common.update_fn import update_mse
from fge.core.envs.jax_task import JaxTask, TimedState


@define
class VDSGAECfg:
    lr: float
    gae_lambda: float = 0.95
    disc_gamma: float = 0.99
    hids: Sequence[int] = (256, 256)
    n_ensemble: int = 3

    value_lims: tuple[float, float] | None = (-1e-3, 1.0)

    n_batches: int = 1
    n_update_epochs: int = 1
    n_sample_reset: int = 1024


class VDSGAE(struct.PyTreeNode):
    Cfg = VDSGAECfg

    key: PRNGKey
    e_V: TrainState[BFloat]
    # V1: TrainState[BFloat]
    # V2: TrainState[BFloat]
    # V3: TrainState[BFloat]

    task: JaxTask = struct.field(pytree_node=False)
    cfg: VDSGAECfg = struct.field(pytree_node=False)

    class Batch(NamedTuple):
        b_obs: BObs
        be_Ql: BFloat

        @property
        def batch_size(self) -> int:
            return len(self.be_Ql)

    @classmethod
    def create(cls, key: PRNGKey, obs: Obs, task: JaxTask, cfg: VDSGAECfg):
        is_obs_array = isinstance(obs, (jnp.ndarray, np.ndarray))

        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jr.key_data(key)
        key0, key_tgt = jr.split(key)

        if is_obs_array:
            base_cls = ft.partial(MLP, hid_sizes=cfg.hids, act=get_act_from_str("tanh"))
            V_def = ft.partial(ScalarValueNet, base_cls, lims=cfg.value_lims)
            e_V_def = Ensemble(V_def, num=cfg.n_ensemble)
        else:
            # Image-based observations!
            from fge.core.envs.kinetix import lander

            assert isinstance(obs, lander.LanderObs)

            encoder_module = encoder_modules["impala_small"]

            base_cls = ft.partial(MLP, hid_sizes=cfg.hids)
            V_def = ft.partial(ScalarValueNet, base_cls, lims=cfg.value_lims)
            head = Ensemble(V_def, num=cfg.n_ensemble)
            e_V_def = MergeCNN2(encoder=encoder_module(), head=head)

        tx = get_default_tx(cfg.lr)
        e_V = TrainState.create_from_def(key0, e_V_def, (obs,), tx=tx)

        # key0_1, key0_2, key0_3 = jr.split(key0, 3)
        # V1 = TrainState.create_from_def(key0_1, V_def, (obs,), tx=tx)
        # V2 = TrainState.create_from_def(key0_2, V_def, (obs,), tx=tx)
        # V3 = TrainState.create_from_def(key0_3, V_def, (obs,), tx=tx)

        return VDSGAE(key0, e_V, task, cfg)
        # return VDSGAE(key0, V1, V2, V3, cfg)

    def get_e_V(self, obs: Obs):
        # V1 = self.V1.apply(obs)
        # V2 = self.V2.apply(obs)
        # V3 = self.V3.apply(obs)
        # e_V = jnp.array([V1, V2, V3])
        e_V = self.e_V.apply(obs)
        return e_V

    def get_std(self, obs: Obs):
        """Compute the std deviation of the ensemble for the given observation."""
        # e_V = self.e_V.apply(obs)
        e_V = self.get_e_V(obs)
        return jnp.std(e_V)

    def make_dset(self, data: Collector.Rollout):
        # Package the processed data into a PPOBatch object.
        bT_obs_nxt = self.task.leaf_to_obs(data.T_obs_nxt)
        bT_obs_now = self.task.leaf_to_obs(data.T_obs_now)

        # 1: Compute e_Vl from data.
        bTe_Vl_nxt = jax_vmap(self.get_e_V, rep=2)(bT_obs_nxt)
        bTe_Vl_now = jax_vmap(self.get_e_V, rep=2)(bT_obs_now)

        b, T = data.T_rew.shape
        assert bTe_Vl_nxt.shape == bTe_Vl_now.shape == (b, T, self.cfg.n_ensemble)

        ebT_Vl_nxt = ei.rearrange(bTe_Vl_nxt, "b T e -> e b T")
        ebT_Vl_now = ei.rearrange(bTe_Vl_now, "b T e -> e b T")

        bT_rew = data.T_rew
        bT_cost = -bT_rew
        bT_isterm = data.T_term
        bT_nextvalid = ~bT_isterm

        # 2: Compute Ql using GAE.
        gae_fn = ft.partial(compute_gae, self.cfg.disc_gamma, self.cfg.gae_lambda)
        vmap_gae_fn = jax.vmap(jax.vmap(gae_fn), in_axes=(None, 0, 0, None))

        ebT_Ql, _ = vmap_gae_fn(bT_cost, ebT_Vl_now, ebT_Vl_nxt, bT_nextvalid)

        assert ebT_Ql.shape == ebT_Vl_now.shape == (self.cfg.n_ensemble, b, T)

        bT_obs_now = bT_obs_now

        # Merge the b and T
        b_obs_now = jtu.tree_map(merge01, bT_obs_now)
        be_Ql = ei.rearrange(ebT_Ql, "e b T -> (b T) e")

        be_batch = VDSGAE.Batch(b_obs_now, be_Ql)
        return be_batch

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, data: Collector.Rollout):
        def updates_body(alg_: VDSGAE, b_batch: VDSGAE.Batch):
            alg_, val_info = alg_.update_value(b_batch)
            return alg_, val_info

        new_self = self
        key_shuffle = jr.fold_in(self.key, self.e_V.step)

        for ii in range(self.cfg.n_update_epochs):
            # Compute GAE values.
            b_dset = new_self.make_dset(data)

            n_batches = self.cfg.n_batches
            assert b_dset.batch_size % n_batches == 0
            batch_size = b_dset.batch_size // self.cfg.n_batches
            logger.info(f"[VDS] Using {n_batches} minibatches each epoch!")

            # 2: Shuffle and reshape
            key_shuffle_ = jr.fold_in(key_shuffle, ii)
            rand_idxs = jr.permutation(key_shuffle_, jnp.arange(b_dset.batch_size))
            b_dset = jtu.tree_map(lambda x: x[rand_idxs], b_dset)
            mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))

            new_self, info = lax.scan(updates_body, new_self, mb_dset, length=n_batches)
            # Take the mean.
            info = jtu.tree_map(jnp.mean, info)

        return new_self, info

    def update_value(self, batch: Batch) -> tuple[Self, FloatDict]:
        e_V, info = update_mse(self.e_V, batch.be_Ql, batch.b_obs, "e_V", sum_dim=True)
        return self.replace(e_V=e_V), info
        # V1, info = update_mse(self.V1, batch.be_Ql[:, 0], batch.b_obs, "e_V1", sum_dim=True)
        # V2, _ = update_mse(self.V2, batch.be_Ql[:, 1], batch.b_obs, "e_V2", sum_dim=True)
        # V3, _ = update_mse(self.V3, batch.be_Ql[:, 2], batch.b_obs, "e_V3", sum_dim=True)
        # return self.replace(V1=V1, V2=V2, V3=V3), info

    def get_reset_fn(self, task):
        return VDSResetFn.create(self, task, n_sample=self.cfg.n_sample_reset)


class VDSResetFn(struct.PyTreeNode):
    vds: VDSGAE
    task: JaxTask = struct.field(pytree_node=False)
    n_sample: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, vds: VDSGAE, task: JaxTask, n_sample: int = 1024) -> Self:
        return VDSResetFn(vds, task, n_sample)

    def __call__(self, b_key: PRNGKey, num: int) -> TimedState:
        key0 = b_key[0]
        key_sample0, key_choice = jr.split(key0)
        b_key_sample0 = jr.split(key_sample0, self.n_sample)

        # 1: Sample a bunch of random states.
        b_state = jax.vmap(self.task.reset)(b_key_sample0)
        b_obs = jax.vmap(self.task.get_obs)(b_state)

        # 2: Compute the std dev.
        b_std = jax.vmap(self.vds.get_std)(b_obs)
        #    Make sure b_std is not all zero.
        b_std = jnp.maximum(b_std, 1e-9)
        b_probs = b_std / jnp.sum(b_std)

        # 3: Sample proportional to the std deviation.
        b_idxs = jr.choice(key_choice, self.n_sample, p=b_probs, shape=(num,))

        # 4: Sample the states.
        b_state = jtu.tree_map(lambda x: x[b_idxs], b_state)

        return b_state
