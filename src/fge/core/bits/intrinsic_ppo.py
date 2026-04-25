import functools as ft
from typing import NamedTuple, Self

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from attrs import define
from flax import struct
from loguru import logger
from og.dyn_types import BControl, BObs, BTFloat
from og.jax_types import BFloat, FloatDict, FloatScalar
from og.jax_utils import jax_vmap, merge01
from og.networks.network_utils import get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.train_state import TrainState
from og.tree_utils import tree_split_dims

from fge.core.bits.collector import Collector
from fge.core.bits.gae import compute_gae
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg, SumPPOTrainCfg, SumPPOUpdateData
from fge.core.common.ppo_nets import MLP, ValueNet
from fge.core.common.reset_buf import ResetBuf
from fge.core.common.update_fn import update_mse, update_policy_ppo
from fge.core.envs.jax_task import JaxTask
from fge.core.utils.jax_util import myjit


@define
class IntrinsicSumPPOTrainCfg(SumPPOTrainCfg):

    intrinsic_coef: float = 1.0
    """Coefficient for the intrinsic"""

    enable_intrinsic: bool = False


@define
class IntrinsicSumPPOCfg(SumPPOCfg):
    train_cfg: IntrinsicSumPPOTrainCfg


class IntrinsicSumPPO(struct.PyTreeNode):
    """Add an additional Vl for intrinsic reward. Separate the two value functions."""

    Cfg = IntrinsicSumPPOCfg
    TrainCfg = IntrinsicSumPPOTrainCfg

    ppo: SumPPO
    Vl_int: TrainState[FloatScalar]

    cfg: IntrinsicSumPPOCfg = struct.field(pytree_node=False)

    class IntPPOBatch(NamedTuple):
        b_obs: BObs
        b_control: BControl
        b_logprob: BFloat

        b_Al_tot: BFloat
        """For policy"""

        b_Ql_ext: BFloat
        """For ext_value"""

        b_Ql_int: BFloat
        """For int_value"""

        @property
        def batch_size(self) -> int:
            assert self.b_logprob.ndim == 1
            return len(self.b_logprob)

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def ent_cf(self):
        return self.ppo.ent_cf

    @property
    def pol_lr(self):
        return self.ppo.pol_lr

    @property
    def policy(self):
        return self.ppo.policy

    @property
    def task(self):
        return self.ppo.task

    @property
    def key_shuffle(self):
        return self.ppo.key_shuffle

    @property
    def Vl_ext(self):
        return self.ppo.Vl

    @property
    def disc_gamma(self):
        return self.ppo.disc_gamma

    @property
    def update_idx(self):
        return self.ppo.update_idx

    @classmethod
    def create(cls, key: PRNGKey, task: JaxTask, cfg: IntrinsicSumPPOCfg) -> Self:
        key_ppo, key_Vl_int = jr.split(key, 2)
        ppo = SumPPO.create(key_ppo, task, cfg)
        obs = task.get_dummy_obs()

        if cfg.train_cfg.enable_intrinsic:
            V_base_cls = ft.partial(MLP, act=get_act_from_str("tanh"), hid_sizes=cfg.val_hids)
            Vl_def = ValueNet(V_base_cls, 1)
            Vl_tx = get_default_tx(cfg.val_lr)
            Vl_int = TrainState.create_from_def(key_Vl_int, Vl_def, (obs,), tx=Vl_tx)
        else:
            # Copy, use separate buffer.
            Vl_int = jtu.tree_map(lambda x: jnp.array(x, copy=True) if isinstance(x, jnp.ndarray) else x, ppo.Vl)

        return IntrinsicSumPPO(ppo, Vl_int, cfg)

    def get_data_from_rollout(self, rollout: Collector.Rollout) -> Collector.Rollout:
        return self.ppo.get_data_from_rollout(rollout)

    def make_bT_dset(self, data: Collector.Rollout, bT_rew_int: BTFloat):
        # 1: Compute h_Vl from data.
        bT_Vl_ext_nxt = jax_vmap(self.Vl_ext.apply, rep=2)(data.T_obs_nxt).squeeze(-1)
        bT_Vl_ext_now = jax_vmap(self.Vl_ext.apply, rep=2)(data.T_obs_now).squeeze(-1)

        bT_Vl_int_nxt = jax_vmap(self.Vl_int.apply, rep=2)(data.T_obs_nxt).squeeze(-1)
        bT_Vl_int_now = jax_vmap(self.Vl_int.apply, rep=2)(data.T_obs_now).squeeze(-1)

        # 2: Compute Ql using GAE.
        gae_fn = ft.partial(compute_gae, self.disc_gamma, self.train_cfg.gae_lambda)

        bT_rew = data.T_rew
        bT_cost_ext = -bT_rew
        bT_cost_int = -bT_rew_int
        bT_isterm = data.T_term
        bT_nextvalid = ~bT_isterm

        bT_Ql_ext, _ = jax_vmap(gae_fn)(bT_cost_ext, bT_Vl_ext_now, bT_Vl_ext_nxt, bT_nextvalid)
        bT_Ql_int, _ = jax_vmap(gae_fn)(bT_cost_int, bT_Vl_int_now, bT_Vl_int_nxt, bT_nextvalid)
        assert bT_Ql_ext.shape == bT_Ql_int.shape == bT_cost_ext.shape

        bT_Al_ext = bT_Ql_ext - bT_Vl_ext_now
        bT_Al_int = bT_Ql_int - bT_Vl_int_now

        mean_ext, mean_int = jnp.mean(bT_Al_ext), jnp.mean(bT_Al_int)
        std_ext, std_int = jnp.std(bT_Al_ext), jnp.std(bT_Al_int)

        info = {
            "gae/min_ext": bT_Al_ext.min(),
            "gae/mean_ext": mean_ext,
            "gae/std_ext": std_ext,
            "gae/max_ext": bT_Al_ext.max(),
            # ------------------------
            "gae/min_int": bT_Al_int.min(),
            "gae/mean_int": mean_int,
            "gae/std_int": std_int,
            "gae/max_int": bT_Al_int.max(),
        }

        if self.train_cfg.normalize_adv_pre:
            bT_Al_ext = (bT_Al_ext - mean_ext) / (std_ext + 1e-5)
            bT_Al_int = (bT_Al_int - mean_int) / (std_int + 1e-5)

        bT_Al = bT_Al_ext + self.train_cfg.intrinsic_coef * bT_Al_int

        mean, std = jnp.mean(bT_Al), jnp.std(bT_Al)
        info = {
            "gae/min": bT_Al.min(),
            "gae/mean": mean,
            "gae/std": std,
            "gae/max": bT_Al.max(),
        } | info
        # ------------------------

        if self.train_cfg.normalize_adv:
            bT_Al = (bT_Al - mean) / (std + 1e-5)

        # 3: Make the dataset by flattening (b, T) -> (b * T,)
        bT_batch = IntrinsicSumPPO.IntPPOBatch(
            data.T_obs_now, data.T_control, data.T_logprob, bT_Al, bT_Ql_ext, bT_Ql_int
        )
        return bT_batch, info

    def make_dset(self, data: Collector.Rollout, bT_rew_int: BTFloat) -> tuple[IntPPOBatch, dict]:
        bT_batch, info = self.make_bT_dset(data, bT_rew_int)
        b_batch = jtu.tree_map(merge01, bT_batch)
        return b_batch, info

    def update_intrinsic_only(self, data: SumPPOUpdateData):
        ppo, info = self.ppo.update(data)
        return self.replace(ppo=ppo), info

    @ft.partial(myjit, donate_argnums=0)
    def update(self, data: Collector.Rollout, bT_rew_int: BTFloat) -> tuple[Self, FloatDict]:
        def updates_body(alg_: IntrinsicSumPPO, b_batch: IntrinsicSumPPO.IntPPOBatch):
            alg_, val_info = alg_.update_value(b_batch)
            alg_, pol_info = alg_.update_policy(b_batch)
            return alg_, val_info | pol_info

        new_self = self
        # Before we do anything, set the policy lr.
        pol_lr = self.pol_lr
        new_self.policy.set_lr(pol_lr)
        key_shuffle = jr.fold_in(self.key_shuffle, self.update_idx)

        for ii in range(self.train_cfg.n_update_epochs):
            # Compute GAE values.
            b_dset, info_gae = new_self.make_dset(data, bT_rew_int)

            n_batches = self.train_cfg.n_batches
            assert b_dset.batch_size % n_batches == 0
            batch_size = b_dset.batch_size // self.train_cfg.n_batches
            logger.info(f"Using {n_batches} minibatches each epoch!")

            # 2: Shuffle and reshape
            key_shuffle_ = jr.fold_in(key_shuffle, ii)
            rand_idxs = jr.permutation(key_shuffle_, jnp.arange(b_dset.batch_size))
            b_dset = jtu.tree_map(lambda x: x[rand_idxs], b_dset)
            mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))

            # 3: Perform value function and policy updates.
            new_self, info = lax.scan(updates_body, new_self, mb_dset, length=n_batches)
            # Take the mean, unless it has the words "max", then we take the max.
            info_new = {}
            for k, v in info.items():
                if "max" in k:
                    info_new[k] = jnp.max(v)
                else:
                    info_new[k] = jnp.mean(v)
            # info = jtu.tree_map(jnp.mean, info)
            info = info_new

        info = info | info_gae

        info["steps/policy"] = self.policy.step
        info["steps/Vl"] = self.Vl_ext.step
        info["anneal/ent_cf"] = self.ent_cf
        info["anneal/pol_lr"] = pol_lr
        ppo_new = new_self.ppo.replace(update_idx=self.ppo.update_idx + 1)
        return new_self.replace(ppo=ppo_new), info

    def update_value(self, batch: IntPPOBatch) -> tuple[Self, FloatDict]:
        Vl_ext, Vl_ext_info = update_mse(
            self.Vl_ext,
            batch.b_Ql_ext[:, None],
            batch.b_obs,
            "V",
            clip_grad=self.train_cfg.clip_grad_V,
        )
        Vl_int, Vl_int_info = update_mse(
            self.Vl_int,
            batch.b_Ql_int[:, None],
            batch.b_obs,
            "V_int",
            clip_grad=self.train_cfg.clip_grad_V,
        )
        ppo_new = self.ppo.replace(Vl=Vl_ext)

        return self.replace(ppo=ppo_new, Vl_int=Vl_int), Vl_ext_info | Vl_int_info

    def update_policy(self, batch: IntPPOBatch) -> tuple[Self, FloatDict]:
        key_entropy = jr.split(jr.fold_in(self.ppo.key_entropy, self.policy.step), batch.batch_size)
        train_cfg = self.train_cfg
        policy, pol_info = update_policy_ppo(
            key_entropy,
            self.policy,
            batch.b_obs,
            batch.b_control,
            batch.b_logprob,
            batch.b_Al_tot,
            train_cfg.clip_ratio,
            self.ent_cf,
            clip_grad=train_cfg.clip_grad_pol,
        )

        ppo_new = self.ppo.replace(policy=policy)
        return self.replace(ppo=ppo_new), pol_info

    @property
    def rollout_T(self) -> int:
        return self.ppo.rollout_T

    @ft.partial(jax.jit, donate_argnums=1)
    def collect(self, collector: Collector) -> tuple[Collector, Collector.Rollout, dict]:
        return self.ppo.collect(collector)

    @jax.jit
    def collect_eval(self, collector: Collector) -> tuple[Collector.Rollout, dict]:
        return self.ppo.collect_eval(collector)

    @jax.jit
    def collect_eval_det(self, collector: Collector) -> tuple[Collector.Rollout, dict]:
        return self.ppo.collect_eval_det(collector)

    @ft.partial(myjit, donate_argnums=(1, 2))
    def collect_with_buf(
        self, collector: Collector, reset_buf: ResetBuf
    ) -> tuple[Collector, ResetBuf, Collector.Rollout, dict]:
        return self.ppo.collect_with_buf(collector, reset_buf)
