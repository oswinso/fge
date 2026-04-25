import functools as ft
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from attrs import define
from flax import struct
from og.dyn_types import Obs
from og.grad_utils import compute_norm
from og.jax_types import BFloat
from og.jax_utils import jax_vmap
from og.networks.network_utils import get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.train_state import TrainState

from fge.core.bits.collector import RolloutOutput
from fge.core.common.ppo_nets import MLP
from fge.core.utils.jax_util import myjit


@define
class RNDCfg:
    lr: float
    hids: Sequence[int] = (256, 256, 256)


class RND(struct.PyTreeNode):
    """RND (Random Network Distillation)"""

    Cfg = RNDCfg

    pred: TrainState[BFloat]
    tgt: TrainState[BFloat]
    cfg: RNDCfg = struct.field(pytree_node=False)

    @classmethod
    def create(cls, key: PRNGKey, obs: Obs, cfg: RNDCfg):
        key_pred, key_tgt = jr.split(key)

        pred_def = MLP(
            hid_sizes=cfg.hids, act=get_act_from_str("tanh"), act_final=False
        )
        tx = get_default_tx(cfg.lr)

        pred = TrainState.create_from_def(key_pred, pred_def, (obs,), tx=tx)

        tgt = TrainState.create_from_def(key_tgt, pred_def, (obs,), tx=tx).strip()

        return RND(pred, tgt, cfg)

    @ft.partial(myjit, donate_argnums=(0,))
    def update(self, b_obs: Obs) -> tuple["RND", dict]:
        def get_loss(params):
            b_feat_pred = self.pred.apply_with(b_obs, params=params)
            assert b_feat_pred.shape[0] == b

            b_feat_tgt = self.tgt.apply(b_obs)
            assert b_feat_tgt.shape[0] == b

            # MSE loss.
            loss = jnp.mean(jnp.square(b_feat_pred - b_feat_tgt))
            info = {}
            return loss, info

        b = len(b_obs)
        grads_rnd, rnd_info = jax.grad(get_loss, has_aux=True)(self.pred.params)
        rnd_info["Grad/RND"] = compute_norm(grads_rnd)
        pred = self.pred.apply_gradients(grads=grads_rnd)
        return self.replace(pred=pred), rnd_info

    def get_bonus(self, obs: Obs):
        """Compute the bonus as norm squared."""
        feat = self.pred.apply(obs)
        feat_tgt = self.tgt.apply(obs)

        term1 = jnp.sum(jnp.square(feat - feat_tgt))
        return term1

    @jax.jit
    def get_bonus_rollout(self, b_rollout: RolloutOutput):
        bT_obs_next = b_rollout.T_obs_nxt
        bT_bonus = jax_vmap(self.get_bonus, rep=2)(bT_obs_next)

        # Zero bonus if terminal or truncated.
        bT_isreset = b_rollout.T_term | b_rollout.T_trunc
        bT_bonus = jnp.where(bT_isreset, 0.0, bT_bonus)

        # Normalize to [0, 1].
        bonus_max, bonus_min = jnp.max(bT_bonus), jnp.min(bT_bonus)
        bonus_range = bonus_max - bonus_min
        bT_bonus = (bT_bonus - bonus_min) / bonus_range

        return bT_bonus
