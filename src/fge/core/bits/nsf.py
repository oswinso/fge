import functools as ft
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import paramax
from attrs import define
from et.decorators.timeit import timeit
from flax import struct
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import masked_autoregressive_flow
from jax import lax
from jaxtyping import ArrayLike
from og.dyn_types import Obs
from og.grad_utils import compute_norm
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.train_state import EqTrainState

from fge.core.bits.runmeanstd import RunningMeanStd
from fge.core.utils.jax_util import myjit


@define
class NSFCfg:
    lr: float = 1e-3
    wd: float = 1e-4
    layers: int = 8
    width: int = 50

    epochs: int = 1

    handle_std0: bool = False


class NSF(struct.PyTreeNode):
    Cfg = NSFCfg

    rms: RunningMeanStd

    model: EqTrainState[AbstractDistribution]
    static: Any = struct.field(pytree_node=False)
    cfg: NSFCfg = struct.field(pytree_node=False)

    @staticmethod
    def create(key: PRNGKey, obs: Obs, cfg: NSFCfg):
        mean = jnp.zeros_like(obs)
        rms = RunningMeanStd.create(mean, handle_std0=cfg.handle_std0)
        flow = masked_autoregressive_flow(
            key,
            base_dist=Normal(mean),
            transformer=RationalQuadraticSpline(knots=8, interval=4),
            flow_layers=cfg.layers,
            nn_width=cfg.width,
        )
        params, static = eqx.partition(
            flow,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        tx = get_default_tx(cfg.lr, wd=cfg.wd)
        train_state = EqTrainState.create(params, tx)
        return NSF(rms, train_state, static, cfg)

    @property
    def params(self):
        return self.model.model

    def dist(self, params=None) -> AbstractDistribution:
        if params is None:
            params = self.params
        return paramax.unwrap(eqx.combine(params, self.static))

    @jax.jit
    def log_prob(
        self, obs: ArrayLike, condition: ArrayLike | None = None
    ) -> jnp.ndarray:
        obs_norm = (obs - self.rms.mean) / (self.rms.std + 1e-6)
        return self.dist().log_prob(obs_norm, condition=condition)

    @ft.partial(myjit, donate_argnums=(0,))
    def update(self, b_obs: Obs):
        # Update the running mean and std
        obs_mean, obs_var = jnp.mean(b_obs, axis=0), jnp.var(b_obs, axis=0)
        n_batch = len(b_obs)
        rms_new = self.rms.update_from_moments(obs_mean, obs_var, n_batch)
        self_new = self.replace(rms=rms_new)

        b_obs_norm = (b_obs - self_new.rms.mean) / (self_new.rms.std + 1e-6)

        return self_new._step(b_obs_norm)

    def _step(self, b_obs: Obs):
        def body(carry, inp):
            self_, _ = carry
            self_new, info = self_._single_step(b_obs)
            return (self_new, info), None

        self_new, info = self._single_step(b_obs)

        if self.cfg.epochs > 1:
            (self_new, info) = lax.scan(
                body,
                (self_new, info),
                length=self.cfg.epochs - 1,
                unroll=self.cfg.epochs - 1,
            )
        return self_new, info

    def _single_step(self, b_obs: Obs):
        def loss_fn(params_flat_):
            params = self.model.model_with(params_flat_)
            dist = self.dist(params=params)
            b_neg_logp = -dist.log_prob(b_obs)
            loss = b_neg_logp.mean()
            info = {"NSF/NLL": loss}
            return loss, info

        grad, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        info["NSF/Grad"] = compute_norm(grad)
        model_new = self.model.apply_gradients(grad)
        return self.replace(model=model_new), info
