from typing import Sequence, Type

import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import initializers
from og.dyn_types import HFloat, Obs
from og.iter_utils import signal_last_enumerate
from og.jax_types import AnyFloat
from og.networks.network_utils import ActFn, HidSizes, default_nn_init, scaled_init
from og.tfp import tfb, tfd


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(
        self,
        distribution: tfd.Distribution,
        threshold: float = 0.999,
        validate_args: bool = False,
    ):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )
        self._threshold = threshold
        self.inverse_threshold = self.bijector.inverse(threshold)

        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = np.log(1.0 - threshold)

        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    def _mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class MLP(nn.Module):
    hid_sizes: HidSizes
    act: ActFn = nn.relu
    act_final: bool = True
    scale_final: float | None = None

    @nn.compact
    def __call__(self, x: AnyFloat, apply_dropout: bool = False) -> AnyFloat:
        nn_init = default_nn_init
        for is_last_layer, ii, hid_size in signal_last_enumerate(self.hid_sizes):
            kernel_init, bias_init = nn_init(), initializers.zeros_init()
            if is_last_layer:
                if self.scale_final is not None:
                    kernel_init = scaled_init(kernel_init, self.scale_final)

            x = nn.Dense(hid_size, kernel_init=kernel_init, bias_init=bias_init)(x)

            no_activation = is_last_layer and not self.act_final
            if not no_activation:
                x = self.act(x)
        return x


class SoftmaxDiscrete(nn.Module):
    base_cls: Type[nn.Module]
    n_out: int
    scale_final: float = 1e-2

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        logits = nn.Dense(
            self.n_out,
            kernel_init=scaled_init(default_nn_init(), self.scale_final),
            name="OutLogit",
        )(x)
        return tfd.Categorical(logits=logits)


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class TanhNormal(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5
    scale_final: float = 1.0

    @property
    def std_dev_init_inv(self):
        # Inverse of logsumexp.
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        means = nn.Dense(
            self.nu, kernel_init=scaled_init(default_nn_init(), self.scale_final), name="OutputDenseMean"
        )(x)
        stds_trans = nn.Dense(
            self.nu, kernel_init=default_nn_init(), name="OutputDenseLogStd"
        )(x)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min
        dist = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(
            TanhTransformedDistribution(dist), reinterpreted_batch_ndims=1
        )

    @property
    def nu(self):
        return self._nu


class ValueNet(nn.Module):
    base_cls: Type[nn.Module]
    n_out: int
    lims: tuple[float, float] | None = None

    @nn.compact
    def __call__(self, obs: Obs) -> HFloat:
        x = self.base_cls()(obs)
        x = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)

        if self.lims is not None:
            # Clamp the output to the specified lims using sigmoid.
            lo, hi = self.lims
            # [0, 1]
            x = nnx.sigmoid(x)
            # [lo, hi]
            x = lo + (hi - lo) * x

        return x


class ScalarValueNet(nn.Module):
    base_cls: Type[nn.Module]
    lims: tuple[float, float] | None = None

    @nn.compact
    def __call__(self, obs: Obs) -> HFloat:
        x = self.base_cls()(obs)
        x = nn.Dense(1, kernel_init=default_nn_init())(x)
        x = x.squeeze(-1)

        if self.lims is not None:
            # Clamp the output to the specified lims using sigmoid.
            lo, hi = self.lims
            # [0, 1]
            x = nnx.sigmoid(x)
            # [lo, hi]
            x = lo + (hi - lo) * x

        return x


class ScalarNet(nn.Module):
    base_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: Obs) -> HFloat:
        x = self.base_cls()(obs)
        x = nn.Dense(1, kernel_init=default_nn_init())(x)
        return x.squeeze(-1)
