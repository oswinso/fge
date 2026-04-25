import functools as ft
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from loguru import logger


class MLP(nn.Module):
    hiddens: list[int]
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for hidden in self.hiddens:
            x = nn.Dense(hidden)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class ResnetStack(nn.Module):
    """ResNet stack module.

    x -> Conv -> (maxpool) -> [ResNet Block] x num_blocks -> out
    """

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding="SAME",
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding="SAME",
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class ImpalaEncoder(nn.Module):
    """IMPALA encoder.

    x -> [ [ResNet Stack -> dropout] x stack_sizes ] -> ReLU -> LN -> Flatten -> MLP -> out

    """

    width: int = 1

    stack_sizes: tuple = (16, 32, 32)
    """Determines the number of output channels for each ResNet stack."""

    num_blocks: int = 2
    """Number of ResNet blocks per ResNet stack."""

    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        if x.dtype == jnp.uint8:
            logger.info("uint8 image, normalizing to [0, 1]")
            x = x.astype(jnp.float32) / 255.0
        else:
            logger.info("{} image, assuming already in [0, 1]", x.dtype)

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = ImpalaMLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class ImpalaMLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


encoder_modules = {
    "impala": ImpalaEncoder,
    "impala_debug": ft.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": ft.partial(ImpalaEncoder, num_blocks=1),
    "impala_large": ft.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
