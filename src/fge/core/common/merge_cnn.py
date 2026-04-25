import functools as ft
from typing import Sequence, Type

import flax.linen as nn
import jax.numpy as jnp
from og.networks.network_utils import HidSizes

from fge.core.common.ppo_nets import MLP
from fge.core.envs.kinetix import lander


class MergeCNN(nn.Module):
    encoder: nn.Module
    hid_sizes: HidSizes

    @nn.compact
    def __call__(self, x: lander.LanderObs):
        img_feat = self.encoder(x.img)
        cat_feat = jnp.concatenate([img_feat, x.aux], axis=-1)
        out = MLP(self.hid_sizes, act_final=True)(cat_feat)
        return out


class MergeCNN2(nn.Module):
    """An ensemble of MergeCNNs with a shared encoder but different heads."""

    encoder: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(self, x: lander.LanderObs):
        # 1: Encode the image once.
        img_feat = self.encoder(x.img)
        cat_feat = jnp.concatenate([img_feat, x.aux], axis=-1)

        # 2: Ensemble of heads
        return self.head(cat_feat)
