from typing import NamedTuple, Union

import numpy as np
from jaxtyping import Array, Float, Integer

Arr = Union[np.ndarray, Array]
Observations = Float[Arr, "batch_size obs_dim"]
Actions = Float[Arr, "batch_size action_dim"]
Dones = Float[Arr, "batch_size"]
Rewards = Float[Arr, "batch_size"]
Cost = Float[Arr, "batch_size"]
States = Float[Arr, "batch_size state_dim"]


class MdpRbSamples(NamedTuple):
    observations: Observations
    actions: Actions
    next_observations: Observations
    dones: Dones
    rewards: Rewards


class CmdpRbSamples(NamedTuple):
    observations: Observations
    actions: Actions
    next_observations: Observations
    dones: Dones
    rewards: Rewards
    costs: Cost


class EbmRbSamples(NamedTuple):
    pos_observations: Observations
    neg_observations: Observations


class RndMdpRbSamples(NamedTuple):
    observations: Observations
    actions: Actions
    next_observations: Observations
    dones: Dones
    rewards: Rewards
    int_rewards: Rewards


class RestartMdpRbSamples(NamedTuple):
    state: States
    observations: Observations = None


class InitialStateMdpRbSamples(NamedTuple):
    observations: Observations
    is_safe: Dones
