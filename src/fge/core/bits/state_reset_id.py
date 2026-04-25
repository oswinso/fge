from enum import IntEnum
from typing import Any, NamedTuple, Self

from flax import struct
from og.dyn_types import Obs
from og.jax_types import IntScalar

from fge.core.envs.jax_task import TimedState, TreeLeaves


class Source(IntEnum):
    BASE = 1
    """Sampled from the environment. Base distribution"""

    BUF_CI = 2
    """Sampled from the CI rehearsal buffer."""

    BUF_EXPLORE = 3
    """Sampled from the exploration buffer."""

    BASE_PREDCI = 4
    """Rejection sampling from the predicted CI distribution based on the base distribution."""

    def __str__(self):
        match self:
            case Source.BASE:
                return "Base"
            case Source.BUF_CI:
                return "CI"
            case Source.BUF_EXPLORE:
                return "Explore"
            case Source.BASE_PREDCI:
                return "OnPol"
            case _:
                return "Unk({})".format(self.value)

    @classmethod
    def labels_dict(cls) -> dict[Self, str]:
        return {e: str(e) for e in cls}

    @classmethod
    def colors_dict(cls) -> dict[Self, Any]:
        return {
            Source.BASE: "C0",
            Source.BUF_CI: "C1",
            Source.BUF_EXPLORE: "C2",
            Source.BASE_PREDCI: "C4",
        }

    @classmethod
    def zorder_dict(cls) -> dict[Self, Any]:
        return {
            Source.BASE: 3,
            Source.BUF_CI: 6,
            Source.BUF_EXPLORE: 4,
            Source.BASE_PREDCI: 5,
        }


class StateResetId(NamedTuple):
    """
    Represents a combination of a timed state and a reset identifier.

    This class is used to encapsulate the state of an environment along with
    a unique identifier for resets, which can be used to track and manage
    environment resets during reinforcement learning rollouts.

    Attributes:
        state (TimedState): The current state of the environment, including
            timing information.
        reset_id (IntScalar): A scalar value representing the reset identifier
            for the environment.
    """

    state: TimedState
    reset_id: IntScalar


LeafObs = tuple[TreeLeaves, Obs]
