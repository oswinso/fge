import functools as ft
from types import ModuleType
from typing import Any, NamedTuple, TypeVar

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from gymnasium.spaces import Space
from loguru import logger
from og.dyn_types import BObs, BState
from og.jax_types import BFloat, BoolScalar
from og.rng import PRNGKey


class EvalRegionInfo(NamedTuple):
    name: str
    """Name of the region"""

    n_per_region: int
    """How many samples are from this region"""


@jdc.pytree_dataclass
class TimedState:
    step: int

    source: int
    """From which source the sample was sampled /generated from."""

    def __post_init__(self): ...


State_ = TypeVar("State_", bound=TimedState)
MinState_ = TypeVar("MinState_", bound=TimedState)
Obs_ = TypeVar("Obs_")

type TreeLeaves = tuple[jnp.ndarray, ...]

TreeLeaves_ = TypeVar("TreeLeaves_", bound=TreeLeaves)


def leaf_to_jax(leaf: TreeLeaves_) -> TreeLeaves_:
    return tuple(jnp.array(l) for l in leaf)


def leaf_index(leaf: TreeLeaves_, index: Any) -> TreeLeaves_:
    """We can directly index without using jtu.tree_map for speed."""
    return tuple(arr[index] for arr in leaf)


def leaf_index_copy(leaf: TreeLeaves_, index: Any) -> TreeLeaves_:
    """Same as leaf_index but makes a copy to prevent memory leaks due to references."""
    return tuple(arr[index].copy() for arr in leaf)


def leaf_set(leaves: TreeLeaves, index: Any, new_leaves: TreeLeaves):
    """Mutate in place."""
    for ii, new_leaf in enumerate(new_leaves):
        leaves[ii][index] = new_leaf


def leaf_jax_set(leaves: TreeLeaves, index: Any, new_leaves: TreeLeaves) -> TreeLeaves:
    return [l.at[index].set(new_leaves) for l, l_new in zip(leaves, new_leaves)]


def leaf_stack(leaf_list: list[TreeLeaves_], axis: int, which: ModuleType = np) -> TreeLeaves_:
    """We can directly stack without using jtu.tree_map for speed."""
    tuple_len = len(leaf_list[0])
    return tuple(which.stack([leaf[ii] for leaf in leaf_list], axis=axis) for ii in range(tuple_len))


def leaf_concat(leaf_list: list[TreeLeaves_], axis: int, which: ModuleType = np) -> TreeLeaves_:
    """We can directly concatenate without using jtu.tree_map for speed."""
    tuple_len = len(leaf_list[0])
    return tuple(which.concatenate([leaf[ii] for leaf in leaf_list], axis=axis) for ii in range(tuple_len))


# def leaf_stack(leaf_list: list[TreeLeaves_], axis: int, which: ModuleType = np) -> TreeLeaves_:
#     transposed = list(zip(*leaf_list))  # shape: tuple_len x len(leaf_list)
#     return tuple(which.stack(arrays, axis=axis) for arrays in transposed)


class StepOutput(NamedTuple):
    """
    Represents the output of a single environment step in a reinforcement learning task.

    Attributes:
        state (State_): The resulting state of the environment after the step.
        obs (Obs_): The observation corresponding to the new state.
        rew (float): The reward received for the step.
        term (bool): A flag indicating whether the episode has terminated.
        trunc (bool): A flag indicating whether the episode was truncated.
        info (dict): Additional information about the step, such as debugging or logging data.
    """

    state: State_
    obs: Obs_
    rew: float
    term: bool
    trunc: bool
    info: dict


class EvalStateInfo(NamedTuple):
    b_state: State_
    n_reset: int
    region_info: list[EvalRegionInfo]


class JaxTask:
    """Environment defined in jax, used for on-policy algorithms such as PPO."""

    def __init__(self, action_space: Space):
        self.action_space = action_space

    @property
    def eval_rollout_T(self) -> int:
        raise NotImplementedError("")

    def step(self, state: State_, action) -> StepOutput:
        raise NotImplementedError("")

    def get_obs(self, state: State_):
        raise NotImplementedError("")

    # @ft.cache
    def get_dummy_obs(self):
        cls = type(self)
        if not hasattr(cls, "obs_shape"):
            state = jax.jit(self.reset)(jr.PRNGKey(1234))
            obs = self.get_obs(state)

            if isinstance(obs, (np.ndarray, jnp.ndarray)):
                cls.obs_shape = obs.shape
            else:
                # For images.
                cls.obs = obs

        if hasattr(cls, "obs"):
            return cls.obs

        return np.zeros(cls.obs_shape)

    def get_dummy_nsf_obs(self):
        return self.get_dummy_obs()

    @property
    def use_ic_obs(self) -> bool:
        return False

    # def get_dummy_obs(self):
    #     return np.zeros(14)

    def reset(self, key: jr.PRNGKey) -> State_:
        raise NotImplementedError("")

    def reset_papereval(self, b_key: PRNGKey, num: int) -> State_:
        """Reset function, but for the paper."""
        raise NotImplementedError("")

    def reset_from_box(self, uniform: jnp.ndarray) -> State_:
        """Interface for allowing adversarial optimization over the initial state. [-1, 1]^n to state."""
        raise NotImplementedError("")

    def box_from_reset(self, state: State_):
        """Inverse of reset_from_box. Return the [-1, 1]^n that generated the state."""
        raise NotImplementedError("")

    def is_unsafe_custom(self, state: State_) -> BoolScalar:
        return False

    @property
    def x0_unif_shape(self) -> tuple[int, ...]:
        """Shape of uniform in reset_from_box."""
        raise NotImplementedError("")

    def get_contour_grid(self):
        raise NotImplementedError("")

    def get_minstate(self, state: State_) -> MinState_:
        """Given the state used for simulation, return a minimal representation of the state."""
        return state

    def compress_state(self, state: State_) -> Any:
        """Compress the state to save memory. Not necessarily lossless."""
        return state

    def _state_treedef(self) -> jtu.PyTreeDef:
        cls = type(self)
        if not hasattr(cls, "state_treedef"):
            state = self.reset(jr.PRNGKey(1234))
            treedef = jtu.tree_structure(state)
            cls.state_treedef = treedef

        treedef = cls.state_treedef
        return treedef

    def _obs_treedef(self) -> jtu.PyTreeDef:
        cls = type(self)
        if not hasattr(cls, "obs_treedef"):
            obs = self.get_dummy_obs()
            treedef = jtu.tree_structure(obs)
            cls.obs_treedef = treedef

        treedef = cls.obs_treedef
        return treedef

    @property
    def minstate_treedef(self):
        return self._state_treedef()

    def state_to_leaf(self, state: State_) -> TreeLeaves:
        return self.minstate_to_leaf(self.minify(state))

    def minstate_to_leaf(self, minstate: MinState_) -> TreeLeaves:
        leaves: list = jtu.tree_leaves(minstate)
        return tuple(leaves)

    def minify(self, state: State_) -> TreeLeaves:
        minstate = self.get_minstate(state)
        # Go one step further and convert it to a list of leaves, so we can save on the tree construction cost.
        return self.minstate_to_leaf(minstate)

    def leaf_to_minstate(self, leaves: TreeLeaves) -> MinState_:
        # Undo jtu.tree_leaves and get back the output of get_minstate.
        return jtu.tree_unflatten(self.minstate_treedef, leaves)

    def leaf_to_state(self, leaves: TreeLeaves) -> State_:
        return self.from_minstate(self.leaf_to_minstate(leaves))

    def leaf_to_obs(self, leaves: TreeLeaves) -> Obs_:
        if isinstance(leaves, (np.ndarray, jnp.ndarray)):
            return leaves

        return jtu.tree_unflatten(self._obs_treedef(), leaves)

    def from_minstate(self, minstate: MinState_) -> State_:
        return minstate

    def to_icval(self, state: State_) -> np.ndarray:
        """Project the state to the line / plane / surface for visualizing the initial conditions."""
        raise NotImplementedError("")

    def icval_bins(self, n_bins: int = 31) -> np.ndarray:
        raise NotImplementedError("")

    def label_ic(self, ax: plt.Axes):
        raise NotImplementedError("")

    def setup_trajplot(self, ax: plt.Axes):
        raise NotImplementedError("")

    def x0_from_box(self, uniform: jnp.ndarray) -> State_:
        """Given a batch of points in the initial condition space, return the state that corresponds to it."""
        raise NotImplementedError("")

    def box_from_x0(self, x0: jnp.ndarray) -> jnp.ndarray:
        """Given a batch of points in the initial condition space, return the box in [-1, 1]^n that contains it."""
        raise NotImplementedError("")

    def get_eval_states(self) -> EvalStateInfo:
        """Get the states used for evaluation.

        returns b_state,
        """
        raise NotImplementedError("")

    def get_eval_contour(self) -> tuple[BFloat, BObs, BState]:
        """Return info used for contour plot.

        b_ic, b_obs, b_state.
        """
        raise NotImplementedError("")

    def eval_ics(self) -> BFloat:
        """ICs for the eval states. Used for the rollout, but could be refactored to do without."""
        raise NotImplementedError("")
