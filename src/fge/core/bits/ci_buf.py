import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from og.dyn_types import BObs, Obs
from og.jax_types import BBool
from og.tree_utils import make_batch

from fge.core.envs.jax_task import JaxTask, MinState_, TreeLeaves, leaf_index, leaf_set


class CIBuf:
    """A buffer that stores the state, observation and reset_id for different initial states.
    Uses reservoir sampling (Algorithm R) to store the data.
    """

    def __init__(
        self,
        capacity: int,
        leaves: TreeLeaves,
        obs: Obs,
        seed: int = 54321,
    ):
        self.state: TreeLeaves = tuple(make_batch(arr, capacity) for arr in leaves)

        self.is_obs_array = isinstance(obs, (jnp.ndarray, np.ndarray))
        self.obs_pytree_def = jtu.tree_structure(obs)
        if self.is_obs_array:
            self.obs = np.zeros((capacity,) + obs.shape, dtype=np.float32)
        else:
            # Convert it to a leaf,
            obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
            # Create a batch for each leaf
            self.obs = [np.zeros((capacity,) + leaf.shape, dtype=leaf.dtype) for leaf in obs_leaf]

        self.reset_id = np.full((capacity,), -1, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

        self.n_pushed = 0
        self.capacity = capacity

    @property
    def size(self):
        return min(self.n_pushed, self.capacity)

    def push(self, state: TreeLeaves, obs: Obs, reset_id: int):
        """Push a new item into the buffer."""
        if self.n_pushed < self.capacity:
            leaf_set(self.state, self.n_pushed, state)

            if self.is_obs_array:
                self.obs[self.n_pushed] = obs
            else:
                obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
                for ii, leaf in enumerate(obs_leaf):
                    self.obs[ii][self.n_pushed] = leaf

            self.reset_id[self.n_pushed] = reset_id
            self.n_pushed += 1
        else:
            # If the buffer is full, replace an existing item with a probability of size / (size + 1)
            idx = self.rng.integers(0, self.n_pushed + 1)
            if idx < self.capacity:
                leaf_set(self.state, idx, state)
                if self.is_obs_array:
                    self.obs[idx] = obs
                else:
                    obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
                    for ii, leaf in enumerate(obs_leaf):
                        self.obs[ii][idx] = leaf
                self.reset_id[idx] = reset_id

    def get_state(self):
        sl = slice(0, self.size)
        return leaf_index(self.state, sl)

    def get_obs(self, b_idx: np.ndarray | None = None):
        if self.is_obs_array:
            out = self.obs[: self.size]

            if b_idx is not None:
                out = out[b_idx]

            return out
        else:
            obs_leaf = [leaf[: self.size] for leaf in self.obs]

            if b_idx is not None:
                obs_leaf = [leaf[b_idx] for leaf in obs_leaf]

            return jtu.tree_unflatten(self.obs_pytree_def, obs_leaf)

    def get_reset_id(self):
        return self.reset_id[: self.size]
