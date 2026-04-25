import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from og.dyn_types import BObs, Obs


class ObsCircBuf:
    """A circular buffer storing observations (arrays) backed by a numpy array.
    If the buffer is full, the oldest observation will be overwritten.
    """

    def __init__(self, capacity: int, obs: Obs):
        self.capacity = capacity
        self.is_obs_array = isinstance(obs, (jnp.ndarray, np.ndarray))
        self.obs_pytree_def = jtu.tree_structure(obs)

        if self.is_obs_array:
            self.data = np.zeros((capacity,) + obs.shape, dtype=np.float32)
        else:
            # Convert it to a leaf,
            obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
            # Create a batch for each leaf
            self.data = [np.zeros((capacity,) + leaf.shape, dtype=leaf.dtype) for leaf in obs_leaf]

        self.head = 0
        self.size = 0

        self.rng = np.random.default_rng(seed=123456)

    def push(self, obs: Obs):
        if self.is_obs_array:
            self.data[self.head] = obs
        else:
            obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
            for ii, leaf in enumerate(obs_leaf):
                self.data[ii][self.head] = leaf

        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self) -> BObs:
        """Get the entire buffer. If the buffer is not full, then randomly sample to fill it."""
        if self.size < self.capacity:
            b_idx = self.rng.choice(self.size, self.capacity, replace=True)

            if self.is_obs_array:
                return self.data[b_idx]
            else:
                obs_leaves = [leaf[b_idx] for leaf in self.data]
                return jtu.tree_unflatten(self.obs_pytree_def, obs_leaves)

        if self.is_obs_array:
            return self.data
        else:
            obs_leaves = [leaf[: self.size] for leaf in self.data]
            return jtu.tree_unflatten(self.obs_pytree_def, obs_leaves)
