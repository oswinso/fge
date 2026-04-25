import ipdb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from og.dyn_types import BObs, Obs
from og.jax_types import BBool


class LabeledObsCircBuf:
    """A circular buffer storing observations (arrays) and labels (booleans) backed by a numpy array.
    If the buffer is full, the oldest observation will be overwritten.
    """

    def __init__(self, capacity: int, ic: np.ndarray, obs: Obs):
        self.capacity = capacity
        self.is_obs_array = isinstance(obs, (jnp.ndarray, np.ndarray))
        self.obs_pytree_def = jtu.tree_structure(obs)

        self.ics = np.zeros((capacity,) + ic.shape, dtype=ic.dtype)

        if self.is_obs_array:
            self.data = np.zeros((capacity,) + obs.shape, dtype=np.float32)
        else:
            # Convert it to a leaf,
            obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
            # Create a batch for each leaf
            self.data = [np.zeros((capacity,) + leaf.shape, dtype=leaf.dtype) for leaf in obs_leaf]

        self.labels = np.zeros((capacity,), dtype=np.bool_)
        self.head = 0
        self.size = 0

    def push(self, ic: np.ndarray, obs: Obs, label: bool):
        self.ics[self.head] = ic

        if self.is_obs_array:
            self.data[self.head] = obs
        else:
            obs_leaf: list[np.ndarray] = jtu.tree_leaves(obs)
            for ii, leaf in enumerate(obs_leaf):
                self.data[ii][self.head] = leaf

        self.labels[self.head] = label
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, b_ic: np.ndarray, b_obs: BObs, b_label: BBool):
        n_obs = len(b_obs)
        assert n_obs == len(b_label), "Number of observations and labels must match."
        assert n_obs <= self.capacity, "Batch size exceeds buffer capacity."

        # Calculate the number of elements to write in the current batch
        size1 = min(n_obs, self.capacity - self.head)
        size2 = n_obs - size1

        self.ics[self.head: self.head + size1] = b_ic[:size1]
        self.ics[:size2] = b_ic[size1:]

        # Write the observations and labels to the buffer
        if self.is_obs_array:
            self.data[self.head : self.head + size1] = b_obs[:size1]
            self.data[:size2] = b_obs[size1:]
        else:
            b_obs_leaf: list[np.ndarray] = jtu.tree_leaves(b_obs)
            for ii, leaf in enumerate(b_obs_leaf):
                self.data[ii][self.head : self.head + size1] = leaf[:size1]
                self.data[ii][:size2] = leaf[size1:]

        self.labels[self.head : self.head + size1] = b_label[:size1]
        self.labels[:size2] = b_label[size1:]

        # Update the head and size
        self.head = (self.head + n_obs) % self.capacity
        self.size = min(self.size + n_obs, self.capacity)

    def sample(self, rng: np.random.Generator, size: int, replace: bool = True) -> tuple[BObs, BBool]:
        """Sample `size` observations and labels from the buffer."""
        b_idx = rng.choice(self.size, size, replace=replace)
        if self.is_obs_array:
            b_obs = self.data[b_idx]
        else:
            b_obs_leaf = [leaf[b_idx] for leaf in self.data]
            b_obs = jtu.tree_unflatten(self.obs_pytree_def, b_obs_leaf)
        b_label = self.labels[b_idx]

        return b_obs, b_label

    def sample_ics(self, rng: np.random.Generator, size: int, replace: bool = True) -> np.ndarray:
        """Sample `size` initial conditions from the buffer."""
        b_idx = rng.choice(self.size, size, replace=replace)
        b_ics = self.ics[b_idx]
        b_label = self.labels[b_idx]
        return b_ics, b_label

    def get(self, size: int) -> tuple[BObs, BBool]:
        """Get the last `size` observations and labels."""
        assert size <= self.size, "Requested size exceeds current buffer size."

        # Calculate the starting index for the data retrieval
        start = (self.head - size) % self.capacity

        size1 = min(size, self.capacity - start)
        size2 = size - size1

        # Retrieve the observations and labels
        if self.is_obs_array:
            b_obs = np.concatenate((self.data[start:], self.data[:size2]), axis=0)
        else:
            obs_leaf = [np.concatenate((leaf[start:], leaf[:size2]), axis=0) for leaf in self.data]
            b_obs = jtu.tree_unflatten(self.obs_pytree_def, obs_leaf)
        b_labels = np.concatenate((self.labels[start:], self.labels[:size2]), axis=0)

        assert len(b_obs) == len(b_labels) == size

        return b_obs, b_labels
