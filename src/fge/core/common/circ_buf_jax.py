import functools as ft
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from attrs import define
from flax import struct
from og.cfg_utils import Cfg
from og.dyn_types import Control, Obs, TControl, TFloat, THFloat, TObs
from og.jax_types import FloatScalar, IntScalar
from og.jax_utils import concat_at_end, concat_at_front
from og.rng import PRNGKey
from og.tree_utils import make_batch_pytree

from fge.core.envs.jax_task import JaxTask, StepOutput, TimedState

Item_ = TypeVar("Item_")


class CircOnlyInJax(struct.PyTreeNode, Generic[Item_]):
    """A circular buffer that only allows adding items in batch, for pytrees. Items cannot be popped, only sampled."""

    """Write index."""
    head: int

    """Size of the buffer. Starts at 0."""
    size: int

    """Total number of items that have been added to the buffer. Not clipped."""
    id: int

    """Total capacity of the buffer before wrapping."""
    capacity: int = struct.field(pytree_node=False)

    data: Item_

    @staticmethod
    def create(capacity: int, item: Item_):
        data = make_batch_pytree(item, capacity, fill_value=0, whichnp=jnp)
        return CircOnlyInJax(head=0, size=0, id=0, capacity=capacity, data=data)

    def add(self, item: Item_):
        """Add an item to the buffer."""
        idx = self.head
        data_new = jtu.tree_map(
            lambda d_arr, item_arr: d_arr.at[idx].set(item_arr), self.data, item
        )
        head_new = (self.head + 1) % self.capacity
        size_new = jnp.minimum(self.size + 1, self.capacity)
        id_new = self.id + 1
        return self.replace(head=head_new, size=size_new, id=id_new, data=data_new)

    def add_batch(self, b_item: Item_, num: int):
        """Add num items to the buffer."""
        b_idx = (self.head + jnp.arange(num)) % self.capacity
        data_new = jtu.tree_map(
            lambda d_arr, item_arr: d_arr.at[b_idx].set(item_arr), self.data, b_item
        )
        head_new = (self.head + num) % self.capacity
        size_new = jnp.minimum(self.size + num, self.capacity)
        id_new = self.id + num
        return self.replace(head=head_new, size=size_new, id=id_new, data=data_new)

    def add_all(self, b_item: Item_, num: int):
        """Fill up the entire buffer, discarding all of the current contents."""
        assert num == self.capacity

        # Make sure that b_item has the same structure and shape as self.data.
        assert jtu.tree_structure(b_item) == jtu.tree_structure(self.data)
        data_leaves = jtu.tree_leaves(self.data)
        new_leaves = jtu.tree_leaves(b_item)
        assert len(data_leaves) == len(new_leaves)
        for d_arr, new_arr in zip(data_leaves, new_leaves):
            assert d_arr.shape == new_arr.shape
            assert d_arr.dtype == new_arr.dtype

        head_new = 0
        size_new = num
        id_new = self.id + num
        return self.replace(head=head_new, size=size_new, id=id_new, data=b_item)

    def sample(
        self,
        key: PRNGKey,
        num: int,
        replace: bool = True,
        frac_latest: float | None = None,
    ) -> Item_:
        """Sample num items from the buffer without popping."""

        assert replace is True

        key_int, key_recent = jr.split(key)

        # Sample indices from the buffer
        b_idx = jr.randint(key_int, (num,), 0, self.size - 1)

        if frac_latest is not None:
            # Replace a random number of those indices with the most recent index.
            b_uselatest = jr.bernoulli(key_recent, frac_latest, shape=(num,))
            b_idx = jnp.where(b_uselatest, self.head - 1, b_idx)

        # Get the data at those indices
        data_out = jtu.tree_map(lambda d_arr: d_arr[b_idx], self.data)
        return data_out

    def most_recent(self, num: int) -> Item_:
        """Get the most recent num items from the buffer without popping."""
        b_idx = (self.head - 1 - jnp.arange(num)) % self.capacity
        data_out = jtu.tree_map(lambda d_arr: d_arr[b_idx], self.data)
        return data_out

    def all_valid(self):
        """Return all valid data."""
        data_valid = jtu.tree_map(lambda d_arr: d_arr[: self.size], self.data)
        return data_valid


class CircFIFOJax(struct.PyTreeNode):
    """A jax-compatible circular buffer that allows adding and popping in batch for pytrees."""

    """Write index."""
    head: int

    """Read index."""
    tail: int

    """Size of the buffer. Starts at 0."""
    size: int

    """Total capacity of the buffer before wrapping."""
    capacity: int

    data: Item_

    @staticmethod
    def create(capacity: int, item: Item_):
        data = jtu.tree_map(lambda x: jnp.zeros((capacity,) + x.shape), item)
        return CircFIFOJax(head=0, tail=0, size=0, capacity=capacity, data=data)

    def add_batch(self, b_item: Item_, num: int):
        """Add num items to the buffer."""
        b_idx = (self.head + jnp.arange(num)) % self.capacity
        data_new = jtu.tree_map(
            lambda d_arr, item_arr: d_arr.at[b_idx].set(item_arr), self.data, b_item
        )
        head_new = (self.head + num) % self.capacity
        size_new = jnp.minimum(self.size + num, self.capacity)
        return self.replace(head=head_new, size=size_new, data=data_new)

    def get_batch(self, num: int) -> Item_:
        """Get the next num items from the buffer without popping."""
        b_idx = (self.tail + jnp.arange(num)) % self.capacity
        data_out = jtu.tree_map(lambda d_arr: d_arr[b_idx], self.data)
        return data_out

    def advance(self, num: int):
        """Advance the read index by num items."""
        tail_new = (self.tail + num) % self.capacity
        size_new = jnp.maximum(self.size - num, 0)
        return self.replace(tail=tail_new, size=size_new)


class CircLIFOJax(struct.PyTreeNode, Generic[Item_]):
    """A jax-compatible last-in first-out stack that allows adding and popping in batch for pytrees."""

    size: int
    """Stack pointer. Points to the next item to replace. Equal to the unbounded size. [0, infty)"""

    n_overwrite: int
    """Number of items we have overwritten. After mod capacity, should point to the oldest item."""

    n_valid: int
    """Number of valid (not uninitialized) items in the stack."""

    capacity: int
    """Total capacity of the stack before wrapping."""

    n_wraparound: int
    """Number of times we advanced too far (size negative) and wrapped around."""

    data: Item_

    @staticmethod
    def create(capacity: int, item: Item_):
        data = make_batch_pytree(item, capacity, fill_value=0, whichnp=jnp)
        return CircLIFOJax(
            size=0,
            n_overwrite=0,
            n_valid=0,
            capacity=capacity,
            n_wraparound=0,
            data=data,
        )

    def add_batch(self, b_item: Item_, num: int, first_at_top: bool = True):
        """
        :param first_at_top: If True, then the most recently added item is at the first index.
        """
        indices = jnp.arange(num)
        b_idx = (self.size + indices) % self.capacity
        if first_at_top:
            b_idx = b_idx[::-1]
        data_new = jtu.tree_map(
            lambda d_arr, item_arr: d_arr.at[b_idx].set(item_arr), self.data, b_item
        )
        size_new = self.size + num
        size_cap_new = jnp.minimum(size_new, self.capacity)
        valid_new = jnp.maximum(size_cap_new, self.n_valid)

        n_newly_overwritten = (size_new - self.n_overwrite) % self.capacity
        n_overwrite_new = self.n_overwrite + n_newly_overwritten

        return self.replace(
            size=size_new, n_overwrite=n_overwrite_new, n_valid=valid_new, data=data_new
        )

    def get_batch(self, num: int, first_at_top: bool = True) -> Item_:
        """Get the next num items from the stack without popping.
        If the stack is empty, then return only items that have been inserted (and popped) before.

        :param first_at_top: If True, then the first index is the most recently added item.
        """
        indices = jnp.arange(num)
        b_idx = (self.size - indices - 1) % self.n_valid
        if not first_at_top:
            b_idx = b_idx[::-1]
        data_out = jtu.tree_map(lambda d_arr: d_arr[b_idx], self.data)
        return data_out

    def advance(self, num: int):
        """Advance the stack pointer by num items."""
        size_new = self.size - num
        # If it goes below n_overwrite.... TODO
        wraparound = size_new < 0
        size_new = jnp.where(wraparound, self.n_valid, size_new)
        n_wraparound = jnp.where(wraparound, self.n_wraparound + 1, self.n_wraparound)
        return self.replace(size=size_new, n_wraparound=n_wraparound)
