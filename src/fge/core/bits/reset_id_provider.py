from typing import Self

import jax.numpy as jnp
from flax import struct
from og.jax_types import BBool, BInt, IntScalar
from og.treenode_utils import prettynode


@prettynode
class ResetIDProvider(struct.PyTreeNode):
    next_id: IntScalar

    @staticmethod
    def create():
        zero = jnp.array(0, dtype=jnp.int32)
        return ResetIDProvider(zero)

    def get_id(self) -> tuple[Self, IntScalar]:
        return self.replace(next_id=self.next_id + 1), self.next_id

    def get_ids(self, num: int) -> tuple[Self, IntScalar]:
        b_ids = jnp.arange(num) + self.next_id
        assert b_ids.shape == (num,)
        return self.replace(next_id=self.next_id + num), b_ids

    def get_masked_ids(self, b_valid: BBool) -> tuple[Self, BInt]:
        """Return unique ids, but only for the locations where b_valid is True.

        b_valid  = [0 0 1 0 1 1 0]
        b_cumsum = [0 0 1 1 2 3 3]
        b_idx    = [0 0 0 0 1 2 2]
        """
        assert b_valid.ndim == 1

        b_cumsum = jnp.cumsum(b_valid)
        n_new_idxs = b_cumsum[-1]
        b_idx_offset = jnp.maximum(0, b_cumsum - 1)
        b_ids = self.next_id + b_idx_offset
        assert b_ids.shape == b_valid.shape
        return self.replace(next_id=self.next_id + n_new_idxs), b_ids
