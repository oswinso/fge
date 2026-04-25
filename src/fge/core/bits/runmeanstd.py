import jax.numpy as jnp
from flax import struct
from og.treenode_utils import prettynode


@prettynode
class RunningMeanStd(struct.PyTreeNode):
    """"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    handle_std0: bool = struct.field(pytree_node=False)

    @staticmethod
    def create(arr, handle_std0: bool = False) -> "RunningMeanStd":
        mean = jnp.zeros_like(arr)
        var = jnp.zeros_like(arr)
        count = jnp.array(0, dtype=jnp.int32)
        return RunningMeanStd(mean, var, count, handle_std0=handle_std0)

    def update_from_moments(
        self, batch_mean, batch_var, batch_count
    ) -> "RunningMeanStd":
        assert batch_count > 1
        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        return self.replace(mean=new_mean, var=new_var, count=new_count)

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    @property
    def std(self):
        if self.handle_std0:
            var_zero = self.var == 0
            std = jnp.sqrt(self.var)
            std = jnp.where(var_zero, 1, std)
        else:
            std = jnp.sqrt(self.var)

        return std
