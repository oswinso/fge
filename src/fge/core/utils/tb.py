import jax.numpy as jnp
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def log_dict_tb(writer: SummaryWriter, d: dict[str, float], global_step: int):
    for k, v in d.items():
        if isinstance(v, jnp.ndarray):
            v = np.array(v)

        writer.add_scalar(k, v, global_step=global_step)
