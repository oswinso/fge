import time
from collections import deque
from itertools import zip_longest
from typing import Any, SupportsFloat

import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium.core import ActType, ObsType


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# RunningMeanStd code (from OpenAI baselines, stolen from PPO RLE github) - adapted for jax numpy
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = epsilon

    def update(self, x: jnp.ndarray):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


@jax.jit
def rms_norm(observations, rm, rv, bound):
    """
    Normalize the observations or rewards using the running mean and std for RND predictor and target networks

    According to RND paper section 2.4, they clip between -5 and 5 for features.
    """
    norm = (observations - rm) / jnp.sqrt(rv + 1e-8)
    return jnp.clip(norm, -bound, bound)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RecordEpisodeStatistics(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
     * time_queue: The time length of the last ``deque_size``-many episodes
     * return_queue: The cumulative rewards of the last ``deque_size``-many episodes
     * length_queue: The lengths of the last ``deque_size``-many episodes

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Removed vector environment support (see :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`) and add attribute ``time_queue``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        num_steps: int = float("inf"),  # For On-policy methods
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key for the episode statistics
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self._stats_key = stats_key

        self.episode_count = 0
        self.episode_start_time: float = -1
        self.episode_returns: float = 0.0
        self.episode_lengths: int = 0

        self.time_queue: deque[float] = deque(maxlen=buffer_length)
        self.return_queue: deque[float] = deque(maxlen=buffer_length)
        self.length_queue: deque[int] = deque(maxlen=buffer_length)

        self.num_steps = num_steps

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        obs, reward, terminated, truncated, info = super().step(action)

        self.episode_returns += reward
        self.episode_lengths += 1

        truncated = truncated or (self.episode_lengths >= self.num_steps)

        if terminated or truncated:
            assert self._stats_key not in info

            episode_time_length = round(
                time.perf_counter() - self.episode_start_time, 6
            )
            info[self._stats_key] = {
                "r": self.episode_returns,
                "l": self.episode_lengths,
                "t": episode_time_length,
            }

            self.time_queue.append(episode_time_length)
            self.return_queue.append(self.episode_returns)
            self.length_queue.append(self.episode_lengths)

            self.episode_count += 1
            self.episode_start_time = time.perf_counter()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment using seed and options and resets the episode rewards and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_start_time = time.perf_counter()
        self.episode_returns = 0.0
        self.episode_lengths = 0

        return obs, info


def interleave_arrays(*arrays):
    """
    Stolen from chatgpt. Just interleave arrays of different lengths.

    e.g.
    a = [1, 2, 3]
    b = [4, 5]

    interleave_arrays(a, b) -> [1, 4, 2, 5, 3]
    """
    # Use zip_longest to interleave while handling different lengths
    interleaved = [
        x for tup in zip_longest(*arrays, fillvalue=None) for x in tup if x is not None
    ]
    # Convert result to a NumPy array
    return interleaved


def find_env_wrapper(env, wrapper_type):
    # Helps find the intermediate env wrapper of a specific wrapper type. Returns by reference.
    current_env = env
    while hasattr(current_env, "env"):
        if isinstance(current_env, wrapper_type):
            return current_env
        current_env = current_env.env
    return None
