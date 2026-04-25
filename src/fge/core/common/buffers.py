from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, override

import numpy as np
from et.decorators.timeit import timeit
from gymnasium import Space
from numpy.random import Generator

from fge.core.common.extract import get_action_dim, get_obs_shape
from fge.core.common.type_aliases import (
    Actions,
    CmdpRbSamples,
    Cost,
    Dones,
    EbmRbSamples,
    InitialStateMdpRbSamples,
    MdpRbSamples,
    Observations,
    RestartMdpRbSamples,
    Rewards,
)


class MdpReplayBuffer:
    def __init__(
        self, observation_space: Space, action_space: Space, capacity: int, **kwargs
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        # Override observation shape if necessary
        if "obs_shape" in kwargs:
            self.obs_shape = kwargs["obs_shape"]
        else:
            self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        self.observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(
        self,
        observation: Observations,
        action: Actions,
        next_observation: Observations,
        done: Dones,
        reward: Rewards,
        **kwargs
    ) -> Dict[Any, Any]:
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.next_observations[self.idx] = next_observation
        self.dones[self.idx] = done
        self.rewards[self.idx] = reward

        add_info = {
            "idx": self.idx,
            "size": self.size,
        }

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return add_info

    def sample(
        self, batch_size: int, rng: Generator, **kwargs
    ) -> [MdpRbSamples, Generator, Dict[Any, Any]]:
        idx = rng.integers(0, self.size, size=batch_size, endpoint=False)

        sample_info = {
            "idx": idx,
        }

        return (
            MdpRbSamples(
                observations=self.observations[idx],
                actions=self.actions[idx],
                next_observations=self.next_observations[idx],
                dones=self.dones[idx],
                rewards=self.rewards[idx],
            ),
            rng,
            sample_info,
        )


class CmdpReplayBuffer:
    def __init__(
        self, observation_space: Space, action_space: Space, capacity: int, **kwargs
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        self.observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(
        self,
        observation: Observations,
        action: Actions,
        next_observation: Observations,
        done: Dones,
        reward: Rewards,
        cost: Cost,
        **kwargs
    ) -> Dict[Any, Any]:
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.next_observations[self.idx] = next_observation
        self.dones[self.idx] = done
        self.rewards[self.idx] = reward
        self.costs[self.idx] = cost

        # Save add info in case this class is inherited
        add_info = {
            "idx": self.idx,
            "size": self.size,
        }

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return add_info

    def sample(
        self, batch_size: int, rng: Generator, **kwargs
    ) -> [CmdpRbSamples, Generator]:
        idx = rng.integers(0, self.size, size=batch_size, endpoint=False)

        return (
            CmdpRbSamples(
                observations=self.observations[idx],
                actions=self.actions[idx],
                next_observations=self.next_observations[idx],
                dones=self.dones[idx],
                rewards=self.rewards[idx],
                costs=self.costs[idx],
            ),
            rng,
        )


class EbmMdpReplayBuffer(MdpReplayBuffer):
    def __init__(
        self, observation_space: Space, action_space: Space, capacity: int, **kwargs
    ):
        super().__init__(observation_space, action_space, capacity, **kwargs)

        # Non-collision trajectories
        self.pos_observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        # Collision trajectories
        self.neg_observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)

        self.pos_idx = 0
        self.pos_size = 0
        self.neg_idx = 0
        self.neg_size = 0

        self.ep_buf = []  # Store the buffer indices of the episodes

    def ready_to_train(self):
        # Only ready to train if both positive and negative samples exist in the buffer
        return self.pos_size > 0 and self.neg_size > 0

    def add(
        self,
        observation: Observations,
        action: Actions,
        next_observation: Observations,
        done: Dones,
        reward: Rewards,
        info: Dict[Any, Any] = None,
        **kwargs
    ) -> Dict[Any, Any]:
        assert info is not None, "Info must be provided for EBM replay buffer"

        add_info = super().add(
            observation, action, next_observation, done, reward, **kwargs
        )

        if not done:
            return add_info

        # Flush
        assert "cost" in info, "Key 'cost' not found in info"

        # Consider case where we collided. NOTE: Only works with sparse reach/avoid tasks
        if info["cost"] > 0:
            self.neg_observations[self.neg_idx : self.neg_idx + self.idx] = (
                self.observations[: self.idx]
            )
            self.neg_idx += self.idx
            self.neg_size += self.idx

            # Reset the main buffer
            self.idx = 0
            self.size = 0

            # Reset add info
            add_info["idx"] = self.idx
            add_info["size"] = self.size

        else:
            self.pos_observations[self.pos_idx : self.pos_idx + self.idx] = (
                self.observations[: self.idx]
            )
            self.pos_idx += self.idx
            self.pos_size += self.idx

            # Reset the main buffer
            self.idx = 0
            self.size = 0

            # Reset add info
            add_info["idx"] = self.idx
            add_info["size"] = self.size

        return add_info

    def sample(
        self, batch_size: int, rng: Generator
    ) -> [EbmRbSamples, Generator, Dict[Any, Any]]:

        # Sample even amount of positive and negative samples with replacement
        pos_idx = rng.integers(0, self.pos_size, size=batch_size, endpoint=False)
        neg_idx = rng.integers(0, self.neg_size, size=batch_size, endpoint=False)

        sample_info = {
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
        }

        return (
            EbmRbSamples(
                pos_observations=self.pos_observations[pos_idx],
                neg_observations=self.neg_observations[neg_idx],
            ),
            rng,
            sample_info,
        )


class StateMdpReplayBuffer(MdpReplayBuffer):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        capacity: int,
        state_shape: tuple,
        **kwargs
    ):
        super().__init__(observation_space, action_space, capacity, **kwargs)
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)

    def add(
        self,
        observation: Observations,
        action: Actions,
        next_observation: Observations,
        done: Dones,
        reward: Rewards,
        **kwargs
    ) -> Dict[Any, Any]:
        add_info = super().add(
            observation, action, next_observation, done, reward, **kwargs
        )
        assert "state" in kwargs, "State must be provided"
        self.states[add_info["idx"]] = kwargs["state"]
        return add_info

    @override
    def sample(
        self, batch_size: int, rng: Generator, **kwargs
    ) -> [RestartMdpRbSamples, Generator, Dict[Any, Any]]:
        _, rng, sample_info = super().sample(batch_size, rng, **kwargs)
        idx = sample_info["idx"]
        return (
            RestartMdpRbSamples(
                state=self.states[idx],
                observations=self.observations[idx],
            ),
            rng,
            sample_info,
        )


class NonzeroCostTrajBuffer:
    """
    Stores non-zero cost trajectories only
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        capacity: int,
        state_shape: tuple,
        **kwargs
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity
        self.state_shape = state_shape

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        self.rb_idxs = np.zeros(
            (capacity, 1), dtype=int
        )  # The index of the sample in the main replay buffer
        self.observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.nonzero_costs = np.zeros((capacity, 1), dtype=np.float32)
        self.traj_ids = np.zeros(
            (capacity, 1), dtype=np.int32
        )  # Trajectory id of the sample
        self.solved = np.zeros(
            (capacity, 1), dtype=int
        )  # Whether the trajectory is solved
        # self.inject = np.zeros((capacity, 1), dtype=bool)
        self.inject = {}  # {traj_id: idxs}

        self.ep_states = []

        self.idx = 0
        self.size = 0
        self.traj_idx = 1

    def update_solved(self, traj_id, solved: bool):
        self.solved[self.traj_ids == traj_id] = np.clip(
            self.solved[self.traj_ids == traj_id] + int(solved), 0, 1
        )

    def update_inject_by_traj_id(self, traj_ids, injects):
        # Shape idxs: (batch_size,)
        # Shape injects: (batch_size,)
        for i, traj_id in enumerate(traj_ids):
            idxs = np.where(self.traj_ids == traj_id)[0]
            if len(idxs) > 0:
                self.inject[traj_id] = idxs

    def get_idxs_given_traj_id(self, traj_id):
        return self.inject[traj_id]

    def get_traj_ids_given_idxs(self, idxs):
        # Assert that all idxs < self.size
        assert np.all(idxs < self.size), "All idxs must be less than size"
        return np.unique(self.traj_ids[idxs].squeeze())

    def num_injects(self):
        if len(self.inject) == 0:
            return 0
        return np.concatenate(list(self.inject.values()), axis=0).shape[0]

    def has_injects(self):
        return len(self.inject)

    def buffer_add(
        self,
        rb_idx: int,
        observation: Observations,
        reward: Rewards,
        done: Dones,
        info: Dict[Any, Any] = None,
        reset_traj_id: Any = None,
        **kwargs
    ) -> Dict[Any, Any]:
        assert info is not None, "Info must be provided for Restart buffer"
        assert "state" in kwargs, "State must be provided for Restart buffer"

        add_info = {
            "idx": self.idx,
            "size": self.size,
            "flushed": False,
        }

        state = kwargs["state"]
        self.ep_states.append(
            (
                deepcopy(rb_idx),
                deepcopy(observation),
                deepcopy(reward),
                deepcopy(state),
            )
        )

        if not done:
            return add_info

        # Flush
        assert "cost" in info, "Key 'cost' not found in info"
        # check reset traj id to make sure we only save trajs starting from initial state dist.
        if info["cost"] > 0 and reset_traj_id is None:
            for rb_idx, observation, reward, state in self.ep_states[
                :-2
            ]:  # Exclude collision state
                self.rb_idxs[self.idx] = rb_idx
                self.observations[self.idx] = observation
                self.rewards[self.idx] = reward
                self.states[self.idx] = state
                self.nonzero_costs[self.idx] = 1
                self.traj_ids[self.idx] = self.traj_idx
                self.solved[self.idx] = 0  # Not solved

                self.idx = (self.idx + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)

            # Reset add info
            add_info["idx"] = self.idx
            add_info["size"] = self.size
        add_info["flushed"] = True

        self.ep_states = []
        self.traj_idx += 1

        return add_info

    def sample(
        self, batch_size: int, rng: Generator, inject_only=False, **kwargs
    ) -> [RestartMdpRbSamples, Generator, Dict[Any, Any]]:
        if inject_only:
            assert self.has_injects(), "No inject samples in the buffer"
            inject_idxs = np.concatenate(list(self.inject.values()), axis=0)
            idx = rng.integers(0, len(inject_idxs), size=batch_size, endpoint=False)
            idx = inject_idxs[idx]
        else:
            idx = rng.integers(0, self.size, size=batch_size, endpoint=False)

        sample_info = {
            "idx": idx,
            "traj_id": self.traj_ids[idx].squeeze(),
            "rb_idx": self.rb_idxs[idx].squeeze(),
        }
        return (
            RestartMdpRbSamples(
                state=self.states[idx],
                observations=self.observations[idx],
            ),
            rng,
            sample_info,
        )


##### PLR #####
class PLRResetBuffer(MdpReplayBuffer):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        capacity: int,
        state_shape: tuple,
        **kwargs
    ):
        super().__init__(observation_space, action_space, capacity, **kwargs)
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.S = np.zeros(capacity, dtype=np.float32)  # Level scores
        self.C = np.zeros(capacity, dtype=np.float32)  # Visit counts

    def add(self, observation, action, next_observation, done, reward, **kwargs):
        add_info = super().add(
            observation, action, next_observation, done, reward, **kwargs
        )
        assert "state" in kwargs, "State must be provided for Restart buffer"
        self.states[add_info["idx"]] = kwargs["state"]
        self.S[add_info["idx"]] = kwargs["score"]
        self.C[add_info["idx"]] = kwargs["c"]
        return add_info

    def h(self, S):
        # Compute ranks which is the order of the scores, where is the largest score
        # Need to argsort twice since
        # np.argsort(-S): returns the indices that would sort the array descending.
        # np.argsort(...) again gives the rank (i.e., position in the sorted array) for each element in the original array.
        ranks = np.argsort(np.argsort(-S)) + 1
        return 1 / ranks

    def sample_level(
        self, rho: float, beta: float, c: int, rng: Generator
    ) -> [np.ndarray, Dict[Any, Any]]:
        """
        Compute P_replay(l_i)

        c: global episode counter
        """
        S = self.S[: self.size]
        C = self.C[: self.size]

        # Compute P_S(l_i | \Lambda_seen, S)
        h_scores = self.h(S)
        P_S = np.power(h_scores, 1 / beta) / np.sum(np.power(h_scores, 1 / beta))

        # Compute P_C(l_i | \Lambda_seen, C, c)
        P_C = (c - C) / np.sum(c - C)

        # Compute P_replay(l_i)
        P_replay = (1 - rho) * P_S + rho * P_C
        P_replay = (
            P_replay / P_replay.sum()
        )  # Renormalize due to numerical imprecisions

        # Sample from P_replay
        reset_idx = rng.choice(np.arange(self.size), p=P_replay)

        sample_info = {
            "reset_idx": reset_idx,
        }

        return self.states[reset_idx], sample_info


###############


class InitialStateMdpReplayBuffer(StateMdpReplayBuffer):
    """
    Keep track of all initial reset states, and whether the trajectory was safe or not
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        capacity: int,
        state_shape: tuple,
        **kwargs
    ):
        super().__init__(
            observation_space, action_space, capacity, state_shape, **kwargs
        )
        self.is_safe = np.zeros(
            capacity, dtype=np.float32
        )  # Whether the initial state is safe

    def add(
        self, observation: Observations, is_safe: int, **kwargs  # 0 or 1
    ) -> Dict[Any, Any]:

        # Dummy
        action = np.zeros_like(self.action_dim)
        next_observation = np.zeros_like(self.obs_shape)
        done = False
        reward = 0

        add_info = super().add(
            observation, action, next_observation, done, reward, **kwargs
        )

        self.is_safe[add_info["idx"]] = is_safe

        return add_info

    @override
    def sample(
        self, batch_size: int, rng: Generator, **kwargs
    ) -> [InitialStateMdpRbSamples, Generator, Dict[Any, Any]]:
        _, rng, sample_info = super().sample(batch_size, rng, **kwargs)
        idx = sample_info["idx"]
        return (
            InitialStateMdpRbSamples(
                observations=self.observations[idx],
                is_safe=self.is_safe[idx],
            ),
            rng,
            sample_info,
        )
