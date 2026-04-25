from collections import defaultdict

import numpy as np

from fge.core.bits.collector import RolloutOutput
from fge.core.envs.jax_task import TreeLeaves, leaf_index


class X0Saver:
    def __init__(self):
        self._x0s: list[TreeLeaves] = []
        self._trajs_added = 0

    @property
    def trajs(self):
        return []

    @property
    def x0s(self):
        return self._x0s

    def clear_trajs(self):
        self._x0s = []

    def add_x0s(self, b_x0: TreeLeaves):
        b = b_x0[0].shape[0]

        for bb in range(b):
            x0 = leaf_index(b_x0, bb)
            self._x0s.append(x0)


class FastTrajSaver:
    """Trying to do TrajSaver, but as fast as possible, since this is the bottleneck.

    Speed it up by only saving the initial and final states, not the full trajectory.
    """

    def __init__(self):
        self._x0s: list[TreeLeaves] = []
        self._trajs: list[RolloutOutput] = []
        self._reset_ids: list[int] = []
        self._traj_lens: list[int] = []

        # List will only be empty or size 1.
        self._cur_x0s: dict[int, list[TreeLeaves]] = defaultdict(list)

        # The list will only be of the start and end state.
        self._cur_trajs: dict[int, list[RolloutOutput]] = defaultdict(list)

        self._cur_traj_lens: dict[int, int] = defaultdict(int)

        # Save a count of how many trajectories were added for each batch.
        self._trajs_count = []

        self._trajs_added = 0
        self._last_reset_id = -1

    @property
    def x0s(self):
        return self._x0s

    def add_full_traj(self, full_traj: RolloutOutput, traj_len: int):
        reset_id = full_traj.T_reset_id[0]
        assert np.all(reset_id == full_traj.T_reset_id)

        if reset_id == -1:
            raise ValueError("Invalid reset_id")

        self._trajs.append(full_traj)
        self._reset_ids.append(reset_id)
        self._traj_lens.append(traj_len)
        self._trajs_added += 1
        assert len(self._trajs) == len(self._reset_ids)

        # x0 = leaf_index(full_traj.T_state_now, 0)
        x0 = full_traj.x0
        self._x0s.append(x0)

    def add_rollout(self, b_rollout: RolloutOutput):
        assert isinstance(b_rollout.T_rew, np.ndarray)
        b, T = b_rollout.T_rew.shape

        bT_isfinal = b_rollout.T_trunc | b_rollout.T_term
        trajs_added = 0

        bT_state_now = b_rollout.T_state_now

        for bb in range(b):
            T_isfinal = bT_isfinal[bb]
            idxs_done = np.where(T_isfinal)[0]

            idx_start = 0
            for idx_end in idxs_done:
                # idx_start should be the first timestep.
                index_start = (bb, idx_start)
                index_end = (bb, idx_end)
                trans_T = b_rollout.tree_index(index_end)

                if len(self._cur_trajs[bb]) == 0:
                    x0 = leaf_index(bT_state_now, index_start)
                    self._cur_x0s[bb].append(x0)
                    # x[0] is the step, x[1] is the source, etc.
                    assert x0[0] == 0

                    # Start and end of the trajectory. x0 is the start of the trajectory.
                    trans_0 = b_rollout.tree_index(index_start)
                    full_traj = RolloutOutput.tree_stack([trans_0, trans_T], axis=0)
                    # full_traj = RolloutOutput.tree_stack_lazy([trans_0, trans_T], axis=0)
                else:
                    assert len(self._cur_trajs[bb]) == 1
                    trans_0 = self._cur_trajs[bb][0]
                    full_traj = RolloutOutput.tree_stack([trans_0, trans_T], axis=0)
                    # full_traj = RolloutOutput.tree_stack_lazy([trans_0, trans_T], axis=0)

                traj_len = self._cur_traj_lens[bb] + (idx_end - idx_start) + 1
                self.add_full_traj(full_traj, traj_len)
                trajs_added += 1

                # Clear.
                self._cur_trajs[bb] = []
                self._cur_x0s[bb] = []
                self._cur_traj_lens[bb] = 0
                idx_start = idx_end + 1

            # Add the remaining segment to the current trajectory, but only if it contains the initial state.
            if idx_start < T:
                if len(self._cur_trajs[bb]) == 0:
                    index_start = (bb, idx_start)
                    trans_0 = b_rollout.tree_index(index_start)
                    self._cur_trajs[bb].append(trans_0)

                    x0 = leaf_index(bT_state_now, index_start)
                    self._cur_x0s[bb].append(x0)
                    # x[0] is the step, x[1] is the source, etc.
                    assert x0[0] == 0

                # Count the traj len so we have stats for the total traj len.
                self._cur_traj_lens[bb] += T - idx_start

        self._trajs_count.append(trajs_added)

    def get_stats(self, n_last: int):
        """Compute the mean total reward and trajectory length for the trajectories from the last n_last batches."""
        ntraj = np.sum(self._trajs_count[-n_last:])

        if ntraj == 0:
            return {}

        s_rew = []
        for traj in self._trajs[-ntraj:]:
            s_rew.append(traj.T_rew.sum())

        s_trajlen = []
        for trajlen in self._traj_lens[-ntraj:]:
            s_trajlen.append(trajlen)

        s_rew = np.array(s_rew)
        s_trajlen = np.array(s_trajlen)

        info = {
            "RewSum": s_rew.mean(),
            "TrajLen": s_trajlen.mean(),
            "TrajLenMin": s_trajlen.min(),
        }
        return info

    @property
    def trajs(self):
        return self._trajs

    @property
    def x0s(self):
        return self._x0s

    @property
    def cur_x0s(self):
        # Return the current x0s.
        b_x0s = []
        for k, v in self._cur_x0s.items():
            if len(v) > 0:
                assert len(v) == 1
                b_x0s.append(v[0])
        return b_x0s

    def all_x0s(self):
        return self._x0s + self.cur_x0s

    @property
    def reset_ids(self):
        return self._reset_ids

    def clear_trajs(self):
        self._x0s = []
        self._trajs = []
        self._reset_ids = []
        self._traj_lens = []
        self._trajs_count = []
