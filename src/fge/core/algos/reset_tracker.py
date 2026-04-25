from collections import defaultdict
from dataclasses import dataclass

import jax.tree_util as jtu
import numpy as np
from loguru import logger
from og.dyn_types import Obs

from fge.core.bits.collector import RolloutOutput
from fge.core.bits.obs_circbuf import ObsCircBuf
from fge.core.envs.jax_task import TreeLeaves, leaf_index, leaf_index_copy


@dataclass(slots=True)
class TrajInfo:
    ic: np.ndarray
    x0: TreeLeaves
    obs0: Obs
    reset_id: int
    is_safe: bool
    traj_len: int


class ResetTracker:
    """Tracks the resets.

    - A buffer of the most recent observations, for use with NSF.
    """

    def __init__(self, n_save_x0_obs: int, use_ic_obs: bool = False):
        self.n_save_x0_obs = n_save_x0_obs
        self.x0_obs_buf: ObsCircBuf | None = None

        self.use_ic_obs = use_ic_obs

        # List will only be empty or size 1.
        self._cur_x0s: dict[int, list[TreeLeaves]] = defaultdict(list)
        self._cur_obs0: dict[int, list[Obs]] = defaultdict(list)
        self._cur_ic0: dict[int, list[np.ndarray]] = defaultdict(list)

        self._cur_rewsum: dict[int, int] = defaultdict(int)
        self._cur_trajlen: dict[int, int] = defaultdict(int)

    def add_rollout(self, b_rollout: RolloutOutput) -> list[TrajInfo]:
        """Process the batch of rollouts.

        Returns a list of completed trajectories. The only information about the completed trajectory that we need is
        the x0, obs0, reset_id, and whether it was safe or not.
        """
        assert isinstance(b_rollout.T_rew, np.ndarray)

        if self.x0_obs_buf is None:
            if self.use_ic_obs:
                bT_ic = b_rollout.T_info["ic"]
                # [step, source, ic, ...]
                obs_ic_tmp = bT_ic[0, 0]
                self.x0_obs_buf = ObsCircBuf(self.n_save_x0_obs, obs_ic_tmp)
            else:
                # Initialize the buffer.
                obs_tmp = jtu.tree_map(lambda x: x[0, 0], b_rollout.T_obs_now)
                self.x0_obs_buf = ObsCircBuf(self.n_save_x0_obs, obs_tmp)
                del obs_tmp

        b, T = b_rollout.T_rew.shape
        bT_isfinal = b_rollout.T_trunc | b_rollout.T_term

        bT_state_now = b_rollout.T_state_now
        bT_obs_now = b_rollout.T_obs_now
        bT_info = b_rollout.T_info

        traj_infos = []

        for bb in range(b):
            T_isfinal = bT_isfinal[bb]
            idxs_done = np.where(T_isfinal)[0]

            # Traj ended.
            idx_start = 0
            for idx_end in idxs_done:
                index_start = (bb, idx_start)

                if len(self._cur_x0s[bb]) == 0:
                    # The entire trajectory is [idx_start:idx_end+1].

                    # Note: VERY important to copy here, otherwise the entire array will be kept in memory since
                    # both x0 and obs0 will reference the entire bT_state_now and bT_obs_now arrays.
                    x0 = leaf_index_copy(bT_state_now, index_start)
                    obs0 = jtu.tree_map(lambda x: x[bb, idx_start].copy(), bT_obs_now)
                    ic0 = bT_info["ic"][bb, idx_start]

                    reset_id = b_rollout.T_reset_id[bb, idx_start]
                    assert x0[0] == 0
                    rew_sum = b_rollout.T_rew[bb, idx_start : idx_end + 1].sum()
                    is_safe = rew_sum >= 0
                    traj_len = idx_end - idx_start + 1

                    traj_info = TrajInfo(ic0, x0, obs0, reset_id, is_safe, traj_len)
                    traj_infos.append(traj_info)

                    if self.use_ic_obs:
                        self.x0_obs_buf.push(ic0)
                    else:
                        self.x0_obs_buf.push(obs0)
                else:
                    assert len(self._cur_x0s[bb]) == len(self._cur_obs0[bb]) == 1
                    # The trajectory started from before, but ended at idx_end.

                    # The x0 and obs0 were already copied, so this doesn't reference the entire bT arrays.
                    x0 = self._cur_x0s[bb][0]
                    obs0 = self._cur_obs0[bb][0]
                    ic0 = self._cur_ic0[bb][0]

                    rew_sum = self._cur_rewsum[bb] + b_rollout.T_rew[bb, idx_start : idx_end + 1].sum()
                    is_safe = rew_sum >= 0
                    reset_id = b_rollout.T_reset_id[bb, idx_start]
                    traj_len = self._cur_trajlen[bb] + (idx_end - idx_start + 1)

                    traj_info = TrajInfo(ic0, x0, obs0, reset_id, is_safe, traj_len)
                    traj_infos.append(traj_info)

                    # TODO: Why didn't we have this one before?
                    if self.use_ic_obs:
                        self.x0_obs_buf.push(ic0)
                    else:
                        self.x0_obs_buf.push(obs0)

                # Clear.
                self._cur_x0s[bb] = []
                self._cur_obs0[bb] = []
                self._cur_ic0[bb] = []
                self._cur_rewsum[bb] = 0
                self._cur_trajlen[bb] = 0
                idx_start = idx_end + 1

            if idx_start < T:
                # Traj did not end yet.
                if len(self._cur_x0s[bb]) == 0:

                    # Note: VERY important to copy here, otherwise the entire array will be kept in memory since
                    # both x0 and obs0 will reference the entire bT_state_now and bT_obs_now arrays.
                    x0 = leaf_index_copy(bT_state_now, (bb, idx_start))
                    obs0 = jtu.tree_map(lambda x: x[bb, idx_start].copy(), bT_obs_now)
                    ic0 = bT_info["ic"][bb, idx_start]

                    self._cur_x0s[bb].append(x0)
                    self._cur_obs0[bb].append(obs0)
                    self._cur_ic0[bb].append(ic0)

                    if self.use_ic_obs:
                        self.x0_obs_buf.push(ic0)
                    else:
                        self.x0_obs_buf.push(obs0)

                self._cur_rewsum[bb] += b_rollout.T_rew[bb, idx_start:].sum()
                self._cur_trajlen[bb] += T - idx_start

        return traj_infos
