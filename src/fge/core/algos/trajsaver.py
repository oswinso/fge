from collections import defaultdict

import jax.numpy as jnp
import numpy as np
from og.jax_utils import jax2np
from og.tree_utils import tree_cat, tree_stack

from fge.core.bits.collector import RolloutOutput
from fge.core.bits.obs_circbuf import ObsCircBuf
from fge.core.envs.jax_task import TreeLeaves, leaf_index
from fge.core.utils.jax_util import wlabel


class TrajSaver:
    """Given the partial (batched) trajectories from rollouts, splice them together to get back full trajectories."""

    def __init__(
        self,
        save_full_traj: bool = True,
        n_save_x0_obs: int | None = None,
        obs_shape=None,
    ):
        """
        :param save_full_traj: If True, save the full trajectory. If False, only save the initial and final states.
        """
        self.save_full_traj = save_full_traj

        # Used in all_x0s...
        self._x0s: list[TreeLeaves] = []
        self._trajs: list[RolloutOutput] = []
        self._reset_ids: list[int] = []

        # List will only be empty or size 1.
        self._cur_x0s: dict[int, list[TreeLeaves]] = defaultdict(list)

        self._cur_trajs: dict[int, list[RolloutOutput]] = defaultdict(list)
        # Save a count of how many trajectories were added for each batch.
        self._trajs_count = []

        self._trajs_added = 0

        self.x0_obs_buf = None
        if n_save_x0_obs is not None:
            self.x0_obs_buf = ObsCircBuf(n_save_x0_obs, obs_shape)

    @property
    def x0s(self):
        return self._x0s

    def add_full_traj(self, full_traj: RolloutOutput):
        reset_id = full_traj.T_reset_id[0]
        assert np.all(reset_id == full_traj.T_reset_id)

        reset_id_real = reset_id
        if reset_id_real == -1:
            # Let's not have trajsaver keep track of initial conditions...
            raise ValueError("")

        self._trajs.append(full_traj)
        self._reset_ids.append(reset_id_real)
        self._trajs_added += 1
        assert len(self._trajs) == len(self._reset_ids)

        # logger.debug("Added traj {} with reset_id {}".format(self._trajs_added - 1, reset_id_real))
        # # len(trajs) == 1 has reset_id_real = 0.
        # assert reset_id_real + 1 <= self._trajs_added

        # Add the initial state.
        # x0 = jtu.tree_map(lambda x: x[0], full_traj.T_state_now)
        x0 = leaf_index(full_traj.T_state_now, 0)
        self._x0s.append(x0)

    @property
    def last_trajs(self) -> tuple[list[int], list[RolloutOutput], list[int]]:
        n_trajs_added_last = self._trajs_count[-1]

        n_trajs_total = len(self._trajs)
        # [ n_trajs_total - n_trajs_added_last, ..., n_trajs_total - 1 ]
        traj_ids = np.arange(n_trajs_added_last) + n_trajs_total - n_trajs_added_last

        return (
            traj_ids,
            self._trajs[-n_trajs_added_last:],
            self._reset_ids[-n_trajs_added_last:],
        )

    def add_rollout(self, b_rollout: RolloutOutput):
        if isinstance(b_rollout.T_rew, jnp.ndarray):
            # Put it on CPU.
            b_rollout = jax2np(b_rollout)

        b, T = b_rollout.T_rew.shape

        bT_isfinal = b_rollout.T_trunc | b_rollout.T_term

        trajs_added = 0

        def mytreeindex(rollout_: RolloutOutput, idx_):
            return rollout_.tree_index(idx_)

        bT_obs_now = b_rollout.T_obs_now
        bT_state_now = b_rollout.T_state_now

        tree_index1 = wlabel(mytreeindex, "x[bb]")
        tree_index2 = wlabel(mytreeindex, "x[idx_start:idx_end+1]")
        tree_index3 = wlabel(mytreeindex, "x[first_and_last]")
        tree_index4 = wlabel(mytreeindex, "x[idx_start:]")

        for bb in range(b):
            T_isfinal = bT_isfinal[bb]
            idxs_done = np.where(T_isfinal)[0]

            # rollout: RolloutOutput = tree_map1(lambda x: x[bb], b_rollout)
            # rollout: RolloutOutput = b_rollout.tree_index(bb)
            rollout: RolloutOutput = tree_index1(b_rollout, bb)

            idx_start = 0
            for idx_end in idxs_done:
                # Add the current segment to the current trajectory.
                sl = slice(idx_start, idx_end + 1)
                # rollout_segment: RolloutOutput = tree_map2(lambda x: x[idx_start : idx_end + 1], rollout)
                # rollout_segment: RolloutOutput = rollout.tree_index(sl)
                rollout_segment: RolloutOutput = tree_index2(rollout, sl)

                if len(self._cur_trajs[bb]) == 0:
                    assert len(self._cur_x0s[bb]) == 0

                    x0 = leaf_index(rollout_segment.T_state_now, 0)
                    self._cur_x0s[bb].append(x0)

                    obs0 = rollout_segment.T_obs_now[0]
                    if self.x0_obs_buf is not None:
                        # Store the initial observation.
                        self.x0_obs_buf.push(obs0)

                self._cur_trajs[bb].append(rollout_segment)

                if self.save_full_traj:
                    # Store the completed trajectory.
                    full_traj: RolloutOutput = tree_cat(self._cur_trajs[bb], axis=0, which=np)
                # t0_3 = time.time()
                else:
                    if len(self._cur_trajs[bb]) == 1:
                        # Only one segment, take the first and last state.
                        full_traj = self._cur_trajs[bb][0]
                        first_and_last = np.array([0, -1])
                        # full_traj = tree_map3(lambda x: x[first_and_last], full_traj)
                        # full_traj = full_traj.tree_index(first_and_last)
                        full_traj = tree_index3(full_traj, first_and_last)
                    else:
                        # First step of first segment, last step of last segment.
                        # first_step = tree_index(0, self._cur_trajs[bb][0])
                        first_step = self._cur_trajs[bb][0].tree_index(0)
                        # last_step = tree_index(-1, self._cur_trajs[bb][-1])
                        last_step = self._cur_trajs[bb][-1].tree_index(-1)
                        full_traj = tree_stack([first_step, last_step], axis=0, which=np)

                self.add_full_traj(full_traj)
                # t0_4 = time.time()
                trajs_added += 1

                # Clear.
                self._cur_trajs[bb] = []
                self._cur_x0s[bb] = []
                idx_start = idx_end + 1

            # Add the remaining segment to the current trajectory.
            if idx_start < T:
                if len(self._cur_trajs[bb]) == 0:
                    assert len(self._cur_x0s[bb]) == 0

                    # Initial segment
                    obs0 = bT_obs_now[bb, idx_start]
                    if self.x0_obs_buf is not None:
                        # Store the initial observation.
                        self.x0_obs_buf.push(obs0)

                    idx_0 = (bb, idx_start)
                    x0 = leaf_index(bT_state_now, idx_0)
                    self._cur_x0s[bb].append(x0)

                # rollout_segment = tree_map4(lambda x: x[idx_start:], rollout)
                sl = slice(idx_start, None)
                # rollout_segment = rollout.tree_index(sl)
                rollout_segment = tree_index4(rollout, sl)
                self._cur_trajs[bb].append(rollout_segment)

        self._trajs_count.append(trajs_added)

    def get_stats(self, n_last: int):
        """Compute the mean total reward and trajectory length for the trajectories from the last n_last batches."""
        ntraj = np.sum(self._trajs_count[-n_last:])

        if ntraj == 0:
            return {}

        s_rew = []
        s_trajlen = []
        for traj in self._trajs[-ntraj:]:
            s_rew.append(traj.T_rew.sum())
            s_trajlen.append(traj.T_rew.shape[0])

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
