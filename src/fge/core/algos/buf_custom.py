import functools as ft
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
import numpy as np
from attrs import asdict, define
from loguru import logger
from og.dyn_types import BObs
from og.jax_utils import jax2np
from og.rng import PRNGKey
from og.tree_utils import tree_cat, tree_index, tree_stack, tree_where_dim0

from fge.core.algos.reset_tracker import ResetTracker
from fge.core.bits.ci_buf import CIBuf
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.labeled_obs_circbuf import LabeledObsCircBuf
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.prio_q import PrioQ
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import LeafObs, Source, StateResetId
from fge.core.common.reset_buf import ResetBuf
from fge.core.envs.jax_task import JaxTask, MinState_, TreeLeaves, leaf_index
from fge.core.utils.jax_util import myjit

State_ = TypeVar("State_")


class OPT:
    add_every: int = 0
    add_safe_minsampletimes: int = 1


type ValueOPT = Literal["RND", "NSF", "ValueMiddle"]

type TrainNSFOPT = Literal["bT_obs_now", "b_x0"]
type RehearsalOPT = Literal["Vl", "polsafe"]

BP_HERE = 0


def pytree_nbytes(tree) -> int:
    """Return total number of bytes stored in all arrays in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0
    for leaf in leaves:
        # Handle JAX arrays and DeviceArrays
        if isinstance(leaf, jnp.ndarray):
            raise ValueError("Shouldn't have Jax arrays...")

        # Optionally: handle numpy arrays too
        if isinstance(leaf, np.ndarray):
            total += leaf.size * leaf.dtype.itemsize

    return total


@define
class BufCustomCfg:
    resetbuf: ResetBuf.Cfg = ResetBuf.Cfg(
        capacity_ci=100_000,
        capacity_explore=512,
        capacity_predci=512,
        p_explore=0.4,
        p_base=0.1,
    )
    """Capacity of the LIFO x0 buffer."""

    n_resetbuf_init: int = 4
    """How many samples to initialize the resetbuf with at the beginning from the initial state distribution."""

    n_sample_colbuf_max: int = 128
    n_sample_prioq_max: int = 64

    n_sample_explore: int = 256
    """Number of samples to sample for argmax explore."""

    n_sample_ci_robust: int = 256
    """Number of samples to sample for being robust to CI."""

    prioq_capacity: int = 1024
    """Number of samples to keep track of in the priority queue."""

    n_sample_onpolbuf: int = 2048
    """Number of samples to sample for argmax explore."""

    # mode: OPT = OPT.add_every
    mode: OPT = OPT.add_safe_minsampletimes
    """Which mode for the argmax."""

    value_opt: ValueOPT = "NSF"

    minsampletimes: int = 4
    """Minimum number of times to sample from the argmax state before considering a new one."""

    train_nsf_mode: TrainNSFOPT = "bT_obs_now"
    rehearsal_mode: RehearsalOPT = "polsafe"

    use_nsf_explore: bool = True
    """If True, use NSF to argmin."""

    n_x0_nsf: int = 8192

    polcond_buf_capacity: int = 16_384

    ci_buf_capacity: int = 32_768

    pred_ci_thresh: float = 0.4

    rehearsal_only_frac: float | None = None
    """If not None, then this is the fraction of iterations at the end to only sample from the rehearsal buffer."""

    rnd_apply: list[Source] = [Source.BUF_EXPLORE, Source.BASE]
    """Where to apply RND to."""

    ci_clsfy_fracci: float | None = None
    """If None, then do the (incorrect) thing we were doing before where frac_ci ranges from [0, 0.5].
    If true, then sample with replacement.
    """

    n_sample_nsf_ci: int = 64
    """Number of samples (with replacement) to sample from the CI buffer for the NSF CI classifier."""

    use_nsf_ci: bool = False
    """If true, use NSF for the CI classification instead of the CI classifier.
    We do this by using an EMA of some percentile of the logprob evaluated over the CI buffer.
    """

    nsf_ci_beta: float = 0.2
    """Decay factor for the EMA of the logprob threshold for the NSF CI classifier."""

    n_nsf_ci_sample_thresh: int = 1024
    """How many samples to use to compute the logprob threshold for the NSF CI classifier."""

    nsf_ci_quantile: float = 0.05
    """The quantile to use for computing the logprob threshold for deciding if a sample is in the CI using NSF."""

    def asdict(self):
        return asdict(self)


class BufCustom:
    """Buffer for our custom method."""

    Cfg = BufCustomCfg

    def __init__(
            self,
            key: PRNGKey,
            task: JaxTask,
            cfg: BufCustomCfg,
            reset_id_provider: ResetIDProvider,
    ):
        self.task = task

        self.cfg = cfg

        self.in_ci: dict[int, bool] = {}
        """True if we know this initial condition is in the CI. Index is reset_id."""

        self.collide_x0tup: dict[int, LeafObs] = {}
        """Dictionary of reset_id -> (x0, obs) that are not in the CI. The dictionary is used for fast delete."""

        self.safe_traj_id: dict[int, int] = {}
        """Index into trajsaver of the most recent safest trajectory. Index is reset_id."""

        self.reset_count: dict[int, int] = {}

        self.key_sample0, self.key_sample1, key, key_seed = jr.split(key, 4)

        state = task.reset(jr.PRNGKey(0))
        obs = task.get_dummy_obs()
        ic = np.zeros(task.x0_unif_shape)
        statetup = StateResetId(state, np.array(-1, dtype=np.int32))
        buffer = ResetBuf.create(statetup, task, cfg.resetbuf, reset_id_provider)

        gpu_device = jax.devices("gpu")[0]
        self._buffer: ResetBuf = jax.device_put(buffer, gpu_device)

        b = self._buffer
        logger.critical(
            "buffer p_explore: {:.3f} | p_base: {:.3f} | p_predci: {:.3f} | p_ci: {:.3f}".format(
                b.p_explore, b.p_base, b.p_predci, 1 - (b.p_explore + b.p_base + b.p_predci)
            )
        )

        leaves = self.task.minify(state)
        self.ci_buf = CIBuf(cfg.ci_buf_capacity, leaves, obs)

        seed = jr.randint(key_seed, tuple(), minval=0, maxval=np.iinfo(np.int32).max)
        self.rng = np.random.default_rng(seed=int(seed))
        self.sample_iters = 0

        self.prioq = PrioQ(cfg.prioq_capacity)

        # These are used with mode == add_safe_minsampletimes
        self.waiting_for_end = set()
        """Set of reset_ids that were argmax'ed but haven't ended yet, so we don't know if they are in the CI."""
        self.added_argmax = set()
        """Set of reset_ids that were argmax'ed and added to the buffer, to prevent adding it multiple times."""

        # self.argmaxes: list[StateResetId] = []
        # self.argmaxes_ic: list[Any] = []

        self.argmax_ci_ic: np.ndarray = np.array([])
        self.argmax_explore_ic: np.ndarray = np.array([])

        self.polcond_obs0 = LabeledObsCircBuf(cfg.polcond_buf_capacity, ic, obs)

        # self.trajsaver = TrajSaver(save_full_traj=False, n_save_x0_obs=cfg.n_x0_nsf, obs_shape=obs.shape)
        use_ic_obs = task.use_ic_obs
        self.reset_tracker = ResetTracker(n_save_x0_obs=cfg.n_x0_nsf, use_ic_obs=use_ic_obs)

        self.logprob_thresh = 0
        """EMA estimate of the quantile"""

        self.n_logprob_thresh = 0
        """Counter used to debias the logprob_thresh EMA estimate."""

    @property
    def logprob_thresh_unbiased(self) -> float:
        n_logprob_thresh = max(1, self.n_logprob_thresh)
        return self.logprob_thresh / (1 - self.cfg.nsf_ci_beta ** n_logprob_thresh)

    @property
    def n_ci_buf(self) -> int:
        return self.ci_buf.size

    def add_rollout(self, b_rollout: RolloutOutput):
        """Called after rollout."""
        completed_trajs = self.reset_tracker.add_rollout(b_rollout)

        for traj in completed_trajs:
            assert isinstance(traj.reset_id, np.int32)

            ic = traj.ic
            x0 = traj.x0
            obs0 = traj.obs0

            # Add it to the dictionary if it is not in the CI yet.
            if traj.reset_id not in self.in_ci:
                self.in_ci[traj.reset_id] = False
                # self.safe_traj_id[reset_id] = traj_id
                self.reset_count[traj.reset_id] = 0

                self.collide_x0tup[traj.reset_id] = (x0, obs0)

            # Increment reset count.
            self.reset_count[traj.reset_id] += 1

            if traj.is_safe:
                if self.in_ci[traj.reset_id] != True:
                    self.ci_buf.push(x0, obs0, traj.reset_id)

                    # Remove it from ci_obs.
                    if traj.reset_id in self.collide_x0tup:
                        del self.collide_x0tup[traj.reset_id]

                self.in_ci[traj.reset_id] = True
                # self.safe_traj_id[reset_id] = traj_id

            # Add the pair (obs0, traj_is_safe) to the circular buffer.
            self.polcond_obs0.push(ic, obs0, traj.is_safe)

        del completed_trajs

        # If collide_x0tup is too big, then remove the oldest few elements.
        collide_x0tup_maxsize = 20_000
        if len(self.collide_x0tup) > collide_x0tup_maxsize:
            # Remove the oldest elements.
            keys = list(self.collide_x0tup.keys())
            keys = keys[: len(keys) - collide_x0tup_maxsize]
            for key in keys:
                del self.collide_x0tup[key]

    @ft.partial(jax.jit, static_argnums=0)
    def sample_from_env(self, sample_iters: int):
        key = jr.fold_in(self.key_sample0, sample_iters)
        b_key = jr.split(key, self.cfg.n_sample_explore)
        b_x0 = jax.vmap(self.task.reset)(b_key)
        return b_x0

    @ft.partial(jax.jit, static_argnums=0)
    def approx_rejection_sample_from_env(self, classifier: CIClassifier, sample_iters: int):
        key = jr.fold_in(self.key_sample1, sample_iters)
        b_key = jr.split(key, self.cfg.n_sample_onpolbuf)
        b_x0 = jax.vmap(self.task.reset)(b_key)
        b_obs = jax.vmap(self.task.get_obs)(b_x0)
        b_prob_ci = jax.vmap(classifier.get_probs)(b_obs)
        b_pred_inci = b_prob_ci >= self.cfg.pred_ci_thresh

        # Try to get capacity_predci samples that are in the CI by argsorting.
        # Put all the Trues at the front.
        s_idx_inci = jnp.argsort(b_pred_inci, descending=True)[: self.cfg.resetbuf.capacity_predci]

        # Not all of s_x0 are in CI, so after returning this we need to replace ones that aren't.
        s_pred_inci = b_pred_inci[s_idx_inci]
        s_x0 = tree_index(s_idx_inci, b_x0)

        n_inci = np.sum(s_pred_inci)
        n_to_add = self.cfg.resetbuf.capacity_predci - n_inci

        return s_x0, s_pred_inci, n_to_add

    @ft.partial(jax.jit, static_argnums=0)
    def sample_from_env_min(self, sample_iters: int):
        b_x0 = self.sample_from_env(sample_iters)
        b_x0_min = jax.vmap(self.task.get_minstate)(b_x0)
        return b_x0_min

    @ft.partial(jax.jit, static_argnums=0)
    def from_minstate(self, minstate):
        return self.task.from_minstate(minstate)

    @ft.partial(jax.jit, static_argnums=0)
    def get_Vl_nsf(self, ppo: IntrinsicSumPPO, nsf: NSF, b_obs: BObs):
        b_logprob = jax.vmap(nsf.log_prob)(b_obs)
        b_Vl = jax.vmap(ppo.Vl_ext.apply)(b_obs).squeeze(-1)
        assert b_logprob.shape == b_Vl.shape

        return b_Vl, b_logprob

    @ft.partial(jax.jit, static_argnums=0)
    def get_nsf(self, nsf: NSF, b_obs: BObs):
        b_logprob = jax.vmap(nsf.log_prob)(b_obs)
        return b_logprob

    @ft.partial(jax.jit, static_argnums=0)
    def get_rehearsal_metrics(self, ppo: IntrinsicSumPPO, polsafe: CIClassifier, b_x_min: MinState_):
        b_obs = self.get_b_obs(b_x_min)
        b_Vl = jax.vmap(ppo.Vl_ext.apply)(b_obs).squeeze(-1)
        b_psafe_pol = jax.vmap(polsafe.get_probs)(b_obs)
        return b_Vl, b_psafe_pol

    @ft.partial(jax.jit, static_argnums=0)
    def get_b_psafe_ci(self, polsafe: CIClassifier, b_x_min: MinState_):
        b_obs = self.get_b_obs(b_x_min)
        b_psafe_pol = jax.vmap(polsafe.get_probs)(b_obs)
        return b_psafe_pol

    @ft.partial(jax.jit, static_argnums=0)
    def get_rnd(self, rnd: RND, b_obs: BObs):
        b_rnd = jax.vmap(rnd.get_bonus)(b_obs)
        return b_rnd

    def update_prioq(self, b_reset_idx: np.ndarray, b_x: TreeLeaves, b_prio: np.ndarray):
        # All this should happen on CPU and should be fast.
        with jax.transfer_guard("disallow"):
            num = len(b_prio)
            # Convert b_x to a list of states.
            b_x: list[TreeLeaves] = [leaf_index(b_x, ii) for ii in range(num)]
            self.prioq.add_or_update(b_reset_idx, b_prio, b_x)

    @ft.partial(myjit, static_argnums=0)
    def _predci_jit_things(
            self,
            reset_id_provider: ResetIDProvider,
            b_pred_inci: jnp.ndarray,
            b_x0,
            s_x_ci_sampled_min,
            s_reset_ids,
    ):
        """If b_pred_inci is True, then use b_x0 (rejection sample), otherwise use s_x_ci_sampled (CI buffer)."""
        reset_id_provider, b_reset_ids = reset_id_provider.get_masked_ids(b_pred_inci)

        s_x_ci_sampled = jax.vmap(self.task.from_minstate)(s_x_ci_sampled_min)
        b_x0 = tree_where_dim0(b_pred_inci, b_x0, s_x_ci_sampled)
        b_reset_ids = jnp.where(b_pred_inci, b_reset_ids, s_reset_ids)

        return reset_id_provider, b_x0, b_reset_ids

    def sample_and_update_predci(
            self,
            classifier: CIClassifier,
            b_x_ci_leaf: TreeLeaves,
            b_reset_id_ci: np.ndarray,
    ):
        # CI predictor only valid if we have something in CI.
        assert len(self.in_ci) > 0

        # Approximately rejection sample.
        b_x0, b_pred_inci, n_to_add = self.approx_rejection_sample_from_env(classifier, self.sample_iters)

        # The n_to_add samples at the end are not predicted CI. Replace them with the CI buffer.
        capacity = self.cfg.resetbuf.capacity_predci
        s_idx = self.rng.choice(self.n_ci_buf, capacity, replace=True)
        s_x_ci_sampled_leaf = leaf_index(b_x_ci_leaf, s_idx)
        s_reset_ids = b_reset_id_ci[s_idx]
        s_x_ci_sampled_min = self.task.leaf_to_minstate(s_x_ci_sampled_leaf)

        self.reset_id_provider, b_x0, b_reset_ids = self._predci_jit_things(
            self.reset_id_provider, b_pred_inci, b_x0, s_x_ci_sampled_min, s_reset_ids
        )

        # Set the buffer.
        b_ic = self.task.to_icval(b_x0)
        b_statetup = StateResetId(b_x0, b_reset_ids)
        self.update_predci(b_statetup, b_ic)

    def sample_states(self) -> tuple[MinState_, np.ndarray]:
        if len(self.in_ci) == 0:
            n_colbuf = 0
        else:
            n_colbuf = len(self.collide_x0tup)

        n_sample_colbuf = min(n_colbuf, self.cfg.n_sample_colbuf_max)
        n_sample_prioq = min(self.prioq.size, self.cfg.n_sample_prioq_max)
        n_sample_env = self.cfg.n_sample_explore - n_sample_colbuf - n_sample_prioq

        # 1: Sample N states from environment.
        b_x_sample: MinState_ = self.sample_from_env_min(self.sample_iters)
        jax.copy_to_host_async(b_x_sample)

        b_xs = []
        b_reset_idxs = []

        # 2: Sample N states from collision buffer
        if n_sample_colbuf > 0:
            b_reset_idxs_colbuf = list(self.collide_x0tup.keys())
            b_reset_idxs_colbuf = self.rng.choice(b_reset_idxs_colbuf, n_sample_colbuf, replace=False)
            b_x_colbuf_leaf_list = [self.collide_x0tup[ii][0] for ii in b_reset_idxs_colbuf]
            b_x_colbuf_leaf = tree_stack(b_x_colbuf_leaf_list, axis=0, which=np)
            b_x_colbuf: MinState_ = self.task.leaf_to_minstate(b_x_colbuf_leaf)

            b_xs.append(b_x_colbuf)
            b_reset_idxs.append(b_reset_idxs_colbuf)

        # 3: Sample N states from the (approx) priority queue
        if n_sample_prioq > 0:
            b_reset_idx_prioq, _, b_x_leaf_prioq = self.prioq.get_top_k(n_sample_prioq)
            b_x_leaf_prioq = tree_stack(b_x_leaf_prioq, axis=0, which=np)
            b_x_prioq = self.task.leaf_to_minstate(b_x_leaf_prioq)

            b_xs.append(b_x_prioq)
            b_reset_idxs.append(b_reset_idx_prioq)

        # 4: To hide latency, append b_x_sample to the list at the end.
        # b_x_sample = jax.device_get(b_x_sample)
        b_x_sample = jax2np(b_x_sample)
        b_x_sample = jtu.tree_map(lambda x: x[:n_sample_env], b_x_sample)
        b_reset_idx = np.full(n_sample_env, -1, dtype=np.int32)

        b_xs.append(b_x_sample)
        b_reset_idxs.append(b_reset_idx)

        # Concatenate all.
        b_xs = tree_cat(b_xs, axis=0, which=np)
        b_reset_idx = np.concatenate(b_reset_idxs, axis=0)

        self.sample_iters += 1
        return b_xs, b_reset_idx

    @ft.partial(jax.jit, static_argnums=0)
    def get_b_obs(self, b_x_min: MinState_) -> BObs:
        b_x = jax.vmap(self.task.from_minstate)(b_x_min)
        b_obs = jax.vmap(self.task.get_obs)(b_x)
        return b_obs

    @ft.partial(jax.jit, static_argnums=0)
    def get_b_ic_obs(self, b_x_min: MinState_) -> BObs:
        b_x = jax.vmap(self.task.from_minstate)(b_x_min)
        b_ic_obs = jax.vmap(self.task.to_icval)(b_x)
        return b_ic_obs

    @ft.partial(jax.jit, static_argnums=0)
    def get_b_is_unsafe_env(self, b_x_min: MinState_) -> BObs:
        b_x = jax.vmap(self.task.from_minstate)(b_x_min)
        b_is_unsafe = jax.vmap(self.task.is_unsafe_custom)(b_x)
        return b_is_unsafe

    @ft.partial(jax.jit, static_argnums=0)
    def predict_ci(self, classifier: CIClassifier, b_obs: BObs):
        b_prob = jax.vmap(classifier.get_probs)(b_obs)
        return b_prob

    @ft.partial(jax.jit, static_argnums=0)
    def predict_ci_nsf(self, nsf_ci: NSF, b_obs: BObs):
        b_logprob = jax.vmap(nsf_ci.log_prob)(b_obs)
        return b_logprob

    def update_resets_lean(self, nsf: NSF | None, classifier: CIClassifier, polsafe: CIClassifier):
        """Lean version of update_resets."""
        gpu_device = jax.devices("gpu")[0]

        assert self.cfg.rehearsal_mode == "polsafe"

        if len(self.in_ci) == 0:
            frac_in_ci = 0.0
        else:
            b_inci = np.array(list(self.in_ci.values()))
            frac_in_ci = np.mean(b_inci)
        info_custom_log = {
            "BufCustom/frac_in_ci": frac_in_ci,
            "BufCustom/n_ci_buf": self.n_ci_buf,
            "BufCustom/n_collide": len(self.collide_x0tup),
        }

        # ------ Explore -----
        # Sample a large batch of states from the environment, evaluate using the classifier.
        b_x_min, b_reset_idx = self.sample_states()
        b_x_leaf = self.task.minstate_to_leaf(b_x_min)
        b_x_min_gpu = jax.device_put(b_x_min, gpu_device)
        b_obs = self.get_b_obs(b_x_min_gpu)

        b_prob_ci = np.array(self.predict_ci(classifier, b_obs))
        b_pred_inci = b_prob_ci >= self.cfg.pred_ci_thresh

        # Exclude initial states that are already unsafe according to the env.
        b_is_unsafe_env = np.array(self.get_b_is_unsafe_env(b_x_min_gpu))

        if self.cfg.use_nsf_explore:
            assert nsf is not None
            if self.task.use_ic_obs:
                b_ic_obs = self.get_b_ic_obs(b_x_min_gpu)
                b_logprob = np.array(self.get_nsf(nsf, b_ic_obs))
            else:
                b_logprob = np.array(self.get_nsf(nsf, b_obs))

            # Exclude points predicted to be in CI
            b_cost = np.where(b_pred_inci, 133742069.0, b_logprob)

            # Filter out initial states that are unsafe according to the env.
            b_cost = np.where(b_is_unsafe_env, 1337420.0, b_cost)

            ii_argmin = np.argmin(b_cost)
        else:
            # If it's not in the CI, find the min logprob from NSF.
            b_cost = np.where(b_pred_inci, 133742069.0, 0.0)

            # Adding a tiny epsilon then argmin should be equivalent to uniform sampling.
            b_tmp = b_cost + self.rng.uniform(-0.1, 0.1, size=b_cost.shape)

            # Filter out initial states that are unsafe according to the env.
            b_tmp = np.where(b_is_unsafe_env, 1337420.0, b_tmp)

            ii_argmin = np.argmin(b_tmp)

        # Get the argmin state.
        x_opt_leaf = tree_index(ii_argmin, b_x_leaf)
        x_opt_min = self.task.leaf_to_minstate(x_opt_leaf)
        x_opt_min_gpu = jax.device_put(x_opt_min, gpu_device)
        x_opt = self.from_minstate(x_opt_min_gpu)
        reset_idx_opt = b_reset_idx[ii_argmin]

        if reset_idx_opt:
            # if reset_idx_opt == -1, them assign it a new one.
            self.reset_id_provider, reset_idx_opt = self.reset_id_provider.get_id()
            reset_idx_opt = int(reset_idx_opt)

            b_reset_idx[ii_argmin] = reset_idx_opt

        statetup_opt = StateResetId(x_opt, reset_idx_opt)
        ic = self.task.to_icval(x_opt_min)
        self.add_explore_argmax(statetup_opt, ic)

        # Update the prioq.
        # Remove all points with reset_idx = -1.
        b_shouldaddprio = b_reset_idx >= 0
        self.update_prioq(
            b_reset_idx[b_shouldaddprio],
            leaf_index(b_x_leaf, b_shouldaddprio),
            b_cost[b_shouldaddprio],
        )

        info = {"Explore/ic": self.task.to_icval(b_x_min)}

        # ------ Robust over CI -----
        if self.n_ci_buf > 0:
            b_x_ci_leaf = self.get_b_x0_ci()
            b_reset_id_ci = self.get_b_reset_id()

            n_sample_ci_robust = self.cfg.n_sample_ci_robust

            if self.n_ci_buf < n_sample_ci_robust:
                # Sample a batch of states from the CI.
                n_extra = n_sample_ci_robust - self.n_ci_buf
                b_idx = np.concatenate([np.arange(self.n_ci_buf), np.zeros(n_extra, dtype=np.int32)])
            else:
                b_idx = self.rng.choice(self.n_ci_buf, n_sample_ci_robust, replace=False)

            b_x_ci_sampled_leaf = leaf_index(b_x_ci_leaf, b_idx)
            b_x_ci_sampled_min = self.task.leaf_to_minstate(b_x_ci_sampled_leaf)

            b_ic_ci_sampled = self.task.to_icval(b_x_ci_sampled_min)

            # Find the worst state using polsafe.
            b_psafe_ci = self.get_b_psafe_ci(polsafe, jax.device_put(b_x_ci_sampled_min, gpu_device))
            b_psafe_ci = np.array(b_psafe_ci)

            if self.n_ci_buf < n_sample_ci_robust:
                b_ic_ci_sampled = b_ic_ci_sampled[: self.n_ci_buf]
                b_psafe_ci = b_psafe_ci[: self.n_ci_buf]

            ii_ci_argmax = np.argmin(b_psafe_ci)

            info["CI/ic_argmax"] = ii_ci_argmax
            info["CI/ic"] = b_ic_ci_sampled
            info["CI/polsafe"] = b_psafe_ci

            # Get the opt state.
            x_opt_min_ci = tree_index(ii_ci_argmax, b_x_ci_sampled_min)
            x_opt_ci = self.from_minstate(jax.device_put(x_opt_min_ci, gpu_device))
            reset_idx_opt_ci = b_reset_id_ci[ii_ci_argmax]

            statetup_opt = StateResetId(x_opt_ci, reset_idx_opt_ci)
            ic = self.task.to_icval(x_opt_min_ci)
            self.add_ci_argmax(statetup_opt, ic)

            self.sample_and_update_predci(classifier, b_x_ci_leaf, b_reset_id_ci)

        p_explore, p_base = float(self.buffer.p_explore), float(self.buffer.p_base)
        p_predci = float(self.buffer.p_predci)
        p_rehearse = 1 - (p_explore + p_base + p_predci)
        info_custom_log = {
                              "BufCustom/p_explore": p_explore,
                              "BufCustom/p_base": p_base,
                              "BufCustom/p_predci": p_predci,
                              "BufCustom/p_rehearse": p_rehearse,
                          } | info_custom_log

        return info, info_custom_log

    def update_resets(
            self,
            ppo: SumPPO,
            rnd: RND,
            nsf: NSF,
            classifier: CIClassifier,
            polsafe: CIClassifier,
            nsf_ci: NSF | None,
    ):
        """ """
        gpu_device = jax.devices("gpu")[0]

        if len(self.in_ci) == 0:
            frac_in_ci = 0.0
        else:
            b_inci = np.array(list(self.in_ci.values()))
            frac_in_ci = np.mean(b_inci)
        info_custom_log = {
            "BufCustom/frac_in_ci": frac_in_ci,
            "BufCustom/n_ci_buf": self.n_ci_buf,
            "BufCustom/n_collide": len(self.collide_x0tup),
        }

        # ------ Explore -----
        # Sample a large batch of states from the environment, evaluate using the classifier.
        b_x_min, b_reset_idx = self.sample_states()
        b_x_leaf = self.task.minstate_to_leaf(b_x_min)
        b_x_min_gpu = jax.device_put(b_x_min, gpu_device)
        b_obs = self.get_b_obs(b_x_min_gpu)

        # Predict whether its in CI using the classifier
        if self.cfg.use_nsf_ci:
            if self.n_ci_buf <= 4:
                # No CI, so everything is not CI.
                b_pred_inci = np.zeros(b_obs.shape[0], dtype=bool)
            else:
                # Sample a batch of states from the CI buffer.
                b_obs_ci = self.get_b_obs_ci()
                b_idx = self.rng.choice(self.n_ci_buf, self.cfg.n_sample_nsf_ci, replace=True)
                b_obs_ci_sampled = b_obs_ci[b_idx]

                # Get the logprob
                b_logprob_ci_test = np.array(self.predict_ci_nsf(nsf_ci, b_obs_ci_sampled))
                quantile = np.quantile(b_logprob_ci_test, self.cfg.nsf_ci_quantile)

                # Update logprob_thresh using EMA.
                beta = self.cfg.nsf_ci_beta
                self.logprob_thresh = beta * self.logprob_thresh + (1 - beta) * quantile
                self.n_logprob_thresh += 1

                b_logprob_ci = np.array(self.predict_ci_nsf(nsf_ci, b_obs))
                b_pred_inci = b_logprob_ci >= self.logprob_thresh_unbiased
        else:
            b_prob_ci = np.array(self.predict_ci(classifier, b_obs))
            b_pred_inci = b_prob_ci >= self.cfg.pred_ci_thresh

        # If it's not in the CI, find the min logprob from NSF.
        b_Vl, b_logprob = jax2np(self.get_Vl_nsf(ppo, nsf, b_obs))
        b_rnd = jax2np(self.get_rnd(rnd, b_obs))

        if self.cfg.use_nsf_explore:
            # Exclude points in the CI buffer.
            b_cost = np.where(b_pred_inci, 133742069.0, b_logprob)

            ii_argmin = np.argmin(b_cost)
        else:
            b_cost = np.where(b_pred_inci, 133742069.0, 0.0)

            # Adding a tiny epsilon then argmin should be equivalent to uniform sampling.
            b_tmp = b_cost + self.rng.uniform(-0.1, 0.1, size=b_cost.shape)
            ii_argmin = np.argmin(b_tmp)

        # Get the argmin state.
        x_opt_leaf = tree_index(ii_argmin, b_x_leaf)
        x_opt_min = self.task.leaf_to_minstate(x_opt_leaf)
        x_opt_min_gpu = jax.device_put(x_opt_min, gpu_device)
        x_opt = self.from_minstate(x_opt_min_gpu)
        reset_idx_opt = b_reset_idx[ii_argmin]

        if reset_idx_opt:
            # if reset_idx_opt == -1, them assign it a new one.
            self.reset_id_provider, reset_idx_opt = self.reset_id_provider.get_id()
            reset_idx_opt = int(reset_idx_opt)

            b_reset_idx[ii_argmin] = reset_idx_opt

        statetup_opt = StateResetId(x_opt, reset_idx_opt)
        ic = self.task.to_icval(x_opt_min)
        self.add_explore_argmax(statetup_opt, ic)

        # Update the prioq.
        # Remove all points with reset_idx = -1.
        b_shouldaddprio = b_reset_idx >= 0
        self.update_prioq(
            b_reset_idx[b_shouldaddprio],
            leaf_index(b_x_leaf, b_shouldaddprio),
            b_cost[b_shouldaddprio],
        )

        info = {
            "Explore/ic": self.task.to_icval(b_x_min),
            "Explore/rnd": b_rnd,
            "Explore/logprob": b_logprob,
        }

        # ------ Robust over CI -----
        if self.n_ci_buf > 0:
            b_x_ci_leaf = self.get_b_x0_ci()
            b_reset_id_ci = self.get_b_reset_id()

            n_sample_ci_robust = self.cfg.n_sample_ci_robust

            if self.n_ci_buf < n_sample_ci_robust:
                # Sample a batch of states from the CI.
                n_extra = n_sample_ci_robust - self.n_ci_buf
                b_idx = np.concatenate([np.arange(self.n_ci_buf), np.zeros(n_extra, dtype=np.int32)])
            else:
                b_idx = self.rng.choice(self.n_ci_buf, n_sample_ci_robust, replace=False)

            b_x_ci_sampled_leaf = leaf_index(b_x_ci_leaf, b_idx)
            b_x_ci_sampled_min = self.task.leaf_to_minstate(b_x_ci_sampled_leaf)

            b_ic_ci_sampled = self.task.to_icval(b_x_ci_sampled_min)

            # Find the worst state using the value function.
            out = self.get_rehearsal_metrics(ppo, polsafe, jax.device_put(b_x_ci_sampled_min, gpu_device))
            # out = jax.device_get(out)
            b_Vl_ci, b_psafe_ci = [np.array(a) for a in out]

            if self.n_ci_buf < n_sample_ci_robust:
                b_ic_ci_sampled = b_ic_ci_sampled[: self.n_ci_buf]
                b_Vl_ci = b_Vl_ci[: self.n_ci_buf]
                b_psafe_ci = b_psafe_ci[: self.n_ci_buf]

            match self.cfg.rehearsal_mode:
                case "Vl":
                    ii_ci_argmax = np.argmax(b_Vl_ci)
                case "polsafe":
                    ii_ci_argmax = np.argmin(b_psafe_ci)

            info["CI/ic_argmax"] = ii_ci_argmax
            info["CI/ic"] = b_ic_ci_sampled
            info["CI/Vl"] = b_Vl_ci
            info["CI/polsafe"] = b_psafe_ci

            # Get the opt state.
            x_opt_min_ci = tree_index(ii_ci_argmax, b_x_ci_sampled_min)
            x_opt_ci = self.from_minstate(jax.device_put(x_opt_min_ci, gpu_device))
            reset_idx_opt_ci = b_reset_id_ci[ii_ci_argmax]

            statetup_opt = StateResetId(x_opt_ci, reset_idx_opt_ci)
            ic = self.task.to_icval(x_opt_min_ci)
            self.add_ci_argmax(statetup_opt, ic)

            # Rejection sampling from the predicted CI distribution.
            self.sample_and_update_predci(classifier, b_x_ci_leaf, b_reset_id_ci)

        p_explore, p_base = float(self.buffer.p_explore), float(self.buffer.p_base)
        p_predci = float(self.buffer.p_predci)
        p_rehearse = 1 - (p_explore + p_base + p_predci)
        info_custom_log = {
                              "BufCustom/p_explore": p_explore,
                              "BufCustom/p_base": p_base,
                              "BufCustom/p_predci": p_predci,
                              "BufCustom/p_rehearse": p_rehearse,
                          } | info_custom_log

        return info, info_custom_log

    def add_ci_argmax(self, statetup: StateResetId, ic):
        # Set the source, so we can track where the sample came from.
        state_new = jdc.replace(statetup.state, source=Source.BUF_CI)
        statetup = statetup._replace(state=state_new)

        # Add the argmax to the buffer.
        self._buffer = self._buffer.add_to_ci(statetup)

        # Also record it, so we can plot a history of the argmaxes.
        # self.argmax_ci_ic.append(ic)
        self.argmax_ci_ic = np.append(self.argmax_ci_ic, ic)

    def add_explore_argmax(self, statetup: StateResetId, ic):
        # Set the source, so we can track where the sample came from.
        state_new = jdc.replace(statetup.state, source=Source.BUF_EXPLORE)
        statetup = statetup._replace(state=state_new)

        # Add the argmax to the buffer.
        self._buffer = self._buffer.add_to_explore(statetup)

        # Also record it, so we can plot a history of the argmaxes.
        # self.argmax_explore_ic.append(ic)
        self.argmax_explore_ic = np.append(self.argmax_explore_ic, ic)

    def update_predci(self, b_statetup: StateResetId, b_ic):
        n_add = len(b_ic)
        b_source = jnp.full(n_add, fill_value=Source.BASE_PREDCI, dtype=np.int32)
        b_state_new = jdc.replace(b_statetup.state, source=b_source)
        b_statetup = b_statetup._replace(state=b_state_new)
        self._buffer = self._buffer.set_predci(b_statetup, n_add)
        self.predci = b_ic

    @property
    def buffer(self) -> ResetBuf:
        return self._buffer

    @buffer.setter
    def buffer(self, buffer: ResetBuf):
        self._buffer = buffer

    @ft.partial(jax.jit, static_argnums=0)
    def get_b_obs_from_leaf(self, b_x0_leaf):
        b_x0_min = self.task.leaf_to_minstate(b_x0_leaf)
        b_x0 = jax.vmap(self.task.from_minstate)(b_x0_min)
        return jax.vmap(self.task.get_obs)(b_x0)

    def get_x0_obs_nsf(self):
        return self.reset_tracker.x0_obs_buf.get()

    def get_b_x0_ci(self) -> TreeLeaves:
        return self.ci_buf.get_state()

    def get_b_obs_ci(self, b_idx: np.ndarray | None = None) -> BObs:
        return self.ci_buf.get_obs(b_idx)

    def get_b_reset_id(self) -> BObs:
        return self.ci_buf.get_reset_id()

    def get_data_nsf_ci(self) -> BObs:
        n_ci = self.n_ci_buf
        assert n_ci > 0

        b_obs_ci = self.get_b_obs_ci()
        b_idx_ci = self.rng.choice(n_ci, self.cfg.n_sample_nsf_ci, replace=True)  # Replace is True now!

        s_obs_ci = b_obs_ci[b_idx_ci]
        return s_obs_ci

    def get_data_ci_classify(self, num: int):
        n_ci = self.n_ci_buf
        assert n_ci > 0

        if self.cfg.ci_clsfy_fracci is None:
            n_sample_ci_max = num // 2
            n_sample_ci = min(n_ci, n_sample_ci_max)
            n_sample_buf = num - n_sample_ci

            # Sample from the CI.
            s_idx_ci = self.rng.choice(n_ci, n_sample_ci, replace=False)
            s_obs_ci = self.get_b_obs_ci(s_idx_ci)

            # s_obs_ci = b_obs_ci[b_idx_ci]
            s_label_ci = np.full(n_sample_ci, True, dtype=np.bool_)

            # Sample from polcond_obs0.
            s_obs_buf, s_label_buf = self.polcond_obs0.sample(self.rng, size=n_sample_buf, replace=True)
        else:
            # This number is static!
            frac_ci = self.cfg.ci_clsfy_fracci
            n_sample_ci = int(round(frac_ci * num))
            n_sample_buf = num - n_sample_ci

            # Sample from the CI.
            s_idx_ci = self.rng.choice(n_ci, n_sample_ci, replace=True)  # Replace is True now!
            s_obs_ci = self.get_b_obs_ci(s_idx_ci)

            # s_obs_ci = b_obs_ci[s_idx_ci]
            s_label_ci = np.full(n_sample_ci, True, dtype=np.bool_)

            # Sample from polcond_obs0.
            s_obs_buf, s_label_buf = self.polcond_obs0.sample(self.rng, size=n_sample_buf, replace=True)

        obs_is_array = isinstance(s_obs_ci, np.ndarray)
        if obs_is_array:
            s_obs = np.concatenate([s_obs_ci, s_obs_buf], axis=0)
        else:
            s_obs = tree_cat([s_obs_ci, s_obs_buf], axis=0, which=np)

        s_label = np.concatenate([s_label_ci, s_label_buf], axis=0)

        return s_obs, s_label

    def get_data_polcond(self, num: int):
        s_obs, s_label = self.polcond_obs0.sample(self.rng, size=num, replace=True)
        return s_obs, s_label

    def set_rehearsal_only(self):
        self.buffer.set_probs_inplace(0.0, self.cfg.resetbuf.p_base, self.cfg.resetbuf.p_predci)

    @property
    def reset_id_provider(self):
        return self._buffer.id_provider

    @reset_id_provider.setter
    def reset_id_provider(self, reset_id_provider: ResetIDProvider):
        self._buffer = self._buffer.replace(id_provider=reset_id_provider)
