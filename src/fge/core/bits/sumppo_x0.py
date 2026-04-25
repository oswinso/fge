import functools as ft
from typing import Callable, NamedTuple, Self

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from attrs import asdict, define
from flax import struct
from loguru import logger
from og.dyn_types import BControl, BObs
from og.jax_types import BBool, BFloat, FloatDict, FloatScalar, IntScalar
from og.jax_utils import jax_vmap
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.schedules import Schedule, as_schedule
from og.train_state import TrainState
from og.tree_utils import tree_split_dims

from fge.core.bits.ppo_core import SumPPO, SumPPOTrainCfg
from fge.core.bits.state_reset_id import StateResetId
from fge.core.bits.x0_collector import X0Collector, X0Data, X0RolloutOutput
from fge.core.common.ppo_nets import Identity, TanhNormal, ValueNet
from fge.core.common.update_fn import update_mse, update_policy_ppo
from fge.core.envs.jax_task import JaxTask, TimedState
from fge.core.utils.jax_util import myjit


@define
class SumPPOX0TrainCfg:
    # Number of batches during training.
    n_batches: int
    # How many epochs to use per update.
    n_update_epochs: int

    # PPO clip ratio
    clip_ratio: float

    # Whether to normalize the advantages.
    normalize_adv: bool = True

    clip_grad_pol: float = 1.0
    clip_grad_V: float = 1.0

    rollout_T_override: int | None = None


@define
class SumPPOX0Cfg:
    pol_lr: Schedule | float
    val_lr: float
    # In units of update_idx.
    entropy_cf: Schedule | float
    disc_gamma: float

    value_lims: tuple[float, float] | None

    train_cfg: SumPPOX0TrainCfg

    def asdict(self):
        return asdict(self)


class X0ResetBuf(struct.PyTreeNode):
    """
    A buffer for managing reset states in the X0 policy.

    This class stores parameters and a policy function, and provides functionality
    to sample reset states based on the policy.

    Attributes:
        params (dict): Parameters for the policy function.
        policy_fn (Callable[[dict], TimedState]): A function that generates reset states
            based on the provided parameters. This field is not part of the PyTree structure.
    """

    params: dict
    policy_fn: Callable[[dict], TimedState] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, params: dict, policy_fn) -> Self:
        """
        Creates an instance of X0ResetBuf.

        Args:
            params (dict): Parameters for the policy function.
            policy_fn (Callable): A function that generates reset states.

        Returns:
            X0ResetBuf: A new instance of the class.
        """
        return X0ResetBuf(params, policy_fn)

    def sample(
        self, key: PRNGKey, n_samples: int, b_valid: BBool
    ) -> tuple[Self, StateResetId]:
        """
        Samples reset states using the policy function.

        Args:
            key (PRNGKey): A random key for generating samples.
            n_samples (int): The number of samples to generate.
            b_valid (BBool): A boolean mask indicating valid samples.

        Returns:
            tuple:
                - Self: The current instance of X0ResetBuf.
                - StateResetId: A tuple containing the sampled reset states and reset IDs.
        """
        # Partially apply the policy function with the stored parameters.
        policy_fn = ft.partial(self.policy_fn, params=self.params)

        # Split the random key into multiple keys for sampling.
        b_key = jr.split(key, n_samples)

        # Generate reset states by applying the policy function to the keys.
        b_x0 = jax.vmap(policy_fn)(b_key)

        # Create a placeholder array for reset IDs.
        b_resetid = jnp.full(n_samples, -42069, dtype=jnp.int32)

        # Combine the reset states and IDs into a StateResetId tuple.
        b_statetup = StateResetId(b_x0, b_resetid)

        return self, b_statetup


class SumPPOX0(struct.PyTreeNode):
    """
    A class implementing the SumPPOX0 algorithm, which extends the PyTreeNode structure.

    This class manages the training and evaluation of a policy and value function
    using the SumPPO algorithm. It includes configurations, schedules, and methods
    for creating datasets, updating the policy and value function, and collecting rollouts.

    Attributes:
        Cfg (type): Configuration class for SumPPOX0.
        TrainCfg (type): Training configuration class for SumPPOX0.
        update_idx (IntScalar): The current update index.
        key_shuffle (PRNGKey): Random key used for shuffling data.
        key_entropy (PRNGKey): Random key used for entropy calculations.
        policy (TrainState[tfd.Independent]): The policy network's train state.
        Vl (TrainState[FloatScalar]): The value function network's train state.
        ent_cf_sched (optax.Schedule): Entropy coefficient schedule.
        pol_lr_sched (optax.Schedule): Policy learning rate schedule.
        task (JaxTask): The task environment.
        cfg (SumPPOX0Cfg): Configuration for the SumPPOX0 algorithm.
    """

    Cfg = SumPPOX0Cfg
    TrainCfg = SumPPOX0TrainCfg

    update_idx: IntScalar
    key_shuffle: PRNGKey
    key_entropy: PRNGKey
    policy: TrainState[tfd.Independent]
    Vl: TrainState[FloatScalar]

    # In units of update_idx.
    ent_cf_sched: optax.Schedule = struct.field(pytree_node=False)
    pol_lr_sched: optax.Schedule = struct.field(pytree_node=False)

    task: JaxTask = struct.field(pytree_node=False)
    cfg: SumPPOX0Cfg = struct.field(pytree_node=False)

    class PPOBatch(NamedTuple):
        """
        A batch of data used for training the PPO algorithm.

        Attributes:
            b_obs (BObs): Batch of observations.
            b_control (BControl): Batch of control actions.
            b_logprob (BFloat): Batch of log probabilities of actions.
            b_Al (BFloat): Batch of advantages.
            b_Ql (BFloat): Batch of Q-values.
        """

        b_obs: BObs
        b_control: BControl
        b_logprob: BFloat
        b_Al: BFloat
        b_Ql: BFloat

        @property
        def batch_size(self) -> int:
            """
            Returns the size of the batch.

            Returns:
                int: The number of samples in the batch.
            """
            assert self.b_logprob.ndim == 1
            return len(self.b_logprob)

    @classmethod
    def create(cls, key: PRNGKey, task: JaxTask, cfg: SumPPOX0Cfg) -> Self:
        """
        Creates an instance of SumPPOX0.

        Args:
            key (PRNGKey): Random key for initialization.
            task (JaxTask): The task environment.
            cfg (SumPPOX0Cfg): Configuration for the algorithm.

        Returns:
            SumPPOX0: A new instance of the class.
        """
        key = jr.key_data(key)
        key, key_pol, key_Vl = jr.split(key, 3)

        # The observation is just zeros(1).
        obs = np.zeros(1)
        pol_lr = as_schedule(cfg.pol_lr).make()
        pol_lr0 = pol_lr(0)

        # -----------------------
        base_cls = Identity
        control_shape = task.x0_unif_shape
        assert len(control_shape) == 1
        nu = control_shape[0]

        pol_def = TanhNormal(base_cls, nu)

        # Create the policy with the default optimizer.
        pol_tx = get_default_tx(pol_lr0, wd=0.0)
        policy = TrainState.create_from_def(key_pol, pol_def, (obs,), tx=pol_tx)

        V_base_cls = Identity
        Vl_def = ValueNet(V_base_cls, 1, lims=cfg.value_lims)
        Vl_tx = get_default_tx(cfg.val_lr)
        Vl = TrainState.create_from_def(key_Vl, Vl_def, (obs,), tx=Vl_tx)

        ent_cf = as_schedule(cfg.entropy_cf).make()
        zero = jnp.array(0, dtype=np.int32)

        return SumPPOX0(zero, *jr.split(key, 2), policy, Vl, ent_cf, pol_lr, task, cfg)

    @property
    def train_cfg(self):
        """
        Returns the training configuration.

        Returns:
            SumPPOX0TrainCfg: The training configuration.
        """
        return self.cfg.train_cfg

    @property
    def ent_cf(self):
        """
        Returns the current entropy coefficient.

        Returns:
            float: The entropy coefficient.
        """
        return self.ent_cf_sched(self.update_idx)

    @property
    def pol_lr(self):
        """
        Returns the current policy learning rate.

        Returns:
            float: The policy learning rate.
        """
        return self.pol_lr_sched(self.update_idx)

    @property
    def curr_pol_lr(self):
        """
        Returns the current learning rate of the policy.

        Returns:
            float: The current policy learning rate.
        """
        return self.policy.lr

    @property
    def disc_gamma(self):
        """
        Returns the discount factor gamma.

        Returns:
            float: The discount factor.
        """
        return self.cfg.disc_gamma

    def make_dset(self, data: X0Data) -> PPOBatch:
        """
        Creates a dataset for training from the provided data.

        Args:
            data (X0Data): Input data containing observations, controls, and rewards.

        Returns:
            PPOBatch: A batch of data for training.
        """
        b_Vl = jax_vmap(self.Vl.apply)(data.b_obs).squeeze(-1)
        b_Ql = data.b_rew
        b_Al = b_Ql - b_Vl

        if self.train_cfg.normalize_adv:
            mean, std = jnp.mean(b_Al), jnp.std(b_Al)
            b_Al = (b_Al - mean) / (std + 1e-5)

        return SumPPOX0.PPOBatch(data.b_obs, data.b_control, data.b_logprob, b_Al, b_Ql)

    @ft.partial(myjit, donate_argnums=0)
    def update(self, data: X0Data) -> tuple[Self, FloatDict]:
        """
        Updates the policy and value function using the provided data.

        Args:
            data (X0Data): Input data for training.

        Returns:
            tuple:
                - Self: The updated instance of SumPPOX0.
                - FloatDict: A dictionary containing update information.
        """

        def updates_body(alg_: SumPPOX0, b_batch: SumPPOX0.PPOBatch):
            alg_, val_info = alg_.update_value(b_batch)
            alg_, pol_info = alg_.update_policy(b_batch)
            return alg_, val_info | pol_info

        new_self = self
        # Before we do anything, set the policy lr.
        pol_lr = self.pol_lr
        new_self.policy.set_lr(pol_lr)
        key_shuffle = jr.fold_in(self.key_shuffle, self.update_idx)

        for ii in range(self.train_cfg.n_update_epochs):
            # Compute GAE values.
            b_dset = new_self.make_dset(data)

            n_batches = self.train_cfg.n_batches
            assert b_dset.batch_size % n_batches == 0
            batch_size = b_dset.batch_size // self.train_cfg.n_batches
            logger.info(f"Using {n_batches} minibatches each epoch!")

            # 2: Shuffle and reshape
            key_shuffle_ = jr.fold_in(key_shuffle, ii)
            rand_idxs = jr.permutation(key_shuffle_, jnp.arange(b_dset.batch_size))
            b_dset = jtu.tree_map(lambda x: x[rand_idxs], b_dset)
            mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))

            # 3: Perform value function and policy updates.
            new_self, info = lax.scan(updates_body, new_self, mb_dset, length=n_batches)
            # Take the mean.
            info = jtu.tree_map(jnp.mean, info)

        info["steps/policy"] = self.policy.step
        info["steps/Vl"] = self.Vl.step
        info["anneal/ent_cf"] = self.ent_cf
        info["anneal/pol_lr"] = pol_lr
        return new_self.replace(update_idx=self.update_idx + 1), info

    def update_value(self, batch: PPOBatch) -> tuple[Self, FloatDict]:
        """
        Updates the value function using the provided batch.

        Args:
            batch (PPOBatch): A batch of data for training.

        Returns:
            tuple:
                - Self: The updated instance of SumPPOX0.
                - FloatDict: A dictionary containing value update information.
        """
        Vl, Vl_info = update_mse(
            self.Vl,
            batch.b_Ql[:, None],
            batch.b_obs,
            "V",
            clip_grad=self.train_cfg.clip_grad_V,
        )
        return self.replace(Vl=Vl), Vl_info

    def update_policy(self, batch: PPOBatch) -> tuple[Self, FloatDict]:
        """
        Updates the policy using the provided batch.

        Args:
            batch (PPOBatch): A batch of data for training.

        Returns:
            tuple:
                - Self: The updated instance of SumPPOX0.
                - FloatDict: A dictionary containing policy update information.
        """
        key_entropy = jr.split(
            jr.fold_in(self.key_entropy, self.policy.step), batch.batch_size
        )
        train_cfg = self.train_cfg
        policy, pol_info = update_policy_ppo(
            key_entropy,
            self.policy,
            batch.b_obs,
            batch.b_control,
            batch.b_logprob,
            batch.b_Al,
            train_cfg.clip_ratio,
            self.ent_cf,
            clip_grad=train_cfg.clip_grad_pol,
        )
        return self.replace(policy=policy), pol_info

    def get_x0_control_dist(self, params: dict | None = None):
        """
        Retrieves the control distribution for the x0 policy.

        Args:
            params (dict | None): Parameters for the policy. If None, uses the current policy parameters.

        Returns:
            tfd.Independent: The control distribution.
        """
        if params is None:
            params = self.policy.params

        obs = jnp.zeros(1)
        dist: tfd.Independent = self.policy.apply_with(obs, params=params)
        return dist

    def get_x0_control_grid_probs(
        self, mesh_grid: jnp.array, params: dict | None = None
    ):
        """
        Computes the probabilities of control actions on a mesh grid for the x0 policy.

        Args:
            mesh_grid (jnp.array): A mesh grid of control actions.
            params (dict | None): Parameters for the policy. If None, uses the current policy parameters.
        Returns:
            jnp.array: The probabilities of control actions on the mesh grid.
        """
        if params is None:
            params = self.policy.params
        dist = self.get_x0_control_dist(params)
        b_probs = dist.prob(mesh_grid)
        return b_probs

    def sample_x0(self, key: PRNGKey, params: dict | None = None):
        """
        Samples control actions for the x0 policy.

        Args:
            key (PRNGKey): Random key for sampling.
            params (dict | None): Parameters for the policy. If None, uses the current policy parameters.

        Returns:
            Any: The sampled control actions.
        """
        dist = self.get_x0_control_dist(params)
        x0_control = dist.sample(seed=key)
        return self.task.reset_from_box(x0_control)

    @myjit
    def collect(
        self, collector: X0Collector, ppo: SumPPO
    ) -> tuple[X0Collector, X0RolloutOutput, dict]:
        """
        Collects rollouts using the current policy.

        Args:
            collector (X0Collector): The collector for rollouts.
            ppo (SumPPO): The PPO algorithm instance.

        Returns:
            tuple:
                - X0Collector: The updated collector.
                - X0RolloutOutput: The rollout output.
                - dict: Additional information about the collection process.
        """
        rollout_T = self.task.eval_rollout_T
        if self.train_cfg.rollout_T_override is not None:
            rollout_T = self.train_cfg.rollout_T_override
        return collector.collect_batch(
            ppo.policy.apply, self.get_x0_control_dist, rollout_T
        )

    def get_reset_buf(self, reset_buf: X0ResetBuf | None = None):
        """
        Retrieves or creates a reset buffer for the policy rollout.

        Args:
            reset_buf (X0ResetBuf | None): An existing reset buffer. If None, a new one is created.

        Returns:
            X0ResetBuf: The reset buffer with updated parameters.
        """
        if reset_buf is None:
            reset_buf = X0ResetBuf.create(self.policy.params, self.sample_x0)
        return reset_buf.replace(params=self.policy.params)
