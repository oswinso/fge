import functools as ft
from typing import NamedTuple, Self, Sequence

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
from gymnasium.spaces import Box, Discrete
from loguru import logger
from og.dyn_types import BControl, BObs, Obs
from og.jax_types import BFloat, FloatDict, FloatScalar, IntScalar
from og.jax_utils import jax_vmap, merge01
from og.networks.network_utils import get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.schedules import Schedule, as_schedule
from og.train_state import TrainState
from og.tree_utils import tree_split_dims

from fge.core.bits.collector import Collector, ResetFn
from fge.core.bits.gae import compute_gae, compute_qvals
from fge.core.common.ppo_nets import MLP, SoftmaxDiscrete, TanhNormal, ValueNet
from fge.core.common.reset_buf import ResetBuf
from fge.core.common.update_fn import update_mse, update_policy_ppo
from fge.core.envs.jax_task import JaxTask
from fge.core.utils.jax_util import myjit


@define
class SumPPOSauteTrainCfg:
    gae_lambda: float

    # Number of batches during training.
    n_batches: int
    # How long to rollout per collect.
    rollout_T: int
    # How many epochs to use per update.
    n_update_epochs: int

    # PPO clip ratio
    clip_ratio: float

    normalize_adv_pre: bool = True
    """If True, normalize bT_Al_ext and bT_Al_int separately"""

    # Whether to normalize the advantages.
    normalize_adv: bool = True

    clip_grad_pol: float = 1.0
    clip_grad_V: float = 1.0


@define
class SumPPOSauteCfg:
    pol_lr: Schedule | float
    val_lr: float
    # In units of update_idx.
    entropy_cf: Schedule | float
    disc_gamma: float

    pol_hids: Sequence[int]
    val_hids: Sequence[int]

    value_lims: tuple[float, float] | None


    train_cfg: SumPPOSauteTrainCfg

    # cost_penalty: float = 50_000.0
    # cost_penalty: float = 500.0
    cost_penalty: float

    def asdict(self):
        return asdict(self)


class SumPPOSaute(struct.PyTreeNode):
    Cfg = SumPPOSauteCfg
    TrainCfg = SumPPOSauteTrainCfg

    update_idx: IntScalar
    key_shuffle: PRNGKey
    key_entropy: PRNGKey
    policy: TrainState[tfd.Independent]
    Vl: TrainState[FloatScalar]

    # In units of update_idx.
    ent_cf_sched: optax.Schedule = struct.field(pytree_node=False)
    pol_lr_sched: optax.Schedule = struct.field(pytree_node=False)

    task: JaxTask = struct.field(pytree_node=False)
    cfg: SumPPOSauteCfg = struct.field(pytree_node=False)

    class PPOBatch(NamedTuple):
        """
        A batch of data.

        Attributes:
            b_obs (BObs): Batch of observations.
            b_control (BControl): Batch of control actions.
            b_logprob (BFloat): Batch of log probabilities of actions taken.
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
            assert self.b_logprob.ndim == 1  # Ensure b_logprob is a 1D array.
            return len(
                self.b_logprob
            )  # Return the length of b_logprob as the batch size.

    class PPOBatchQvals(NamedTuple):
        """
        A batch of Q-values used in the SumPPOSaute algorithm.
        """

        b_Qvals: BFloat = struct.field(pytree_node=False)

        @property
        def batch_size(self) -> int:
            assert self.b_Qvals.ndim == 1
            return len(self.b_Qvals)

    @property
    def Vl_ext(self):
        """
        Returns the value network for external rewards.
        """
        return self.Vl

    @classmethod
    def create(cls, key: PRNGKey, task: JaxTask, cfg: SumPPOSauteCfg) -> Self:
        """
        Creates an instance of the SumPPOSaute class.

        Args:
            key (PRNGKey): A pseudo-random number generator key for initializing the model.
            task (JaxTask): The task environment providing observation and action space details.
            cfg (SumPPOSauteCfg): Configuration object for the SumPPOSaute algorithm.

        Returns:
            Self: An instance of the SumPPOSaute class.
        """

        # Generate key data and split it into three keys for policy and value network initialization.
        key = jr.key_data(key)
        key, key_pol, key_Vl = jr.split(key, 3)

        # Get a dummy observation from the task environment.
        obs = task.get_dummy_obs()

        # Create a learning rate schedule for the policy network.
        pol_lr = as_schedule(cfg.pol_lr).make()
        pol_lr0 = pol_lr(0)

        # -----------------------
        # Define the base MLP architecture for the policy network.
        base_cls = ft.partial(MLP, act=get_act_from_str("tanh"), hid_sizes=cfg.pol_hids)

        # Define the policy network based on the action space type (Discrete or Box).
        action_space = task.action_space
        if isinstance(action_space, Discrete):
            pol_def = SoftmaxDiscrete(base_cls, action_space.n)
        else:
            assert isinstance(action_space, Box)
            assert len(action_space.shape) == 1
            pol_def = TanhNormal(base_cls, action_space.shape[0])

        # Create the policy network with the specified optimizer and architecture.
        pol_tx = get_default_tx(pol_lr0, wd=0.0)
        policy = TrainState.create_from_def(key_pol, pol_def, (obs,), tx=pol_tx)
        # -----------------------

        # Define the base MLP architecture for the value network.
        V_base_cls = ft.partial(
            MLP, act=get_act_from_str("tanh"), hid_sizes=cfg.val_hids
        )

        # Create the value network with optional output limits.
        Vl_def = ValueNet(V_base_cls, 1, lims=cfg.value_lims)
        Vl_tx = get_default_tx(cfg.val_lr)
        Vl = TrainState.create_from_def(key_Vl, Vl_def, (obs,), tx=Vl_tx)
        # -----------------------

        # Create a schedule for the entropy coefficient.
        ent_cf = as_schedule(cfg.entropy_cf).make()

        # Initialize the update index to zero.
        zero = jnp.array(0, dtype=np.int32)

        # Return an instance of the SumPPOSaute class with the initialized components.
        return SumPPOSaute(zero, *jr.split(key, 2), policy, Vl, ent_cf, pol_lr, task, cfg)

    @property
    def train_cfg(self):
        """
        Returns the training configuration for the SumPPOSaute algorithm.
        """
        return self.cfg.train_cfg

    @property
    def ent_cf(self):
        """
        Returns the entropy coefficient for the policy, which encourages exploration.
        """
        return self.ent_cf_sched(self.update_idx)

    @property
    def pol_lr(self):
        """
        Returns the current learning rate for the policy network.
        """
        return self.pol_lr_sched(self.update_idx)

    @property
    def curr_pol_lr(self):
        """
        Returns the current learning rate for the policy network without using the schedule.
        """
        return self.policy.lr

    @property
    def disc_gamma(self):
        """
        Returns the discount factor for future rewards used in the SumPPOSaute algorithm.
        """
        return self.cfg.disc_gamma

    def compute_bT_A_Q(self, data: Collector.Rollout):
        # 1: Compute h_Vl from data. h_Vl is the value function output for the current and next observations.
        bT_Vl_nxt = jax_vmap(self.Vl.apply, rep=2)(data.T_obs_nxt).squeeze(-1)
        bT_Vl_now = jax_vmap(self.Vl.apply, rep=2)(data.T_obs_now).squeeze(-1)

        # 2: Compute Ql using GAE.
        gae_fn = ft.partial(compute_gae, self.disc_gamma, self.train_cfg.gae_lambda)

        # bT_rew = data.T_rew
        # bT_cost = -bT_rew

        # TODO
        # assert len(data.T_control.shape) == 2, (f"Expected T_control to have shape (B, T), got {data.T_control.shape}. "
        #                                         f"This is hardcoded for ToyLevels! If you want to do another env, take the norm.")

        # bT_cost = data.T_control**2  # (B, T)
        # TODO: ctrl cost
        if len(data.T_control.shape) == 2:
            bT_cost = data.T_control**2  # (B, T)
        elif len(data.T_control.shape) == 3:
            bT_cost = jnp.linalg.norm(data.T_control, axis=-1, ord=2)

        # anywhere we have nonzero cost in info, make reward -cost_penalty
        collided = data.T_info['collided']
        bT_cost = jnp.where(collided, 1, bT_cost / self.cfg.cost_penalty)

        bT_isterm = data.T_term
        bT_nextvalid = (~bT_isterm)

        # Compute the advantages and Q-values using GAE.
        bT_Ql, bT_Al = jax_vmap(gae_fn)(bT_cost, bT_Vl_now, bT_Vl_nxt, bT_nextvalid)
        assert bT_Ql.shape == bT_Al.shape == bT_cost.shape

        return bT_Al, bT_Ql

    def make_bT_dset(self, data: Collector.Rollout) -> tuple[PPOBatch, dict]:
        bT_Al, bT_Ql = self.compute_bT_A_Q(data)

        # Compute statistics for the advantages.
        mean, std = jnp.mean(bT_Al), jnp.std(bT_Al)
        info = {
            "gae/min": bT_Al.min(),
            "gae/q10": jnp.quantile(bT_Al, 0.1),
            "gae/mean": mean,
            "gae/std": std,
            "gae/q90": jnp.quantile(bT_Al, 0.9),
            "gae/max": bT_Al.max(),
        }

        # Normalize advantages if specified in the training configuration.
        if self.train_cfg.normalize_adv:
            bT_Al = (bT_Al - mean) / (std + 1e-5)

        # Package the processed data into a PPOBatch object.
        bT_batch = SumPPOSaute.PPOBatch(
            data.T_obs_now, data.T_control, data.T_logprob, bT_Al, bT_Ql
        )
        return bT_batch, info

    def make_bT_dset_Qvals_only(
            self, data: Collector.Rollout
    ) -> tuple[PPOBatchQvals, dict]:
        qval_fn = ft.partial(compute_qvals, self.disc_gamma)
        # Compute Q-values using the provided function.
        dones = data.T_term | data.T_trunc  # Combine terminal and truncated states.
        bT_nextvalid = (
            ~dones
        )  # Shape: (b, T). This indicates whether the next state is valid.
        bT_Qvals = jax_vmap(qval_fn)(data.T_rew, bT_nextvalid)
        assert (
                bT_Qvals.shape == data.T_rew.shape
        )  # Ensure the shape matches the rewards tensor.
        # Compute statistics for the Q-values.
        info = {
            "Qvals/min": bT_Qvals.min(),
            "Qvals/q10": jnp.quantile(bT_Qvals, 0.1),
            "Qvals/mean": jnp.mean(bT_Qvals),
            "Qvals/std": jnp.std(bT_Qvals),
            "Qvals/q90": jnp.quantile(bT_Qvals, 0.9),
            "Qvals/max": bT_Qvals.max(),
        }
        # Package the processed data into a PPOBatch object with Q-values.
        bT_batch = SumPPOSaute.PPOBatchQvals(bT_Qvals)
        return bT_batch, info

    def make_dset(self, data: Collector.Rollout) -> tuple[PPOBatch, dict]:
        """
        Creates a dataset for training by processing a rollout of data.

        Args:
            data (Collector.Rollout): A rollout object containing observations, actions, rewards,
                                      and other relevant information.

        Returns:
            tuple:
                - PPOBatch: A batch of processed data including observations, controls,
                            log probabilities, advantages, and Q-values.
                - dict: A dictionary containing statistics about the computed advantages.
        """
        # Process the rollout data into a batch dataset.
        bT_batch, info = self.make_bT_dset(data)

        # Merge the first two dimensions of the batch dataset (e.g., batch and time).
        b_batch = jtu.tree_map(merge01, bT_batch)

        # Return the processed batch dataset and the associated statistics.
        return b_batch, info

    @ft.partial(myjit, donate_argnums=0)
    def update(self, data: Collector.Rollout, **kwargs) -> tuple[Self, FloatDict]:
        """
        Updates the policy and value networks using the provided rollout data.

        Args:
            data (Collector.Rollout): A rollout object containing observations, actions, rewards,
                                      and other relevant information.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            tuple:
                - Self: The updated instance of the SumPPOSaute class.
                - FloatDict: A dictionary containing statistics and information about the update process.
        """

        def updates_body(alg_: SumPPOSaute, b_batch: SumPPOSaute.PPOBatch):
            """
            Performs a single update step for the value and policy networks.

            Args:
                alg_ (SumPPOSaute): The current instance of the SumPPOSaute class.
                b_batch (SumPPOSaute.PPOBatch): A batch of data for training.

            Returns:
                tuple:
                    - SumPPOSaute: The updated instance of the SumPPOSaute class.
                    - FloatDict: A dictionary containing statistics for the value and policy updates.
            """
            alg_, val_info = alg_.update_value(b_batch)  # Update the value function.
            alg_, pol_info = alg_.update_policy(b_batch)  # Update the policy.
            return alg_, val_info | pol_info

        # Initialize a new instance of SumPPOSaute for the update.
        new_self = self
        # Before we do anything, set the policy learning rate.
        pol_lr = self.pol_lr
        new_self.policy.set_lr(pol_lr)
        key_shuffle = jr.fold_in(self.key_shuffle, self.update_idx)

        for ii in range(self.train_cfg.n_update_epochs):
            # Compute GAE values and create a dataset.
            b_dset, info_gae = new_self.make_dset(data)

            # n_batches is the number of minibatches to split the dataset into.
            n_batches = self.train_cfg.n_batches
            assert (
                    b_dset.batch_size % n_batches == 0
            )  # Ensure the dataset can be evenly split into batches.
            batch_size = b_dset.batch_size // self.train_cfg.n_batches
            logger.info(f"Using {n_batches} minibatches each epoch!")

            # Shuffle and reshape the dataset.
            key_shuffle_ = jr.fold_in(key_shuffle, ii)
            rand_idxs = jr.permutation(key_shuffle_, jnp.arange(b_dset.batch_size))
            b_dset = jtu.tree_map(
                lambda x: x[rand_idxs], b_dset
            )  # Shuffle the dataset.
            mb_dset = tree_split_dims(
                b_dset, (n_batches, batch_size)
            )  # Split the dataset into minibatches.

            # Perform value function and policy updates for each minibatch.
            new_self, info = lax.scan(updates_body, new_self, mb_dset, length=n_batches)
            # Take the mean of the update statistics across all minibatches.
            info = jtu.tree_map(jnp.mean, info)

        # Merge the GAE statistics with the update statistics.
        info = info | info_gae

        # Add additional information about the current steps and annealing parameters.
        info["steps/policy"] = self.policy.step  # Current step of the policy network.
        info["steps/Vl"] = self.Vl.step  # Current step of the value network.
        info["anneal/ent_cf"] = self.ent_cf  # Current entropy coefficient.
        info["anneal/pol_lr"] = pol_lr  # Current policy learning rate.

        # Increment the update index and return the updated instance and statistics.
        return new_self.replace(update_idx=self.update_idx + 1), info

    def update_value(self, batch: PPOBatch) -> tuple[Self, FloatDict]:
        """
        Updates the value network using the provided batch of data.

        Args:
            batch (PPOBatch): A batch of data that includes:
                - b_Ql: Batch of Q-values, used as targets for the value network.
                - b_obs: Batch of observations, used as inputs to the value network.

        Returns:
            tuple:
                - Self: The updated instance of the SumPPO class with the updated value network.
                - FloatDict: A dictionary containing statistics about the value network update,
                  such as loss and gradient clipping information.
        """
        # Perform the value network update using mean squared error loss.
        Vl, Vl_info = update_mse(
            self.Vl,  # Current value network.
            batch.b_Ql[
            :, None
            ],  # Q-values as targets, reshaped to match the network's output shape.
            batch.b_obs,  # Observations as inputs to the value network.
            "V",  # Identifier for logging or debugging purposes.
            clip_grad=self.train_cfg.clip_grad_V,  # Gradient clipping value from the training configuration.
        )
        # Replace the current value network with the updated one and return the result.
        return self.replace(Vl=Vl), Vl_info

    def update_policy(self, batch: PPOBatch) -> tuple[Self, FloatDict]:
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

    def get_policy_params(self):
        return self.policy.params

    @ft.partial(jax.jit, donate_argnums=0)
    def set_policy_params(self, params: dict):
        # Make sure the params are compatible with the current policy params.
        if jtu.tree_structure(self.policy.params) != jtu.tree_structure(params):
            raise ValueError(
                "Loaded params structure does not match current policy params structure."
            )

        # Load the parameters into the policy.
        policy = self.policy.replace(params=params)
        return self.replace(policy=policy)

    def policy_det(self, obs: Obs):
        """
        Computes the deterministic policy for a given observation.

        Args:
            obs (Obs): The input observation for which the deterministic policy is computed.

        Returns:
            tfd.Deterministic or tfd.VectorDeterministic:
                - If the control output is scalar (0-dimensional), returns a `tfd.Deterministic` distribution.
                - If the control output is a vector (1-dimensional or higher), returns a `tfd.VectorDeterministic` distribution.
        """
        control = self.policy.apply(
            obs
        ).mode()  # Compute the mode of the policy distribution for the given observation.
        if control.ndim == 0:
            return tfd.Deterministic(
                control
            )  # Return a scalar deterministic distribution.
        else:
            return tfd.VectorDeterministic(
                control
            )  # Return a vector deterministic distribution.

    @property
    def rollout_T(self) -> int:
        """
        Returns the length of the rollout for training.
        """
        return self.train_cfg.rollout_T

    @ft.partial(myjit, donate_argnums=1)
    def collect(
            self, collector: Collector
    ) -> tuple[Collector, Collector.Rollout, dict]:
        """
        Collects a batch of data using the provided collector and the current policy.

        This method uses a custom jit called `myjit` to compile and optimize the function for performance.
        The `donate_argnums=1` argument allows JAX to reuse the memory of the `collector` argument,
        which can improve efficiency.

        Args:
            collector (Collector): The collector object responsible for gathering data
                                   from the environment using the current policy.

        Returns:
            tuple:
                - Collector: The updated collector after data collection.
                - Collector.Rollout: The collected rollout data, including observations, actions, and rewards.
                - dict: Additional information or statistics about the collection process.
        """
        logger.debug("jitting collect...")
        out = collector.collect_batch(self.policy.apply, self.train_cfg.rollout_T)
        logger.debug("jitting collect... done!")
        return out

    @myjit
    def collect_eval(self, collector: Collector) -> tuple[Collector.Rollout, dict]:
        """
        Collects evaluation data using the provided collector and the current policy. Note that since we're using
        the policy.apply method, it returns a probability distribution over actions.

        Returns:
            tuple:
                - Collector.Rollout: The collected rollout data, including observations, actions, and rewards.
                - dict: Additional information or statistics about the evaluation process.
        """
        logger.debug("jitting collect_eval...")
        _, rollout, info = collector.collect_batch(
            self.policy.apply, self.task.eval_rollout_T
        )
        logger.debug("jitting collect_eval... done!")
        return rollout, info

    @myjit
    def collect_eval_w_col(
            self, collector: Collector
    ) -> tuple[Collector, Collector.Rollout, dict]:
        """
        Collects evaluation data using the provided collector and the current policy. Note that since we're using
        the policy.apply method, it returns a probability distribution over actions.

        Returns:
            tuple:
                - Collector.Rollout: The collected rollout data, including observations, actions, and rewards.
                - dict: Additional information or statistics about the evaluation process.
        """
        return collector.collect_batch(self.policy.apply, self.task.eval_rollout_T)

    @ft.partial(myjit, static_argnums=2)
    def collect_eval_w_col_spec_T(
            self, collector: Collector, T: int
    ) -> tuple[Collector, Collector.Rollout, dict]:
        """
        Collects evaluation data using the provided collector and the current policy. Note that since we're using
        the policy.apply method, it returns a probability distribution over actions.

        Returns:
            tuple:
                - Collector.Rollout: The collected rollout data, including observations, actions, and rewards.
                - dict: Additional information or statistics about the evaluation process.
        """
        return collector.collect_batch(self.policy.apply, T)

    @myjit
    def collect_eval_det(self, collector: Collector) -> tuple[Collector.Rollout, dict]:
        logger.debug("jitting collect_eval_det...")
        _, rollout, info = collector.collect_batch(
            self.policy_det, self.task.eval_rollout_T
        )
        logger.debug("jitting collect_eval_det... done!")
        return rollout, info

    @ft.partial(myjit, donate_argnums=1)
    def collect_with_buf(
            self, collector: Collector, reset_buf: ResetBuf
    ) -> tuple[Collector, ResetBuf, Collector.Rollout, dict]:
        return collector.collect_batch_with_buf(
            self.policy.apply, reset_buf, self.train_cfg.rollout_T
        )

    @ft.partial(myjit, donate_argnums=1)
    def collect_with_fn(
            self, collector: Collector, reset_fn: ResetFn
    ) -> tuple[Collector, Collector.Rollout, dict]:
        return collector.collect_batch_with_fn(
            self.policy.apply, reset_fn, self.train_cfg.rollout_T
        )
