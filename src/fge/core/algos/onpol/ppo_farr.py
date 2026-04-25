import copy
import functools as ft
import pathlib
import time
from typing import Any, Callable, Self

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm
from attr import asdict
from attrs import define
from chex import PRNGKey
from flax import nnx, struct
from loguru import logger
from og.ckpt_utils import get_ckpt_manager
from og.jax_utils import jax2np
from og.tree_utils import make_batch_pytree
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.onpol.nash import fictitious_play
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, TrainCfg, make_collector_evals
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector, CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.sumppo_x0 import SumPPOX0, X0ResetBuf
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import (
    JaxTask,
    TimedState,
    TreeLeaves,
    leaf_index,
    leaf_set,
    leaf_stack,
    leaf_to_jax,
)
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class FARRCfg:
    n_ppo_steps: int
    """How many ppo steps to run to train the protagonist."""

    n_fp_iters: int = 10_000
    """How many iterations of fictitious play to run to solve the meta-game."""

    n_env_eval: int = 256
    """How many environments to use for evaluating the performance of the policy to add to the payoff matrix."""

    n_theta_per_iter: int = 3


@define
class PPOFARRCfg:
    ppo: SumPPO.Cfg
    farr: FARRCfg

    def asdict(self):
        return asdict(self)


class SingleX0ResetFn(struct.PyTreeNode):
    x0: Any

    @classmethod
    def create(cls, x0) -> Self:
        return SingleX0ResetFn(x0)

    def __call__(self, b_key: PRNGKey, num: int) -> TimedState:
        return make_batch_pytree(self.x0, num, fill_value="repeat")


class PPOForSingleX0:
    def __init__(
        self,
        rngs: nnx.Rngs,
        task: JaxTask,
        cfg_ppo: SumPPOCfg,
        cfg_col: CollectorCfg,
        cfg_col_eval: CollectorCfg,
    ):
        gpu_device = jax.devices("gpu")[0]

        self.task = task
        self.cfg_ppo = cfg_ppo
        self.cfg_col = cfg_col
        self.cfg_col_eval = cfg_col_eval

        self.key_resetalg_base = rngs()

        ppo = SumPPO.create(rngs(), task, cfg_ppo)
        self.ppo: SumPPO = jax.device_put(ppo, gpu_device)

        # We're not gonna use this state in the init because we don't know what the x0 will be yet.
        collector, _ = Collector.create(rngs(), task, cfg_col)
        self.collector = jax.device_put(collector, gpu_device)

        self.n_collects_global = 0

    # @ft.partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    @ft.partial(jax.jit, static_argnums=0)
    def reset_alg(
        self, x0, ppo: SumPPO, collector: Collector, idx: int
    ) -> tuple[SumPPO, Collector]:
        """Reset the algorithm to a new state."""
        key = jr.fold_in(self.key_resetalg_base, idx)
        key_ppo, key_col = jr.split(key, 2)

        same_x0_reset_fn = SingleX0ResetFn(x0)

        ppo = SumPPO.create(key_ppo, self.task, self.cfg_ppo)

        collector = Collector.create_resetfn(
            key_col, self.task, same_x0_reset_fn, self.cfg_col
        )

        return ppo, collector

    def train_on_x0(
        self, x0: Any, run_cfg: RunCfg, fast_cbs: list[Callable], writer: SummaryWriter
    ):
        gpu_device = jax.devices("gpu")[0]

        self.ppo, self.collector = self.reset_alg(
            x0, self.ppo, self.collector, self.n_collects_global
        )

        # trajsaver = FastTrajSaver()
        trajsaver = TrajSaver(save_full_traj=True)

        train_cfg = run_cfg.train_cfg

        same_x0_reset_fn = SingleX0ResetFn(x0)

        # collector_eval, _ = Collector.create_resetfn(jr.PRNGKey(0), self.task, same_x0_reset_fn, self.cfg_col_eval)
        # collector_eval = jax.device_put(collector_eval, device=gpu_device)

        pbar = tqdm.trange(train_cfg.n_steps)
        for n_collects in pbar:
            # 1: Collect rollout. Always reset with the same x0.
            pbar.set_description("[Inner] Collecting...")
            rollout: Collector.Rollout
            self.collector, rollout, _ = self.ppo.collect_with_fn(
                self.collector, same_x0_reset_fn
            )

            # If the rollout has NaN, then debug.
            if not all(
                [jnp.isfinite(l).all() for l in jtu.tree_leaves(rollout.T_state_now)]
            ):
                logger.critical("[Inner] Rollout has NaN values!")
                ipdb.set_trace()

            # 2: Update PPO.
            pbar.set_description("[Inner] Updating PPO...")
            self.ppo, info_loss = self.ppo.update(rollout)

            # 3: Save the trajs.
            pbar.set_description("[Inner] Add rollout...")
            rollout = jax2np(rollout)
            trajsaver.add_rollout(rollout)

            # # 4: Maybe eval.
            # if n_collects % train_cfg.eval_every == 0:
            #     pbar.set_description("Eval...")
            #     rollout_trunc = split_trajs(jax2np(self.ppo.collect_eval(self.collector_eval)[0]))
            #     rollouts_eval = [rollout_trunc]

            if (train_cfg.fastcb_every > 0) and (
                n_collects % train_cfg.fastcb_every == 0
            ):
                pbar.set_description("[Inner] Fast Eval...")
                props = EvalProps(writer, self.ppo, trajsaver, None, None, None)
                for cb in fast_cbs:
                    cb(self.n_collects_global, run_cfg, props)
                pbar.set_description("[Inner] Fast Eval... Done!")

            self.n_collects_global += 1

        return self.ppo


class FARRResetFn(struct.PyTreeNode):
    b_x0s: TreeLeaves
    b_px0: jnp.ndarray

    env: JaxTask = struct.field(pytree_node=False)

    @classmethod
    def create(cls, env, b_x0s: TreeLeaves, b_px0: jnp.ndarray) -> Self:
        """Create a reset function that resets the environment to the given x0s."""
        # Make sure that they have the same shape.
        b = len(b_x0s[0])
        assert b_px0.shape == (b,), "b_px0 should have the same batch size as b_x0s."

        # Convert b_x0 to jax array.
        b_x0s = leaf_to_jax(b_x0s)
        return FARRResetFn(b_x0s, b_px0, env)

    def __call__(self, b_key: PRNGKey, num: int) -> TimedState:
        # Sample from the b_px0 distribution to get the indices for x0.
        def sample_single_x0(key: PRNGKey):
            idx = jr.categorical(key, jnp.log(self.b_px0))
            leaf = leaf_index(self.b_x0s, idx)
            # Convert from leaf back to state.
            return self.env.leaf_to_state(leaf)

        b_x0 = jax.vmap(sample_single_x0)(b_key)
        return b_x0


class FARRBuf:
    """Buffer for solving the meta-game Nash equilibrium using fictitious play."""

    def __init__(self, task: JaxTask, x0_dummy: TimedState, n_fp_iters: int = 10_000):
        initial_size = 1024

        self.task = task
        self.n_fp_iters = n_fp_iters

        # Treeleaves.
        x0_leaf = jax2np(task.state_to_leaf(x0_dummy))
        self.x0s = make_batch_pytree(x0_leaf, initial_size, fill_value=0)
        self.rews = []

        self.x0_size = initial_size

        # Holds the policy params.
        self.policies = []

        self.payoff = np.zeros((0, 0), dtype=np.float32)

        self.rew_thresh = -0.9

        # Large constant to use for infeasible x0s to discourage adversary from choosing it.
        self.C = 1e6

    def is_feasible(self, x0_idx: int) -> bool:
        assert (
            0 <= x0_idx < self.n_x0s
        ), "x0_idx should be in the range of existing x0s."
        rew = self.rews[x0_idx]
        return rew >= self.rew_thresh

    @property
    def n_x0s(self) -> int:
        return len(self.rews)

    @property
    def n_policies(self) -> int:
        return len(self.policies)

    def get_x0(self, x0_idx: int) -> Any:
        leaf = leaf_index(self.x0s, x0_idx)
        return self.task.leaf_to_state(leaf)

    def add_x0(self, x0: Any, rew: float):
        x0_leaf = self.task.state_to_leaf(x0)
        leaf_set(self.x0s, self.n_x0s, x0_leaf)
        self.rews.append(rew)

        # Resize the payoff matrix by adding a new column.
        self.payoff = np.concatenate(
            [self.payoff, np.zeros((self.n_policies, 1), dtype=np.float32)], axis=1
        )
        assert self.payoff.shape == (self.n_policies, self.n_x0s)

        # If it's infeasible (i.e., rew = -1), then set the payoff to C = large positive number,
        # so the adversary does not select this x0.
        if rew < self.rew_thresh:
            self.payoff[:, -1] = self.C

    def update_payoff(self, pol_idx: int, x0_idx: int, rew: float):
        """Update the payoff matrix for a given policy and x0."""
        self.payoff[pol_idx, x0_idx] = rew

    def update_payoff_for_x0(self, m_rews: np.ndarray, x0_idx: int = -1):
        assert m_rews.shape == (
            self.n_policies,
        ), "The x0 should have been evaluated against all existing policies."
        self.payoff[:, x0_idx] = m_rews

    def add_policy(self, pol_params, n_rews: np.ndarray):
        """Add a new policy to the buffer."""
        assert n_rews.shape == (
            self.n_x0s,
        ), "The new policy should have been evaluated against all existing x0s."
        self.policies.append(pol_params)

        self.payoff = np.concatenate([self.payoff, n_rews[None, :]], axis=0)
        assert self.payoff.shape == (self.n_policies, self.n_x0s)

    def get_prob_dist(self):
        _, b_px0 = fictitious_play(self.payoff, iters=self.n_fp_iters, rng=None)
        assert b_px0.shape == (
            self.n_x0s,
        ), "b_px0 should have the same batch size as the number of x0s."
        return b_px0

    def get_reset_fn(self) -> FARRResetFn:
        # Solve for the mixed strategy sigma_theta.
        b_px0 = self.get_prob_dist()
        assert b_px0.shape == (
            self.n_x0s,
        ), "b_px0 should have the same batch size as the number of x0s."

        if self.n_x0s < self.x0_size:
            # Pad b_px0 to be the same size as x0s.
            b_px0 = np.concatenate([b_px0, np.zeros(self.x0_size - self.n_x0s)])

        b_px0 = jnp.array(b_px0, dtype=jnp.float32)
        return FARRResetFn.create(self.task, self.x0s, b_px0)


@ft.partial(jax.jit, static_argnames=("col_cfg",))
def rollout_for_eval_x0_(ppo: SumPPO, x0: Any, col_cfg: CollectorCfg):
    gpu_device = jax.devices("gpu")[0]

    same_x0_reset_fn = SingleX0ResetFn(x0)
    collector_eval = Collector.create_resetfn(
        jr.PRNGKey(0), ppo.task, same_x0_reset_fn, col_cfg
    )
    collector_eval = jax.device_put(collector_eval, device=gpu_device)

    rollout, info = ppo.collect_eval(collector_eval)
    return rollout


# @ft.partial(jax.jit, static_argnames=("col_cfg",))
def evaluate_policy_on_x0(ppo: SumPPO, x0: Any, col_cfg: CollectorCfg) -> np.ndarray:
    rollout_trunc = split_trajs(jax2np(rollout_for_eval_x0_(ppo, x0, col_cfg)))
    # Compute the mean total reward.
    b_rews = [np.sum(r.T_rew) for r in rollout_trunc]
    mean_rew = np.mean(b_rews)
    return mean_rew


def train_ppo_farr(
    run_cfg: RunCfg[PPOFARRCfg],
    seed,
    fast_cbs: list[Callable] = [],
    fast_cbs_inner: list[Callable] = [],
    eval_cbs: list[Callable] = [],
):
    start_time = time.time()
    train_cfg = run_cfg.train_cfg
    writer = SummaryWriter(run_cfg.paths.run_dir)

    env = make_task(run_cfg.task_cfg)

    rngs = nnx.Rngs(seed)

    gpu_device = jax.devices("gpu")[0]

    ppo = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo: SumPPO = jax.device_put(ppo, gpu_device)

    ppo_for_eval = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo_for_eval = jax.device_put(ppo_for_eval, gpu_device)

    ppo_for_x0 = PPOForSingleX0(
        rngs, env, run_cfg.algo_cfg.ppo, run_cfg.col_cfg, run_cfg.col_cfg
    )

    # Get the eval collectors.
    collector_evals, eval_region_info = make_collector_evals(
        env, run_cfg.eval_cfg.n_seeds
    )
    collector_evals = jax.device_put(collector_evals, gpu_device)

    # Setup ckpting.
    ckpt_cfgs = {
        "seed": seed,
        "task_cfg": run_cfg.task_cfg,
        "alg_cfg": run_cfg.algo_cfg,
    }
    ckpt_dir = run_cfg.paths.models_dir
    ckpt_manager = get_ckpt_manager(
        ckpt_dir, ["alg"], max_to_keep=None, step_format_fixed_length=8
    )
    ckpt_manager.save_config(ckpt_cfgs)

    no_trajsaver = False
    if isinstance(env, (F16AvoidJax, HopperJax)):
        no_trajsaver = True

    if isinstance(env, ToyLevelsJax):
        save_full_traj = True
    else:
        save_full_traj = False

    if no_trajsaver:
        trajsaver = None
    else:
        if save_full_traj:
            trajsaver = TrajSaver(save_full_traj=save_full_traj)
        else:
            trajsaver = FastTrajSaver()
    # trajsaver = None

    farr_cfg = run_cfg.algo_cfg.farr

    key_sample_x0 = rngs()

    @jax.jit
    def sample_x0(i_):
        key_ = jr.fold_in(key_sample_x0, i_)
        x0_ = ppo.task.reset(key_)
        return x0_

    n_thetas_sampled = 0
    x0_dummy = sample_x0(n_thetas_sampled)
    n_thetas_sampled += 1
    farr_buf = FARRBuf(env, x0_dummy, farr_cfg.n_fp_iters)
    del x0_dummy

    col_cfg_eval = CollectorCfg(n_envs=farr_cfg.n_env_eval)

    run_cfg_inner = copy.deepcopy(run_cfg)
    # Change the number of steps to be equal to n_ppo_steps.
    run_cfg_inner.train_cfg.n_steps = run_cfg.algo_cfg.farr.n_ppo_steps

    collector = None

    reset_fn = None

    # Since there's the BR stuff, figure out how many outer iterations will result in the same total number of steps.
    n_steps_total = train_cfg.n_steps
    n_env_steps_total = n_steps_total * run_cfg.col_cfg.n_envs * ppo.train_cfg.rollout_T

    # Every n_ppo_step outer iterations (= n_ppo_step * n_env * rollout_T env steps)
    # we do n_theta_per_iter * n_envs * rollout_T additional steps
    # => (n_theta + 1) * n_ppo_step * n_env * rollout_T steps in total
    # If we want this to equal to n_env_steps, then
    n_steps_real = np.ceil(n_steps_total / (farr_cfg.n_theta_per_iter + 1))
    n_steps_real = int(n_steps_real)

    logger.info(
        "Because of the BR stuff, n_steps: {} -> {}".format(
            train_cfg.n_steps, n_steps_real
        )
    )
    train_cfg.n_steps = n_steps_real

    n_steps = 0
    pbar = tqdm.trange(n_steps_real)
    for n_collects in pbar:
        final_iter = n_collects >= train_cfg.n_steps - 1

        if n_collects == 0:
            # In the first iteration, we need at least one x0 to start with.
            assert farr_buf.n_x0s == 0, "Buffer should be empty at the start."
            x0 = sample_x0(n_thetas_sampled)
            n_thetas_sampled += 1
            ppo_br = ppo_for_x0.train_on_x0(x0, run_cfg_inner, fast_cbs_inner, writer)

            # BR runs n_steps iters, each samples rollout_T steps for n_envs.
            rollout_T = ppo_for_x0.ppo.rollout_T
            n_steps += (
                run_cfg_inner.train_cfg.n_steps * rollout_T * run_cfg.col_cfg.n_envs
            )

            rew = evaluate_policy_on_x0(ppo_br, x0, col_cfg_eval)
            farr_buf.add_x0(x0, rew)

            # We also need at least one policy. Use the initial policy.
            #     Evaluate the initial policy on the x0.
            rew = evaluate_policy_on_x0(ppo, x0, col_cfg_eval)
            farr_buf.add_policy(jax2np(ppo.get_policy_params()), np.array([rew]))

            reset_fn = farr_buf.get_reset_fn()
            collector = Collector.create_resetfn(rngs(), env, reset_fn, run_cfg.col_cfg)
            collector = jax.device_put(collector, gpu_device)

        # 2: Train (from scratch !?!?) but don't do it because it'll be too slow....
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        collector, rollout, _ = ppo.collect_with_fn(collector, reset_fn)
        jax.copy_to_host_async(rollout)

        # If the rollout has NaN, then debug.
        if not all(
            [jnp.isfinite(l).all() for l in jtu.tree_leaves(rollout.T_state_now)]
        ):
            T_state_now = rollout.T_state_now
            logger.critical("Rollout has NaN values!")
            ipdb.set_trace()

        n_steps += ppo.train_cfg.rollout_T * run_cfg.col_cfg.n_envs

        # 3: Train PPO.
        pbar.set_description("Update PPO...")
        ppo, info_loss = ppo.update(rollout)

        if (n_collects + 1) % farr_cfg.n_ppo_steps == 0:
            # If we have converged enough, add the new policy, evaluate against existing x0s,
            # sample thetas, do their best responses and save.

            # 4: Sample new x0s and train on them.
            x0s_new = []
            x0s_idx = farr_buf.n_x0s + np.arange(farr_cfg.n_theta_per_iter)
            for theta_idx in range(farr_cfg.n_theta_per_iter):
                # a) Sample a random theta.
                x0 = sample_x0(n_thetas_sampled)
                n_thetas_sampled += 1

                # b) Train on the sampled x0 for a fixed number of iterations and evaluate the feasibility.
                ppo_br = ppo_for_x0.train_on_x0(
                    x0, run_cfg_inner, fast_cbs_inner, writer
                )
                rew = evaluate_policy_on_x0(ppo_br, x0, col_cfg_eval)

                # BR runs n_steps iters, each samples rollout_T steps for n_envs.
                rollout_T = ppo_for_x0.ppo.rollout_T
                n_steps += (
                    run_cfg_inner.train_cfg.n_steps * rollout_T * run_cfg.col_cfg.n_envs
                )

                # c) Add the sampled x0 and its feasibility to the buffer
                farr_buf.add_x0(x0, rew)
                x0s_new.append(x0)

            # 5: Add the new policy first. Evaluate current policy on all x0s.
            pbar.set_description("Add policy - Evaluating against all x0s...")
            pbar2 = tqdm.trange(farr_buf.n_x0s, desc="Evaluating new policy")
            rews = []
            for x0_idx in pbar2:

                if farr_buf.is_feasible(x0_idx):
                    x0 = farr_buf.get_x0(x0_idx)
                    rew = evaluate_policy_on_x0(ppo, x0, col_cfg_eval)
                else:
                    # If the x0 is infeasible, then we don't evaluate it.
                    rew = farr_buf.C

                rews.append(float(rew))
            rews = np.array(rews)
            farr_buf.add_policy(jax2np(ppo.get_policy_params()), rews)

            # 6: Evaluate all existing policies on all new x0s.
            pbar.set_description("Eval policies on new x0s...")
            pbar2 = tqdm.trange(farr_buf.n_policies - 1)
            for pol_idx in pbar2:
                # Load the policy parameters.
                pol_params = farr_buf.policies[pol_idx]
                ppo_for_eval = ppo_for_eval.set_policy_params(pol_params)

                for x0_idx, x0 in zip(x0s_idx, x0s_new):

                    if farr_buf.is_feasible(x0_idx):
                        rew = evaluate_policy_on_x0(ppo_for_eval, x0, col_cfg_eval)
                    else:
                        # If the x0 is infeasible, then we don't evaluate it.
                        rew = farr_buf.C

                    farr_buf.update_payoff(pol_idx, x0_idx, float(rew))

            # Update the reset fn.
            reset_fn = farr_buf.get_reset_fn()

        # 6: Save trajs.
        if trajsaver is not None:
            pbar.set_description("Add rollout...")
            rollout_cpu = jax2np(rollout)
            trajsaver.add_rollout(rollout_cpu)

        # 7: Maybe log.
        pbar.set_description("Logging...")
        if n_collects % train_cfg.log_every == 0:
            info_collect = {}
            if trajsaver is not None:
                info_collect = trajsaver.get_stats(train_cfg.log_every)

            if "RewSum" in info_collect:
                pbar.set_postfix(
                    RewSum=info_collect["RewSum"], TrajLen=info_collect["TrajLen"]
                )

            # Remove anything from info_loss that starts with nolog.
            info_loss = {
                k: v for k, v in info_loss.items() if not k.startswith("nolog/")
            }

            infos = info_loss
            log_dict = {f"Train/{k}": v for k, v in infos.items()}
            log_dict = log_dict | {f"Collect/{k}": v for k, v in info_collect.items()}
            s_since_start = time.time() - start_time
            log_dict = log_dict | {
                "n_collect": n_collects,
                "n_steps": n_steps,
                "steps_per_second": n_steps / s_since_start,
            }
            log_dict_tb(writer, log_dict, global_step=n_collects)

        # 8: Maybe save model.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
            ckpt_manager.save_ez(n_collects, {"alg": ppo})

            if run_cfg.save_all:
                # Also save recent x0s.
                b_x0 = ppo.task.leaf_to_minstate(
                    leaf_stack(trajsaver.all_x0s(), axis=0, which=np)
                )
                # Save b_x0 as pkl.
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)

        # 9: Maybe eval.
        if (n_collects % train_cfg.eval_every == 0) or final_iter:
            pbar.set_description("Eval...")
            rollouts_eval = []
            for collector_eval in tqdm.tqdm(collector_evals):
                rollout_trunc = split_trajs(jax2np(ppo.collect_eval(collector_eval)[0]))
                rollouts_eval.append(rollout_trunc)

            pbar.set_description("Eval Det...")
            rollout_eval_det = split_trajs(
                jax2np(ppo.collect_eval_det(collector_evals[0])[0])
            )

            # Visualize the most recent train and eval trajectories.
            extra = dict(farr=farr_buf)
            props = EvalProps(
                writer,
                ppo,
                trajsaver,
                rollouts_eval,
                rollout_eval_det,
                eval_region_info,
                extra,
            )
            for cb in eval_cbs:
                pbar.set_description("Eval Plot... ({})".format(str(cb)))
                cb(n_collects, run_cfg, props)

            del rollouts_eval
            del rollout_eval_det

        if (train_cfg.fastcb_every > 0) and (n_collects % train_cfg.fastcb_every == 0):
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, trajsaver, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)

    # Save the final model.
    ckpt_manager.wait_until_finished()
    time.sleep(10)
