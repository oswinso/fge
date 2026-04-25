import functools as ft
import time
from enum import IntEnum
from typing import Callable

import jax
import jax.lax as lax
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
from og.tree_utils import tree_copy
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import X0Saver
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, make_collector_evals
from fge.core.algos.traj_utils import split_trajs
from fge.core.bits.collector import Collector
from fge.core.bits.level_sampler import LevelSampler, Sampler
from fge.core.bits.ppo_core import SumPPO
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import JaxTask, TimedState


class SampleSource(IntEnum):
    DR = 0
    REPLAY = 1
    MUTATE = 2


@define
class AccelCfg:
    # Taken from JaxUED
    level_buffer_capacity: int = 4000
    replay_prob: float = 0.8
    staleness_coeff: float = 0.3
    temperature: float = 0.3
    topk_k: int = 4
    minimum_fill_ratio: float = 0.5
    prioritization: str = "rank"
    buffer_duplicate_check: bool = False

    # For MutateFn
    mutate_scale: float = 0.04


@define
class PPOAccelCfg:
    ppo: SumPPO.Cfg
    accel: AccelCfg

    def asdict(self):
        return asdict(self)


class ConstantResetFn(struct.PyTreeNode):
    b_x0: TimedState

    @classmethod
    def create(cls, b_x0: TimedState) -> "ConstantResetFn":
        return cls(b_x0=b_x0)

    def __call__(self, b_key: PRNGKey, num: int) -> TimedState:
        return self.b_x0


class MutateFn(struct.PyTreeNode):
    cfg: AccelCfg = struct.field(pytree_node=False)
    env: JaxTask = struct.field(pytree_node=False)

    @classmethod
    def create(cls, config: AccelCfg, env) -> "MutateFn":
        return cls(cfg=config, env=env)

    @jax.jit
    def __call__(self, key: PRNGKey, x0: TimedState) -> TimedState:
        # Convert from state back to [-1, 1] box.
        uniform = self.env.box_from_reset(x0)

        # Add noise to mutate it.
        noise = self.cfg.mutate_scale * jr.normal(key, shape=uniform.shape)
        uniform_mutated = uniform + noise

        # Make sure the mutated state is still within the box bounds.
        uniform_mutated = jnp.clip(uniform_mutated, -1.0, 1.0)

        # Convert back to state.
        return self.env.reset_from_box(uniform_mutated)


def train_ppo_accel(
    total_env_steps: int,
    run_cfg: RunCfg[PPOAccelCfg],
    seed,
    fast_cbs: list[Callable] = [],
    eval_cbs: list[Callable] = [],
):
    start_time = time.time()
    train_cfg = run_cfg.train_cfg
    writer = SummaryWriter(run_cfg.paths.run_dir)

    env = make_task(run_cfg.task_cfg)

    rngs = nnx.Rngs(seed)

    gpu_device = jax.devices("gpu")[0]

    obs = env.get_dummy_obs()
    ppo = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo = jax.device_put(ppo, gpu_device)

    collector, _ = Collector.create(rngs(), env, run_cfg.col_cfg)
    collector = jax.device_put(collector, gpu_device)

    # Create evaluation collectors and move them to the GPU.
    collector_evals, eval_region_info = make_collector_evals(env, run_cfg.eval_cfg.n_seeds)
    collector_evals = [
        jax.device_put(collector_eval, device=jax.devices("gpu")[0]) for collector_eval in collector_evals
    ]

    # Setup checkpointing for saving models and configurations.
    ckpt_cfgs = {
        "seed": seed,
        "task_cfg": run_cfg.task_cfg,
        "alg_cfg": run_cfg.algo_cfg,
        "col_cfg": run_cfg.col_cfg,
        "run_cfg": run_cfg,
    }
    ckpt_dir = run_cfg.paths.models_dir
    ckpt_manager = get_ckpt_manager(ckpt_dir, ["alg"], max_to_keep=None, step_format_fixed_length=8)
    ckpt_manager.save_config(ckpt_cfgs)

    # Determine whether to save full trajectories based on the environment type.
    save_full_traj = False
    # if isinstance(env, ToyLevelsJax):
    #     save_full_traj = True
    # else:
    #     save_full_traj = False

    # Initialize the trajectory saver.
    # if save_full_traj:
    #     trajsaver = TrajSaver(save_full_traj=save_full_traj)
    # else:
    #     trajsaver = FastTrajSaver()
    # trajsaver = None
    trajsaver = X0Saver()

    # Calculate the number of steps per collect based on rollout length and number of environments.
    rollout_T = run_cfg.algo_cfg.ppo.train_cfg.rollout_T
    n_steps_per_collect = run_cfg.col_cfg.n_envs * rollout_T

    config: AccelCfg = run_cfg.algo_cfg.accel
    level_sampler = LevelSampler(
        capacity=config.level_buffer_capacity,
        replay_prob=config.replay_prob,
        staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio,
        prioritization=config.prioritization,
        prioritization_params={"temperature": config.temperature, "k": config.topk_k},
        duplicate_check=config.buffer_duplicate_check,
    )
    state = env.reset(jr.PRNGKey(0))
    state_leaf_ph = env.state_to_leaf(state)
    sampler = level_sampler.initialize(state_leaf_ph, {"max_return": -np.inf})

    source = SampleSource.DR
    rng = np.random.default_rng(seed - 1)

    mutate_fn = MutateFn.create(run_cfg.algo_cfg.accel, env)
    vmap_mutate = jax.jit(jax.vmap(mutate_fn.__call__))

    key_replay = rngs()

    # Mimic n_collects % save_every == 0, but instead of n_collects / n_steps,
    # we look at env_steps_taken / total_env_steps.
    def get_fracs(every: int, total: int):
        # e.g., every = 2, total = 10 => [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        return list(np.arange(0, total + 1, every) / total)

    def reached_frac(frac: float, fracs: list[float]) -> bool:
        has_reached = frac >= fracs[0]
        # Modify fracs in place to remove all elements that are less than or equal to frac.
        while len(fracs) > 0 and frac >= fracs[0]:
            fracs.pop(0)

        return has_reached

    save_model_fracs = get_fracs(train_cfg.save_every, train_cfg.n_steps)
    eval_fracs = get_fracs(train_cfg.eval_every, train_cfg.n_steps)
    fastcb_fracs = get_fracs(train_cfg.fastcb_every, train_cfg.n_steps)

    # Main training loop. Instead of using a fixed number of iterations though, we stop when we've performed the same
    # number of PPO iterations.
    # pbar = tqdm.trange(train_cfg.n_steps)
    env_steps_taken = 0
    pbar = tqdm.tqdm(total=total_env_steps)
    for n_collects in range(train_cfg.n_steps):
        # Check if this is the final iteration.
        enough_env_steps = env_steps_taken >= total_env_steps
        final_iter = (n_collects >= train_cfg.n_steps - 1) or enough_env_steps

        current_frac = env_steps_taken / total_env_steps

        if source == SampleSource.DR:
            use_replay = level_sampler.sample_replay_decision_np(sampler, rng)
            branch = SampleSource.REPLAY if use_replay else SampleSource.DR
        elif source == SampleSource.REPLAY:
            branch = SampleSource.MUTATE
        else:
            raise ValueError(f"Unknown source: {source}")

        # Mimic the code structure of JaxUED.
        # Here, we will do the rollout, sampler update and ppo update.
        if branch == SampleSource.DR:
            pbar.set_description("DR Branch...")
            collector, sampler, rollout = accel_branch_dr(ppo, collector, sampler, level_sampler)

            source = SampleSource.DR
        elif branch == SampleSource.REPLAY:
            pbar.set_description("Replay Branch...")
            key = jr.fold_in(key_replay, n_collects)
            ppo, collector, sampler, rollout, b_x0_replay, info_loss = accel_branch_replay(
                key, ppo, collector, sampler, level_sampler
            )
            source = SampleSource.REPLAY

            # rollout = jax2np(rollout)
            # pbar.set_description("Adding rollout to trajsaver...")
            # trajsaver.add_rollout(rollout)

            # Add b_x0_replay (treeleaves) to the trajsaver.
            b_x0_replay_leaf = jax2np(env.state_to_leaf(b_x0_replay))
            trajsaver.add_x0s(b_x0_replay_leaf)

            # Increment the number of env steps we took to update PPO.
            n_env_steps = collector.cfg.n_envs * ppo.rollout_T
            pbar.update(n_env_steps)
            env_steps_taken += n_env_steps

        elif branch == SampleSource.MUTATE:
            pbar.set_description("Mutate Branch...")
            key = jr.fold_in(key_replay, n_collects)
            b_key = jr.split(key, collector.cfg.n_envs)
            b_x0_mutated = vmap_mutate(b_key, b_x0_replay)
            collector, sampler, rollout = accel_branch_mutate(ppo, collector, sampler, level_sampler, b_x0_mutated)

            # In mutate, treat it like a DR.
            source = SampleSource.DR
        else:
            raise ValueError(f"Unknown branch: {branch}")

        # 6: Maybe save model.
        if reached_frac(current_frac, save_model_fracs) or final_iter:
            ckpt_manager.save_ez(n_collects, {"alg": ppo})

        # 7: Maybe eval.
        if reached_frac(current_frac, eval_fracs) or final_iter:
            pbar.set_description("Eval...")
            rollouts_eval = []
            for collector_eval in tqdm.tqdm(collector_evals):
                rollout_trunc = split_trajs(jax2np(ppo.collect_eval(collector_eval)[0]))
                rollouts_eval.append(rollout_trunc)

            pbar.set_description("Eval Det...")
            rollout_eval_det = split_trajs(jax2np(ppo.collect_eval_det(collector_evals[0])[0]))

            # Visualize the most recent train and eval trajectories.
            extra = dict(level_sampler=level_sampler, sampler=sampler)
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

        if (train_cfg.fastcb_every > 0) and reached_frac(current_frac, fastcb_fracs) or final_iter:
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, trajsaver, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)

        if final_iter:
            break

    # Save the final model.
    ckpt_manager.wait_until_finished()
    time.sleep(10)


def accumulate_rollout_stats(T_term, T_metric, *, time_average: bool):
    def body(carry, inp):
        val_sum, val_max, val_accum, step_count, episode_count = carry
        term, val_now = inp

        val_accum = jtu.tree_map(lambda x, y: x + y, val_accum, val_now)
        step_count += 1

        if time_average:
            # Compute the current time averaged value.
            val = jtu.tree_map(lambda x: x / step_count, val_accum)
        else:
            val = val_accum

        # Sum the episode averaged values. Only FULL episodes (so we only add if we terminate)
        val_sum = jtu.tree_map(lambda x, y: x + term * y, val_sum, val)
        val_max = jtu.tree_map(lambda x, y: (1 - term) * x + term * jnp.maximum(x, y), val_max, val)

        # If we terminate, then it's a new episode.
        episode_count += term

        # Only keep the accumulated values for the next step if we don't terminate.
        val_accum = jtu.tree_map(lambda x: (1 - term) * x, val_accum)
        # Same for step count.
        step_count = (1 - term) * step_count

        return (val_sum, val_max, val_accum, step_count, episode_count), None

    (T,) = T_term.shape
    metric_zero = jtu.tree_map(lambda T_x: jnp.zeros_like(T_x[0]), T_metric)
    carry0 = (metric_zero, metric_zero, metric_zero, 0, 0)
    (val_sum, val_max, _, _, episode_count), _ = lax.scan(body, carry0, (T_term, T_metric))
    val_mean = jtu.tree_map(lambda x: x / jnp.maximum(episode_count, 1), val_sum)

    return val_mean, val_max, episode_count


@ft.partial(jax.jit, static_argnames=("level_sampler",))
def accel_branch_dr(ppo: SumPPO, collector: Collector, sampler: Sampler, level_sampler: LevelSampler):
    logger.debug("Jitting DR branch...")

    # Reset all envs using the base distribution.
    collector = collector.reset_all()

    # Store these initial states. If we reset during the rollout, we want to reset to the EXACT SAME INITIAL STATE.
    b_x0 = tree_copy(collector.collect_state.state)
    reset_fn = ConstantResetFn.create(b_x0)

    # b_x0: ToyLevelsJax.State
    # b_px = b_x0.pos[:, 0]
    # n_easy = jnp.sum(b_px <= 35)
    # logger.info("n_easy: {}".format(n_easy))

    # Collect FULL trajectories.
    collector, rollout, _ = ppo.collect_with_fn(collector, reset_fn)

    # Compute GAE.
    bT_Al, bT_Ql = ppo.compute_bT_A_Q(rollout)

    # Compute the score from the GAE using PVL
    bT_term = rollout.T_term
    bT_Al_positive = jnp.maximum(bT_Al, 0.0)
    b_scores, _, b_episode_count = jax.vmap(ft.partial(accumulate_rollout_stats, time_average=True))(
        bT_term, bT_Al_positive
    )

    incomplete_value = -jnp.inf
    b_scores = jnp.where(b_episode_count > 0, b_scores, incomplete_value)

    # Minify b_x0 to insert into the sampler.
    b_x0_leaf = ppo.task.state_to_leaf(b_x0)
    sampler, _ = level_sampler.insert_batch(sampler, b_x0_leaf, b_scores)

    # import ipdb
    # ipdb.set_trace()

    # We don't update PPO in this branch, this is the robust PLR variant.
    return collector, sampler, rollout


@ft.partial(jax.jit, static_argnames=("level_sampler",))
def accel_branch_replay(key: PRNGKey, ppo: SumPPO, collector: Collector, sampler: Sampler, level_sampler: LevelSampler):
    logger.debug("Jitting replay branch...")

    # Sample a batch of states from the level sampler.
    # logger.debug("Sampling replay levels...")
    sampler, (b_level_inds, b_x0_leaf) = level_sampler.sample_replay_levels(sampler, key, collector.cfg.n_envs)

    # logger.debug("Sampling replay levels... Done!")
    b_x0 = jax.vmap(ppo.task.leaf_to_state)(b_x0_leaf)

    # Reset the collector with the sampled states.
    collector = collector.reset_with_state(b_x0)

    # Store these initial states. If we reset during the rollout, we want to reset to the EXACT SAME INITIAL STATE.
    b_x0 = tree_copy(collector.collect_state.state)
    reset_fn = ConstantResetFn.create(b_x0)

    # Collect FULL trajectories.
    collector, rollout, _ = ppo.collect_with_fn(collector, reset_fn)

    # Compute GAE.
    bT_Al, bT_Ql = ppo.compute_bT_A_Q(rollout)

    # Compute the score from the GAE using PVL
    bT_term = rollout.T_term
    bT_Al_positive = jnp.maximum(bT_Al, 0.0)
    b_scores, _, b_episode_count = jax.vmap(ft.partial(accumulate_rollout_stats, time_average=True))(
        bT_term, bT_Al_positive
    )

    incomplete_value = -jnp.inf
    b_scores = jnp.where(b_episode_count > 0, b_scores, incomplete_value)

    # Minify b_x0 to insert into the sampler.
    logger.debug("Updating batch...")
    sampler = level_sampler.update_batch(sampler, b_level_inds, b_scores)
    logger.debug("Updating batch... Done!")

    # This is the only branch we update PPO.
    logger.debug("Updating PPO...")
    ppo, info_loss = ppo.update(rollout)
    logger.debug("Updating PPO... Done!")

    return ppo, collector, sampler, rollout, b_x0, info_loss


@ft.partial(jax.jit, static_argnames=("level_sampler",))
def accel_branch_mutate(
    ppo: SumPPO,
    collector: Collector,
    sampler: Sampler,
    level_sampler: LevelSampler,
    b_x0_mutated: TimedState,
):
    logger.debug("Jitting mutate branch...")

    # Reset all envs to the mutated states.
    collector = collector.reset_with_state(b_x0_mutated)

    # Store these initial states. If we reset during the rollout, we want to reset to the EXACT SAME INITIAL STATE.
    b_x0 = tree_copy(collector.collect_state.state)
    reset_fn = ConstantResetFn.create(b_x0)

    # Collect FULL trajectories.
    collector, rollout, _ = ppo.collect_with_fn(collector, reset_fn)

    # Compute GAE.
    bT_Al, bT_Ql = ppo.compute_bT_A_Q(rollout)

    # Compute the score from the GAE using PVL
    bT_term = rollout.T_term
    bT_Al_positive = jnp.maximum(bT_Al, 0.0)
    b_scores, _, b_episode_count = jax.vmap(ft.partial(accumulate_rollout_stats, time_average=True))(
        bT_term, bT_Al_positive
    )

    incomplete_value = -jnp.inf
    b_scores = jnp.where(b_episode_count > 0, b_scores, incomplete_value)

    # Minify b_x0 to insert into the sampler.
    b_x0_leaf = ppo.task.state_to_leaf(b_x0)
    sampler, _ = level_sampler.insert_batch(sampler, b_x0_leaf, b_scores)

    # We don't update PPO in this branch, this is the robust PLR variant.
    return collector, sampler, rollout
