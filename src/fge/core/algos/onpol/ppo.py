import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import jax
import jax.random as jr
import numpy as np
import tqdm
import wandb
from attrs import asdict, define
from jax_array_info import pretty_memory_stats, print_array_stats, sharding_info, sharding_vis, simple_array_info
from loguru import logger
from og.ckpt_utils import get_ckpt_manager
from og.jax_utils import jax2np
from og.wandb_utils import flatten_dict, reorder_wandb_name
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector
from fge.core.bits.collector_savemem import collect_eval_savemem
from fge.core.bits.ppo_core import SumPPO
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.dict_utils import any_to_dict
from fge.core.utils.path_utils import get_benchmark_dir, get_runs_dir
from fge.core.utils.tb import log_dict_tb


@define
class TrainCfg:
    n_steps: int
    """Number of times we do a rollout"""

    log_every: int
    """How often to log the train info, as a function of n_steps"""

    fastcb_every: int

    eval_every: int
    """How often to evaluate, as a function of n_steps"""

    save_every: int
    """How often to save the model, as a function of n_steps. Don't save if negative."""

    def asdict(self):
        return asdict(self)


@define
class EvalCfg:
    n_seeds: int

    def asdict(self):
        return asdict(self)


@define
class EvalProps:
    writer: SummaryWriter
    ppo: SumPPO
    trajsaver: TrajSaver | None
    rollouts_eval: list[list[Collector.Rollout]]
    rollout_eval_det: list[Collector.Rollout]
    eval_region_info: list[tuple[str, int]]

    @property
    def task(self):
        return self.ppo.task

    extra: dict = {}


@define
class RunPaths:
    run_dir: pathlib.Path
    models_dir: pathlib.Path
    train_plots: pathlib.Path
    eval_plots: pathlib.Path
    data_dir: pathlib.Path

    @staticmethod
    def setup(base_name: str, run_id: str):
        runs_dir = get_runs_dir()
        parent_dir = runs_dir / "{}".format(base_name)

        run_dir = parent_dir / "{}".format(run_id)
        models_dir = run_dir / "models"
        tplots_dir = run_dir / "train_plots"
        eplots_dir = run_dir / "eval_plots"
        data_dir = run_dir / "data"

        for d in [models_dir, tplots_dir, eplots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return RunPaths(run_dir, models_dir, tplots_dir, eplots_dir, data_dir)

    @staticmethod
    def setup_bk(task_name, alg_name, seed):
        run_dir = get_benchmark_dir() / task_name / alg_name / str(seed)
        if run_dir.exists():
            shutil.rmtree(run_dir)

        models_dir = run_dir / "models"
        tplots_dir = run_dir / "train_plots"
        eplots_dir = run_dir / "eval_plots"
        data_dir = run_dir / "data"

        for d in [models_dir, tplots_dir, eplots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return RunPaths(run_dir, models_dir, tplots_dir, eplots_dir, data_dir)


AlgoCfg_ = TypeVar("AlgoCfg_")


@dataclass
class RunCfg(Generic[AlgoCfg_]):
    algo_cfg: AlgoCfg_
    task_cfg: dataclass
    train_cfg: TrainCfg
    col_cfg: Collector.Cfg
    eval_cfg: dataclass

    paths: RunPaths

    # Wandb
    use_wandb: bool = False
    task_name: str = "DubinsJax"

    save_all: bool = False
    """Save all NN and save the recent x0s every save ckpt."""

    @staticmethod
    def setup(
        seed: int,
        algo_cfg: SumPPO.Cfg,
        task_cfg: dataclass,
        train_cfg: TrainCfg,
        col_cfg: Collector.Cfg,
        eval_cfg: EvalCfg,
        task_name: str,
        use_wandb: bool,
        wandb_name: str | None,
        tags: list[str] | None = [],
        benchmark: bool = False,
        alg_name: str = None,
    ) -> "RunCfg":
        base_name = task_name

        algo_cfg_cls = type(algo_cfg).__name__
        # If algo_cfg_cls ends in Cfg, remove it.
        if algo_cfg_cls.endswith("Cfg"):
            algo_cfg_cls = algo_cfg_cls[:-3]

        if use_wandb:
            wandb_cfg = {
                "seed": seed,
                "alg": algo_cfg.asdict(),
                "train": train_cfg.asdict(),
                "task": any_to_dict(task_cfg),
                "col": col_cfg.asdict(),
                "eval_cfg": eval_cfg.asdict(),
                "alg_cfg_cls": algo_cfg_cls,
                "task_name": task_name,
                "benchmark": benchmark,
            }
            wandb_cfg = flatten_dict(wandb_cfg, separator=".")
            tags = [task_name] + tags
            wandb_dir = get_runs_dir()
            run = wandb.init(
                dir=wandb_dir,
                project="resetrl",
                config=wandb_cfg,
                tags=tags,
                save_code=True,
                sync_tensorboard=True,
            )

            if benchmark:
                paths = RunPaths.setup_bk(task_name, alg_name, seed)
                wandb.run.name = f"{alg_name}_{seed}"
            else:
                wandb_name = reorder_wandb_name(wandb_name)
                run_id = wandb_name

                logger.debug("wandb mode: {}".format(wandb.run.settings.mode))
                if wandb.run.settings.mode == "offline":
                    logger.warning("wandb running in offline mode, adding suffix to make sure the run_id is unique.")
                    runs_dir = get_runs_dir()
                    parent_dir = runs_dir / "{}".format(base_name)
                    run_dir = parent_dir / "{}".format(run_id)

                    if run_dir.exists():
                        logger.warning("Run dir already exists, adding seed suffix to run_id to make it unique.")
                        run_id = "{}_{}".format(run_id, seed)

                        wandb_name = run_id
                        wandb.run.name = run_id

                # If offline, make sure its a unique run.
                paths = RunPaths.setup(base_name, run_id)

            wandb.tensorboard.unpatch()
            wandb.tensorboard.patch(root_logdir=str(paths.run_dir.absolute()))
        else:
            run_id = "test_{}_{}".format(algo_cfg_cls, wandb_name)
            paths = RunPaths.setup(base_name, run_id)

        return RunCfg(
            algo_cfg=algo_cfg,
            task_cfg=task_cfg,
            train_cfg=train_cfg,
            col_cfg=col_cfg,
            eval_cfg=eval_cfg,
            paths=paths,
            use_wandb=use_wandb,
            task_name=task_name,
        )


def make_collector_evals(env: ToyLevelsJax, n_seeds: int) -> tuple[list[Collector], list[tuple[str, int]]]:
    key0 = jr.PRNGKey(12345)

    # Retrieve evaluation states, the number of evaluation states, and region information from the environment.
    b_state0, n_eval_states, region_info = env.get_eval_states()

    # Create a configuration for the collectors with the number of evaluation environments.
    cfg = Collector.Cfg(n_envs=n_eval_states)

    # Create the collectors.
    collectors = []
    for ii in range(n_seeds):
        # Generate a unique random key for each collector.
        key = jr.fold_in(key0, 1 + ii)
        collector, _ = Collector.create(key, env, cfg)

        # Update the collector's state with the evaluation states.
        collect_state_new = collector.collect_state._replace(state=b_state0)
        collector = collector.replace(collect_state=collect_state_new)
        collectors.append(collector)

    return collectors, region_info


@define
class PPOOnlyCfg:
    ppo: SumPPO.Cfg

    def asdict(self):
        return asdict(self)


def train_ppo(
    run_cfg: RunCfg[PPOOnlyCfg],
    seed,
    fast_cbs: list[Callable] = [],
    eval_cbs: list[Callable] = [],
):
    train_cfg = run_cfg.train_cfg
    writer = SummaryWriter(run_cfg.paths.run_dir)

    # Create the environment based on the task configuration.
    env = make_task(run_cfg.task_cfg)

    # Initialize random keys for the algorithm and collector.
    key_base = jr.PRNGKey(seed)
    key_alg, key_col = jr.split(key_base)

    # Move the PPO agent and collector to the GPU.
    gpu_device = jax.devices("gpu")[0]
    ppo: SumPPO = jax.device_put(SumPPO.create(key_alg, env, run_cfg.algo_cfg.ppo), gpu_device)
    collector, _ = Collector.create(key_col, env, run_cfg.col_cfg)
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
    if isinstance(env, ToyLevelsJax):
        save_full_traj = True
    else:
        save_full_traj = False

    # Initialize the trajectory saver.
    if save_full_traj:
        trajsaver = TrajSaver(save_full_traj=save_full_traj)
    else:
        trajsaver = FastTrajSaver()

    # Calculate the number of steps per collect based on rollout length and number of environments.
    rollout_T = run_cfg.algo_cfg.ppo.train_cfg.rollout_T
    n_steps_per_collect = run_cfg.col_cfg.n_envs * rollout_T

    # Main training loop.
    pbar = tqdm.trange(train_cfg.n_steps)
    for n_collects in pbar:
        print_dbg = (n_collects <= 2) or (n_collects % 200 == 0)

        # Check if this is the final iteration.
        final_iter = n_collects >= train_cfg.n_steps - 1

        # 1: Collect rollout data.
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        collector, rollout, _ = ppo.collect(collector)
        jax.copy_to_host_async(rollout)

        if print_dbg:
            logger.debug("After rollout")
            pretty_memory_stats(gpu_device)

        # 2: Update the PPO agent using the collected rollout.
        pbar.set_description("Update PPO...")
        update_data = ppo.get_data_from_rollout(rollout)

        if print_dbg:
            logger.debug("get_data_from_rollout")
            pretty_memory_stats(gpu_device)

        rollout = jax2np(rollout)

        if print_dbg:
            logger.debug("After jax2np")
            pretty_memory_stats(gpu_device)

        ppo, info_loss = ppo.update(update_data)

        if print_dbg:
            logger.debug("After update")
            pretty_memory_stats(gpu_device)

        # 3: Save the collected trajectories.
        pbar.set_description("Add rollout...")
        trajsaver.add_rollout(rollout)

        # 4: Log training progress at specified intervals.
        if n_collects % train_cfg.log_every == 0:
            info_collect = trajsaver.get_stats(train_cfg.log_every)

            if "RewSum" in info_collect:
                pbar.set_postfix(RewSum=info_collect["RewSum"], TrajLen=info_collect["TrajLen"])

            # Filter out non-loggable information from the loss info.
            info_loss = {k: v for k, v in info_loss.items() if not k.startswith("nolog/")}

            # Combine and log training and collection information.
            log_dict = {f"Train/{k}": v for k, v in info_loss.items()}
            log_dict = log_dict | {f"Collect/{k}": v for k, v in info_collect.items()}
            log_dict = log_dict | {
                "n_collect": n_collects,
                "n_steps": n_collects * n_steps_per_collect,
            }
            log_dict_tb(writer, log_dict, global_step=n_collects)

        # 5: Save the model at specified intervals or at the final iteration.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
            ckpt_manager.save_ez(n_collects, {"alg": ppo})

            if run_cfg.save_all:
                # Save the initial states of recent trajectories.
                b_x0 = ppo.task.leaf_to_minstate(leaf_stack(trajsaver.all_x0s(), axis=0, which=np))
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)

        # 6: Perform evaluation at specified intervals.
        if n_collects % train_cfg.eval_every == 0:
            pbar.set_description("Eval...")
            small_T = 10
            rollouts_eval = []
            for collector_eval in tqdm.tqdm(collector_evals):
                # rollout_trunc = split_trajs(jax2np(ppo.collect_eval(collector_eval)[0]))
                rollout_trunc = split_trajs(jax2np(collect_eval_savemem(ppo, collector_eval, small_T, rng=True)[0]))
                rollouts_eval.append(rollout_trunc)

            pbar.set_description("Eval Det...")
            # rollout_eval_det = split_trajs(
            #     jax2np(ppo.collect_eval_det(collector_evals[0])[0])
            # )
            rollout_eval_det = split_trajs(jax2np(collect_eval_savemem(ppo, collector_evals[0], small_T, rng=False)[0]))

            # Visualize the most recent training and evaluation trajectories.
            props = EvalProps(
                writer,
                ppo,
                trajsaver,
                rollouts_eval,
                rollout_eval_det,
                eval_region_info,
            )
            for cb in eval_cbs:
                pbar.set_description("Eval Plot... ({})".format(str(cb)))
                cb(n_collects, run_cfg, props)

            pbar.set_description("Eval... Done!")

        # Execute fast callbacks at specified intervals.
        if (train_cfg.fastcb_every > 0) and (n_collects % train_cfg.fastcb_every == 0):
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, trajsaver, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)
            pbar.set_description("Fast Eval... Done!")

    # Save the final model and ensure all checkpointing is complete.
    ckpt_manager.wait_until_finished()
    time.sleep(10)
