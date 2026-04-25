import time
from typing import Callable

import jax
import jax.random as jr
import numpy as np
import tqdm
from attrs import asdict, define
from flax import nnx
from og.ckpt_utils import get_ckpt_manager
from og.jax_utils import jax2np
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, make_collector_evals
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector
from fge.core.bits.collector_savemem import collect_eval_savemem
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class PPOVDSCfg:
    ppo: SumPPO.Cfg
    vds: VDSGAE.Cfg

    def asdict(self):
        return asdict(self)


def train_ppo_vds(
    run_cfg: RunCfg[PPOVDSCfg],
    seed,
    fast_cbs: list[Callable] = [],
    eval_cbs: list[Callable] = [],
):
    train_cfg = run_cfg.train_cfg
    writer = SummaryWriter(run_cfg.paths.run_dir)

    env = make_task(run_cfg.task_cfg)

    rngs = nnx.Rngs(seed)

    gpu_device = jax.devices("gpu")[0]

    obs = env.get_dummy_obs()
    ppo = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo = jax.device_put(ppo, gpu_device)

    vds = VDSGAE.create(rngs(), obs, env, run_cfg.algo_cfg.vds)
    vds = jax.device_put(vds, gpu_device)

    collector, _ = Collector.create(rngs(), env, run_cfg.col_cfg)
    collector = jax.device_put(collector, gpu_device)

    # Get the eval collectors.
    collector_evals, eval_region_info = make_collector_evals(env, run_cfg.eval_cfg.n_seeds)
    collector_evals = jax.device_put(collector_evals, gpu_device)

    # Setup ckpting.
    ckpt_cfgs = {
        "seed": seed,
        "task_cfg": run_cfg.task_cfg,
        "alg_cfg": run_cfg.algo_cfg,
    }
    ckpt_dir = run_cfg.paths.models_dir
    ckpt_manager = get_ckpt_manager(ckpt_dir, ["alg"], max_to_keep=None, step_format_fixed_length=8)
    # Save the config.
    ckpt_manager.save_config(ckpt_cfgs)

    if isinstance(env, ToyLevelsJax):
        save_full_traj = True
    else:
        save_full_traj = False

    if save_full_traj:
        trajsaver = TrajSaver(save_full_traj=save_full_traj)
    else:
        trajsaver = FastTrajSaver()

    n_steps_per_collect = run_cfg.col_cfg.n_envs * run_cfg.algo_cfg.ppo.train_cfg.rollout_T

    # Logging
    pbar = tqdm.trange(train_cfg.n_steps)
    adv_reset_fn_key = jr.split(rngs(), train_cfg.n_steps)[0]
    for n_collects in pbar:
        # 1: Collect rollout.
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        vds_reset_fn = vds.get_reset_fn(ppo.task)
        collector, rollout, _ = ppo.collect_with_fn(collector, vds_reset_fn)
        jax.copy_to_host_async(rollout)

        # 2: Update PPO.
        pbar.set_description("Update PPO...")
        ppo, info_loss = ppo.update(rollout)

        # # breakpoint if there are any NaNs in the loss or in the ppo weights.
        # for k, v in info_loss.items():
        #     if np.isnan(v).any():
        #         logger.critical("NaN in key {} = {}".format(k, v))
        #         ipdb.set_trace()
        #
        # for path, v in jtu.tree_leaves_with_path(ppo):
        #     if jnp.isnan(v).any():
        #         logger.critical("NaN in path {} = {}".format(path, v))
        #         ipdb.set_trace()

        # 3: Update VDS.
        vds, info_vds = vds.update(rollout)

        # 4: Save the trajs.
        pbar.set_description("Add rollout (trajsaver)...")
        # logger.debug("trajsaver.add_rollout")
        rollout = jax2np(rollout)
        trajsaver.add_rollout(rollout)

        # 5: Maybe log.
        if n_collects % train_cfg.log_every == 0:
            info_collect = trajsaver.get_stats(train_cfg.log_every)

            if "RewSum" in info_collect:
                pbar.set_postfix(RewSum=info_collect["RewSum"], TrajLen=info_collect["TrajLen"])

            # Remove anything from info_loss that starts with nolog.
            info_loss = {k: v for k, v in info_loss.items() if not k.startswith("nolog/")}

            log_dict = {f"Train/{k}": v for k, v in (info_loss | info_vds).items()}
            log_dict = log_dict | {f"Collect/{k}": v for k, v in info_collect.items()}
            log_dict = log_dict | {
                "n_collect": n_collects,
                "n_steps": n_collects * n_steps_per_collect,
            }
            log_dict_tb(writer, log_dict, global_step=n_collects)

        # 6: Maybe save model.
        if n_collects % train_cfg.save_every == 0:
            ckpt_manager.save_ez(n_collects, {"alg": ppo})

            if run_cfg.save_all:
                # Also save recent x0s.
                b_x0 = ppo.task.leaf_to_minstate(leaf_stack(trajsaver.all_x0s(), axis=0, which=np))
                # Save b_x0 as pkl.
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)

        # 5: Maybe eval.
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

            # Visualize the most recent train and eval trajectories.
            extra = dict(vds=vds)
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

        if (train_cfg.fastcb_every > 0) and (n_collects % train_cfg.fastcb_every == 0):
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, trajsaver, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)

    # Save the final model.
    ckpt_manager.save_ez(n_collects, {"alg": ppo})
    ckpt_manager.wait_until_finished()
    time.sleep(10)
