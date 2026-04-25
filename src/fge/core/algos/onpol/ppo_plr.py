import os
import time
from typing import Callable

import jax
import numpy as np
import tqdm
from attr import asdict
from attrs import define
from flax import nnx
from loguru import logger
from og.ckpt_utils import get_ckpt_manager
from og.jax_utils import jax2np
from pyinstrument import Profiler
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, make_collector_evals
from fge.core.algos.plr_sampler import PLRSampler
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector
from fge.core.bits.collector_savemem import collect_eval_savemem
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class PPOPLRCfg:
    ppo: SumPPO.Cfg
    plr: PLRSampler.Cfg

    def asdict(self):
        return asdict(self)


def train_ppo_plr(
    run_cfg: RunCfg[PPOPLRCfg],
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

    reset_id_provider = jax.device_put(ResetIDProvider.create(), gpu_device)

    collector, reset_id_provider = Collector.create(rngs(), env, run_cfg.col_cfg, id_provider=reset_id_provider)
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
    ckpt_manager.save_config(ckpt_cfgs)

    no_trajsaver = False
    if isinstance(env, F16AvoidJax):
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

    plr_cfg = run_cfg.algo_cfg.plr
    buf = PLRSampler(env, plr_cfg, reset_id_provider)

    n_steps_per_collect = run_cfg.col_cfg.n_envs * run_cfg.algo_cfg.ppo.train_cfg.rollout_T

    DO_PROFILE = os.environ.get("PROFILE", "0") == "1"
    if DO_PROFILE:
        profiler = Profiler(interval=0.0001)

    pbar = tqdm.trange(train_cfg.n_steps)
    for n_collects in pbar:
        final_iter = n_collects >= train_cfg.n_steps - 1

        if DO_PROFILE:
            if n_collects == 5:
                logger.debug("Starting profiler")
                profiler.start()

            if n_collects == int(10):
                profiler.stop()
                logger.debug("Stopping profiler, writing to file...")
                html_path = run_cfg.paths.run_dir / "profiler.html"
                profiler.write_html(html_path)
                logger.debug("Stopping profiler, writing to file... Done!")
                profiler.reset()
                exit(0)

        # 1: Collect rollout.
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        collector, buf.buffer, rollout, _ = ppo.collect_with_buf(collector, buf.buffer)
        jax.copy_to_host_async(rollout)

        # 2: Process rollout using PLR.
        pbar.set_description("PLR...")
        info_plr = buf.process_rollout(ppo, rollout)

        # 3: Train PPO.
        pbar.set_description("Update PPO...")
        ppo, info_loss = ppo.update(rollout)

        # 4: Save trajs.
        if trajsaver is not None:
            pbar.set_description("Add rollout...")
            rollout_cpu = jax2np(rollout)
            trajsaver.add_rollout(rollout_cpu)

        # 5: Maybe log.
        pbar.set_description("Logging...")
        if n_collects % train_cfg.log_every == 0:
            info_collect = {}
            if trajsaver is not None:
                info_collect = trajsaver.get_stats(train_cfg.log_every)

            if "RewSum" in info_collect:
                pbar.set_postfix(RewSum=info_collect["RewSum"], TrajLen=info_collect["TrajLen"])

            # Remove anything from info_loss that starts with nolog.
            info_loss = {k: v for k, v in info_loss.items() if not k.startswith("nolog/")}

            infos = info_loss | info_plr
            log_dict = {f"Train/{k}": v for k, v in infos.items()}
            log_dict = log_dict | {f"Collect/{k}": v for k, v in info_collect.items()}
            n_steps = n_collects * n_steps_per_collect
            s_since_start = time.time() - start_time
            log_dict = log_dict | {
                "n_collect": n_collects,
                "n_steps": n_collects * n_steps_per_collect,
                "steps_per_second": n_steps / s_since_start,
            }
            log_dict_tb(writer, log_dict, global_step=n_collects)

        # 6: Maybe save model.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
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
        if (n_collects % train_cfg.eval_every == 0) or final_iter:
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
            rollout_eval_det = split_trajs(
                jax2np(collect_eval_savemem(ppo, collector_evals[0], small_T, rng=False)[0])
            )

            # Visualize the most recent train and eval trajectories.
            extra = dict(plr=buf)
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
