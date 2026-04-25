import os
import time
from typing import Callable

import jax
import numpy as np
import tqdm
from attr import asdict
from attrs import define
from flax import nnx
from jax_array_info import pretty_memory_stats
from loguru import logger
from og.ckpt_utils import get_ckpt_manager
from og.jax_utils import jax2np, merge01
from pyinstrument import Profiler
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, make_collector_evals
from fge.core.algos.traj_utils import split_trajs
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import Collector
from fge.core.bits.collector_savemem import collect_eval_savemem
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import StateResetId
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.utils.tb import log_dict_tb


@define
class PPOFGECfg:
    ppo: SumPPO.Cfg
    nsf: NSF.Cfg
    buf: BufCustom.Cfg
    ci_classify: CIClassifier.Cfg
    pol_classify: CIClassifier.Cfg

    nsf_ci: NSF.Cfg = NSF.Cfg()

    def asdict(self):
        return asdict(self)


def train_ppo_fge(
        run_cfg: RunCfg[PPOFGECfg],
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
    nsf_obs = env.get_dummy_nsf_obs()

    ppo = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo: SumPPO = jax.device_put(ppo, gpu_device)

    nsf_ci = None
    buf_cfg = run_cfg.algo_cfg.buf
    # if buf_cfg.use_nsf_ci:
    #     nsf_ci = NSF.create(rngs(), obs, run_cfg.algo_cfg.nsf_ci)

    ci_classify = CIClassifier.create(rngs(), obs, run_cfg.algo_cfg.ci_classify, "CIClassify")
    ci_classify: CIClassifier = jax.device_put(ci_classify, gpu_device)

    pol_classify = CIClassifier.create(rngs(), obs, run_cfg.algo_cfg.pol_classify, "PolClassify")
    pol_classify: CIClassifier = jax.device_put(pol_classify, gpu_device)

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
    if run_cfg.save_all:
        item_names = ["alg", "ci_classify", "pol_classify", "buf_ci_dict"]
    else:
        item_names = ["alg"]

    ckpt_manager = get_ckpt_manager(ckpt_dir, item_names, max_to_keep=None, step_format_fixed_length=8)
    # Save the config.
    ckpt_manager.save_config(ckpt_cfgs)

    trajsaver = FastTrajSaver()

    buf = BufCustom(rngs(), env, buf_cfg, reset_id_provider)

    if buf.cfg.use_nsf_explore:
        # We only need NSF if we are using it to argmax for explore.
        nsf = NSF.create(rngs(), nsf_obs, run_cfg.algo_cfg.nsf)
        nsf = jax.device_put(nsf, gpu_device)
    else:
        nsf = None

    n_steps_per_collect = run_cfg.col_cfg.n_envs * run_cfg.algo_cfg.ppo.train_cfg.rollout_T

    DO_PROFILE = os.environ.get("PROFILE", "0") == "1"
    if DO_PROFILE:
        profiler = Profiler(interval=0.0001)

    # Do rehearsal only if n_collects >= rehearsal_only_iter.
    if buf_cfg.rehearsal_only_frac is None:
        rehearsal_only_iter = None
    else:
        rehearsal_only_iter = int((1 - buf_cfg.rehearsal_only_frac) * train_cfg.n_steps)

    # iter_start_MiB = 0
    pbar = tqdm.trange(train_cfg.n_steps)
    for n_collects in pbar:
        # print_dbg = (n_collects <= 10) or (n_collects % 200 == 0)
        print_dbg = False
        final_iter = n_collects >= train_cfg.n_steps - 1

        if DO_PROFILE:
            if n_collects == (1 * train_cfg.eval_every + 1):
                logger.debug("Starting profiler")
                profiler.start()

            if n_collects == int(1.5 * train_cfg.eval_every):
                profiler.stop()
                logger.debug("Stopping profiler, writing to file...")
                html_path = run_cfg.paths.run_dir / "profiler.html"
                profiler.write_html(html_path)
                logger.debug("Stopping profiler, writing to {}... Done!".format(html_path))
                profiler.reset()
                exit(0)

        if rehearsal_only_iter is not None and n_collects == rehearsal_only_iter:
            logger.info("Starting rehearsal only mode!")
            buf.set_rehearsal_only()

        if print_dbg:
            logger.debug("Before rollout")
            pretty_memory_stats(gpu_device)

        # 1: Collect rollout.
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        collector, buf.buffer, rollout, _ = ppo.collect_with_buf(collector, buf.buffer)

        # dbg_mem("After Rollout")
        jax.copy_to_host_async(rollout)

        if print_dbg:
            logger.debug("After rollout")
            pretty_memory_stats(gpu_device)

        # 3: Update PPO.
        update_data = ppo.get_data_from_rollout(rollout)

        if print_dbg:
            logger.debug("After rollout")
            pretty_memory_stats(gpu_device)

        if print_dbg:
            logger.debug("After jax2np")
            pretty_memory_stats(gpu_device)

        pbar.set_description("Update PPO...")
        ppo, info_loss = ppo.update(update_data)

        if print_dbg:
            logger.debug("After update")
            pretty_memory_stats(gpu_device)

        rollout_cpu = jax.device_get(rollout)

        # 4: Save the trajs.
        pbar.set_description("Add rollout (trajsaver)...")
        trajsaver.add_rollout(rollout_cpu)

        pbar.set_description("Update rollout (buffer)")
        buf.add_rollout(rollout_cpu)

        # 4.5: Update NSF
        if buf.cfg.use_nsf_explore:
            train_nsf_mode = run_cfg.algo_cfg.buf.train_nsf_mode
            if train_nsf_mode == "bT_obs_now":
                bT_obs_now = rollout.T_obs_now
                b_obs_now = merge01(bT_obs_now)
                nsf, info_nsf = nsf.update(b_obs_now)
            elif train_nsf_mode == "b_x0":
                b_x0_obs = buf.get_x0_obs_nsf()
                nsf, info_nsf = nsf.update(jax.device_put(b_x0_obs, gpu_device))
            else:
                raise ValueError("Invalid train_nsf_mode {}".format(train_nsf_mode))
        del rollout, rollout_cpu

        # Train the CI classifier. Only train if there are points in the CI.
        if buf.n_ci_buf > 0:
            pbar.set_description("Get CI Classify Data")
            b_obs, b_inci = buf.get_data_ci_classify(ci_classify.cfg.n_sample)

            pbar.set_description("Update CI Classifier")
            ci_classify, info_ci_classify = ci_classify.update(b_obs, b_inci)

        else:
            info_ci_classify = {}

        # Train on-policy classifier.
        pbar.set_description("Get Pol Classify Data")
        b_obs, b_safe = buf.get_data_polcond(pol_classify.cfg.n_sample)
        pbar.set_description("Update Pol Classify")
        pol_classify, info_pol_classify = pol_classify.update(b_obs, b_safe)

        # Update the buffer.
        pbar.set_description("Update Reset Buffer")

        info_custom, info_custom_log = buf.update_resets_lean(nsf, ci_classify, pol_classify)

        # 5: Maybe log.
        pbar.set_description("Logging...")
        if n_collects % train_cfg.log_every == 0:
            info_collect = trajsaver.get_stats(train_cfg.log_every)

            if "RewSum" in info_collect:
                pbar.set_postfix(RewSum=info_collect["RewSum"], TrajLen=info_collect["TrajLen"], n_ci=buf.n_ci_buf)

            # Remove anything from info_loss that starts with nolog.
            info_loss = {k: v for k, v in info_loss.items() if not k.startswith("nolog/")}

            infos = info_loss | info_ci_classify | info_pol_classify | info_custom_log
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

        # dbg_mem("After log")

        # 5: Maybe save model.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
            if run_cfg.save_all:
                save_dict = {
                    "alg": ppo,
                    "ci_classify": ci_classify,
                    "pol_classify": pol_classify,
                }

                # We can't save 0 length arrays....
                if int(buf.buffer.buf_ci.size) > 0:
                    # Don't save the entire thing, because mujoco state has zero length arrays in it and
                    # orbax cannot handle it...
                    buf_ci = buf.buffer.buf_ci
                    b_stateresetid: StateResetId = buf_ci.data
                    # Replace the state with minstate.
                    b_state_min = ppo.task.get_minstate(b_stateresetid.state)
                    b_stateresetid = b_stateresetid._replace(state=b_state_min)
                    save_dict["buf_ci_dict"] = {
                        "head": buf_ci.head,
                        "size": buf_ci.size,
                        "data": b_stateresetid,
                    }

                ckpt_manager.save_ez(n_collects, save_dict)

                # Also save recent x0s.
                b_x0 = ppo.task.leaf_to_minstate(leaf_stack(trajsaver.all_x0s(), axis=0, which=np))
                # Save b_x0 as pkl.
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)
            else:
                ckpt_manager.save_ez(n_collects, {"alg": ppo})

        # 6: Maybe eval.
        if (n_collects % train_cfg.eval_every == 0) or final_iter:
            pbar.set_description("Eval...")
            small_T = 10
            rollouts_eval = []
            for collector_eval in tqdm.tqdm(collector_evals):
                rollout_trunc = split_trajs(jax2np(collect_eval_savemem(ppo, collector_eval, small_T, rng=True)[0]))
                rollouts_eval.append(rollout_trunc)

            pbar.set_description("Eval Det...")
            rollout_eval_det = split_trajs(
                jax2np(collect_eval_savemem(ppo, collector_evals[0], small_T, rng=False)[0])
            )

            # Visualize the most recent train and eval trajectories.
            extra = dict(
                ci_classify=ci_classify,
                pol_classify=pol_classify,
                buf=buf,
                custom=info_custom,
            )

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
