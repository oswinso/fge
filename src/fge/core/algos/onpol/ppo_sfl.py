import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax_dataclasses as jdc
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
from fge.core.algos.sfl_sampler import SFLSampler
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector, CollectorCfg
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class PPOSFLCfg:
    ppo: SumPPO.Cfg
    sfl: SFLSampler.Cfg

    def asdict(self):
        return asdict(self)


def train_ppo_sfl(
        run_cfg: RunCfg[PPOSFLCfg],
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

    collector, reset_id_provider = Collector.create(
        rngs(), env, run_cfg.col_cfg, id_provider=reset_id_provider
    )
    collector = jax.device_put(collector, gpu_device)

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

    sfl_cfg = run_cfg.algo_cfg.sfl
    buf = SFLSampler(env, sfl_cfg, reset_id_provider)

    # Get collector for eval rollout in SFL collect_learnable_levels.
    sfl_col_cfg = CollectorCfg(n_envs=sfl_cfg.init_sample_size)
    collector_sfl, _ = Collector.create(rngs(), env, sfl_col_cfg)
    collector_sfl = jax.device_put(collector_sfl, gpu_device)

    n_steps_per_collect = (
            run_cfg.col_cfg.n_envs * run_cfg.algo_cfg.ppo.train_cfg.rollout_T
    )

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

        # 0: If time, collect_learnable_levels.
        #           Careful, collect_learnable_levels mutates collector_sfl. Oopsies.
        if n_collects % sfl_cfg.update_buf_every == 0:
            pbar.set_description("Collect Learnable Levels...")
            key_sample_base = jr.fold_in(rngs(), n_collects)
            # buf.collect_learnable_levels(_key, ppo, collector_sfl, pbar)

            # 1. Get N random levels
            pbar.set_description("SFL #1. Collecting N random levels...")
            b_key_sample = jr.split(key_sample_base, buf.cfg.init_sample_size)
            b_state_x0_ = jax.vmap(buf.buffer.task.reset)(b_key_sample)
            b_state_x0_ = jax.device_put(b_state_x0_, gpu_device)

            # 2. Rollout \pi for L steps for all \theta \in B
            pbar.set_description("SFL #2. Collecting rollout for random levels...")
            collector_sfl_state = collector_sfl.collect_state._replace(state=b_state_x0_)
            collector_sfl = collector_sfl.replace(collect_state=collector_sfl_state)
            # collector_sfl, rollout_sfl, _ = ppo.collect_eval_w_col(collector_sfl)
            collector_sfl, rollout_sfl, _ = ppo.collect_eval_w_col_spec_T(collector_sfl,
                                                                          buf.buffer.task.eval_rollout_T * sfl_cfg.T_coef)
            jax.copy_to_host_async(rollout_sfl)

            # 3. Compute learnability score for each level
            pbar.set_description("SFL #3. Computing learnability scores...")
            # p is success rate for each rollout. We compute p as (# truncs) / (# terms + # truncs)
            num_term = jnp.sum(rollout_sfl.T_term, axis=-1)
            num_trunc = jnp.sum(rollout_sfl.T_trunc, axis=-1)
            p = num_trunc / (num_term + num_trunc)
            learnability = p * (1 - p)

            # 4. Get the top K levels by learnability
            pbar.set_description("SFL #4. Getting top K levels by learnability...")
            # Get the top K levels by learnability
            K = buf.cfg.sfl_buf_size
            b_idxs_topk = jnp.argsort(learnability)[-K:]
            b_state_x0_topk = jtu.tree_map(
                lambda x: x[b_idxs_topk], b_state_x0_
            )
            # Replace the source to all be 2, which is the "SFL" source.
            b_state_x0_topk = jdc.replace(
                b_state_x0_topk,
                source=jnp.full(b_state_x0_topk.source.shape, Source.BUF_CI,
                                dtype=b_state_x0_topk.source.dtype)
            )

            # Finally, set the data in the buffer.
            pbar.set_description("SFL #5. Setting data in the buffer...")
            buf.buffer = buf.buffer.set_data(
                b_data=StateResetId(
                    state=b_state_x0_topk,
                    reset_id=jnp.full(buf.cfg.sfl_buf_size, -42069, dtype=jnp.int32),
                ),
                size=buf.cfg.sfl_buf_size,
            )
            # buf.buffer = jdc.replace(buf.buffer,
            #                          b_data=StateResetId(
            #                              state=b_state_x0_topk,
            #                              reset_id=jnp.full(buf.cfg.sfl_buf_size, -42069, dtype=jnp.int32)
            #                          ))

        # 1: Collect rollout.
        pbar.set_description("Collect...")
        rollout: Collector.Rollout
        collector, buf.buffer, rollout, _ = ppo.collect_with_buf(collector, buf.buffer)
        jax.copy_to_host_async(rollout)

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
                b_x0 = ppo.task.leaf_to_minstate(
                    leaf_stack(trajsaver.all_x0s(), axis=0, which=np)
                )
                # Save b_x0 as pkl.
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)

        # 5: Maybe eval.
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
            extra = dict(
                sfl=buf,
                collector_sfl=collector_sfl,
                collector_evals=collector_evals,
                sfl_x0_og=b_state_x0_,
                sfl_x0_topk=b_state_x0_topk,
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
