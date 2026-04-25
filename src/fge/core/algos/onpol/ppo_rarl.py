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
from torch.utils.tensorboard import SummaryWriter

from fge.core.algos.fast_trajsaver import FastTrajSaver
from fge.core.algos.onpol.ppo import EvalProps, RunCfg, make_collector_evals
from fge.core.algos.traj_utils import split_trajs
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.collector import Collector
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.sumppo_x0 import SumPPOX0, X0ResetBuf
from fge.core.bits.x0_collector import X0Collector
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class RARLCfg:
    n_ppo_steps: int
    """How many ppo steps to run before switching to the adversary steps."""

    n_adv_steps: int
    """How many adversary steps to run before switching back to PPO."""


@define
class PPORARLCfg:
    ppo: SumPPO.Cfg
    ppo_x0: SumPPOX0.Cfg
    rarl: RARLCfg

    def asdict(self):
        return asdict(self)


def train_ppo_rarl(
    run_cfg: RunCfg[PPORARLCfg],
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
    ppo: SumPPO = jax.device_put(ppo, gpu_device)

    ppo_x0 = SumPPOX0.create(rngs(), env, run_cfg.algo_cfg.ppo_x0)
    ppo_x0: SumPPOX0 = jax.device_put(ppo_x0, gpu_device)

    x0_reset_fn: X0ResetBuf = ppo_x0.get_reset_buf()
    x0_reset_fn = jax.device_put(x0_reset_fn, gpu_device)

    reset_id_provider = jax.device_put(ResetIDProvider.create(), gpu_device)

    collector, reset_id_provider = Collector.create(
        rngs(), env, run_cfg.col_cfg, id_provider=reset_id_provider
    )
    collector = jax.device_put(collector, gpu_device)

    collector_x0 = X0Collector.create(
        rngs(), env, run_cfg.col_cfg, ppo_x0.cfg.disc_gamma
    )
    collector_x0 = jax.device_put(collector_x0, gpu_device)

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
        ckpt_dir, ["alg", "alg_x0"], max_to_keep=None, step_format_fixed_length=8
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

    rarl_cfg = run_cfg.algo_cfg.rarl
    n_steps_per_round = rarl_cfg.n_ppo_steps + rarl_cfg.n_adv_steps

    pbar = tqdm.trange(train_cfg.n_steps)
    n_collect_ppo = 0
    n_collect_ppo_x0 = 0
    for n_collects in pbar:
        final_iter = n_collects >= train_cfg.n_steps - 1

        train_ppo = n_collects % n_steps_per_round < rarl_cfg.n_ppo_steps

        # -----------------------------
        if train_ppo:
            # 1: Collect rollout.
            pbar.set_description("Collect...")
            rollout: Collector.Rollout
            collector, _, rollout, _ = ppo.collect_with_buf(collector, x0_reset_fn)
            jax.copy_to_host_async(rollout)

            # 2: Train PPO.
            pbar.set_description("Update PPO...")
            ppo, info_loss = ppo.update(rollout)

            # 3: Save the trajs.
            if trajsaver is not None:
                pbar.set_description("Add rollout...")
                rollout = jax2np(rollout)
                trajsaver.add_rollout(rollout)

            # 4: Maybe log.
            if n_collect_ppo % train_cfg.log_every == 0:
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

                log_dict = {f"Train/{k}": v for k, v in info_loss.items()}
                log_dict = log_dict | {
                    f"Collect/{k}": v for k, v in info_collect.items()
                }
                log_dict = log_dict | {"n_collect": n_collects}
                log_dict_tb(writer, log_dict, global_step=n_collects)

            n_collect_ppo += 1
        else:
            # 1: Collect rollout.
            pbar.set_description("Collect (x0)...")
            collector_x0, b_x0_data, info_collect = ppo_x0.collect(collector_x0, ppo)

            # 2: Collect rollout.
            pbar.set_description("Update PPO (x0)...")
            ppo_x0, info_loss = ppo_x0.update(b_x0_data)
            x0_reset_fn = ppo_x0.get_reset_buf(x0_reset_fn)

            # 3: Maybe log.
            if n_collect_ppo_x0 % train_cfg.log_every == 0:
                # Remove anything from info_loss that starts with nolog.
                info_loss = {
                    k: v for k, v in info_loss.items() if not k.startswith("nolog/")
                }

                if "steps" in info_collect:
                    pbar.set_postfix(ColX0Steps=info_collect["steps"])

                log_dict = {f"TrainX0/{k}": v for k, v in info_loss.items()}
                log_dict_tb(writer, log_dict, global_step=n_collects)

            n_collect_ppo_x0 += 1
        # -----------------------------
        # 6: Maybe save model.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
            ckpt_manager.save_ez(n_collects, {"alg": ppo, "alg_x0": ppo_x0})

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
        # -----------------------------
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
            extra = dict(ppo_x0=ppo_x0)
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

            pbar.set_description("Eval... Done!")

        if (train_cfg.fastcb_every > 0) and (n_collects % train_cfg.fastcb_every == 0):
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, trajsaver, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)
            pbar.set_description("Fast Eval... Done!")

    # Save the final model.
    ckpt_manager.wait_until_finished()
    time.sleep(10)
