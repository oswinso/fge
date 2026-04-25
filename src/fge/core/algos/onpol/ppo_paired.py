import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import psutil
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
from fge.core.bits.collector import Collector, RolloutOutput
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.reset_id_provider import ResetIDProvider
from fge.core.bits.sumppo_x0 import SumPPOX0
from fge.core.bits.x0_collector import X0Collector
from fge.core.envs.get_task import make_task
from fge.core.envs.jax_task import leaf_stack
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


@define
class PPOPAIREDCfg:
    ppo: SumPPO.Cfg  # Protagonist PPO configuration
    ppo_ant: SumPPO.Cfg
    ppo_adv_x0: SumPPOX0.Cfg

    pro_ant_obj: str = "regret"  # "regret", "u"

    def asdict(self):
        return asdict(self)


def compute_regret(
        ppo_pro: SumPPO,
        ppo_ant: SumPPO,
        rollouts_pro: RolloutOutput,
        rollouts_ant: RolloutOutput,
):
    dset_pro: SumPPO.PPOBatchQvals
    dset_ant: SumPPO.PPOBatchQvals
    dset_pro = ppo_pro.make_bT_dset_Qvals_only(rollouts_pro)[0]
    dset_ant = ppo_ant.make_bT_dset_Qvals_only(rollouts_ant)[0]

    # Compute the max, but note that Q-values at the last episode may be truncated and thus not valid. So ignore
    # the last episode when computing the max.
    dones_pro = rollouts_pro.T_term | rollouts_pro.T_trunc
    dones_ant = rollouts_ant.T_term | rollouts_ant.T_trunc
    dones_pro_rev = jnp.flip(dones_pro, axis=-1)
    dones_ant_rev = jnp.flip(dones_ant, axis=-1)
    cutoffs_pro = dones_pro.shape[-1] - 1 - jnp.argmax(dones_pro_rev, axis=-1)
    cutoffs_ant = dones_ant.shape[-1] - 1 - jnp.argmax(dones_ant_rev, axis=-1)

    # For each trajectory, find the last True in dones_ant and cut off the Q-values at that point.
    B, T = dset_ant.b_Qvals.shape
    idxs = jnp.arange(
        T
    )  # shape [T]. Create a mask where each entry is True if it's <= cutoff
    mask = idxs[None, :] <= cutoffs_ant[:, None]  # shape [B, T]
    masked_Q = jnp.where(
        mask, dset_ant.b_Qvals, -jnp.inf
    )  # Mask out invalid values with -inf
    max_U_A = jnp.max(masked_Q, axis=-1)  # Take row-wise max

    B, T = dset_pro.b_Qvals.shape
    idxs = jnp.arange(T)
    mask = idxs[None, :] <= cutoffs_pro[:, None]  # shape [B, T]
    masked_Q = jnp.where(
        mask, dset_pro.b_Qvals, 0.0
    )  # Zero out invalid values (instead of -inf)
    sum_Q = jnp.sum(masked_Q, axis=1)  # Compute the sum over valid entries
    lengths = (
            cutoffs_pro + 1
    )  # Compute the number of valid entries per row (cutoff + 1)
    lengths = jnp.maximum(lengths, 1)  # Avoid division by zero (if any cutoff == -1)
    E_U_P = sum_Q / lengths  # Compute the expectation

    regret = max_U_A - E_U_P

    return {
        "regret": regret,
        "max_U_A": max_U_A,
        "E_U_P": E_U_P,
        "dones_pro": dones_pro,
        "dones_ant": dones_ant,
    }


def train_ppo_paired(
        run_cfg: RunCfg[PPOPAIREDCfg],
        seed,
        fast_cbs: list[Callable] = [],
        eval_cbs: list[Callable] = [],
):
    proc = psutil.Process(os.getpid())

    def rss_mb():
        return proc.memory_info().rss / (1024 ** 2)

    start_time = time.time()
    train_cfg = run_cfg.train_cfg
    writer = SummaryWriter(run_cfg.paths.run_dir)

    env = make_task(run_cfg.task_cfg)

    rngs = nnx.Rngs(seed)

    gpu_device = jax.devices("gpu")[0]

    obs = env.get_dummy_obs()

    # Protagonist
    ppo = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo)
    ppo: SumPPO = jax.device_put(ppo, gpu_device)

    # Antagonist
    ppo_ant = SumPPO.create(rngs(), env, run_cfg.algo_cfg.ppo_ant)
    ppo_ant: SumPPO = jax.device_put(ppo_ant, gpu_device)

    # Adversary (Environment Policy that selects x0's)
    ppo_adv_x0 = SumPPOX0.create(rngs(), env, run_cfg.algo_cfg.ppo_adv_x0)
    ppo_adv_x0: SumPPOX0 = jax.device_put(ppo_adv_x0, gpu_device)

    # x0_reset_fn: X0ResetBuf = ppo_adv_x0.get_reset_buf()
    # x0_reset_fn = jax.device_put(x0_reset_fn, gpu_device)

    reset_id_provider = jax.device_put(ResetIDProvider.create(), gpu_device)

    collector_pro, _ = Collector.create(rngs(), env, run_cfg.col_cfg)
    collector_pro = jax.device_put(collector_pro, gpu_device)

    collector_ant, _ = Collector.create(rngs(), env, run_cfg.col_cfg)
    collector_ant = jax.device_put(collector_ant, gpu_device)

    collector_adv_x0 = X0Collector.create(
        rngs(), env, run_cfg.col_cfg, ppo_adv_x0.cfg.disc_gamma
    )
    collector_adv_x0 = jax.device_put(collector_adv_x0, gpu_device)

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
    # ckpt_manager = get_ckpt_manager(ckpt_dir, ["alg_pro", "alg_ant", "alg_adv_x0"], max_to_keep=None,
    #                                 step_format_fixed_length=8)
    ckpt_manager = get_ckpt_manager(
        ckpt_dir,
        ["alg", "alg_ant", "alg_adv_x0"],
        max_to_keep=None,
        step_format_fixed_length=8,
    )
    ckpt_manager.save_config(ckpt_cfgs)

    # Determine whether to save full trajectories based on the environment type.
    if isinstance(env, ToyLevelsJax):
        save_full_traj = True
    else:
        save_full_traj = False

    # Initialize the trajectory savers.
    if save_full_traj:
        trajsaver_pro = TrajSaver(save_full_traj=save_full_traj)
        trajsaver_ant = TrajSaver(save_full_traj=save_full_traj)
    else:
        trajsaver_pro = FastTrajSaver()
        trajsaver_ant = FastTrajSaver()

    rollout_T = run_cfg.algo_cfg.ppo.train_cfg.rollout_T
    assert run_cfg.algo_cfg.ppo_adv_x0.train_cfg.rollout_T_override is not None
    rollout_T_override = run_cfg.algo_cfg.ppo_adv_x0.train_cfg.rollout_T_override
    adj_n_steps = (rollout_T * train_cfg.n_steps) // rollout_T_override
    logger.info(
        "Adjusted number of steps for PAIRED: {} -> {}".format(
            train_cfg.n_steps, adj_n_steps
        )
    )
    pbar = tqdm.trange(adj_n_steps, desc="Collecting Rollouts")
    for n_collects in pbar:
        final_iter = n_collects >= adj_n_steps

        # 1: Collect x0s from the adversary.
        pbar.set_description("Collect Adv...")
        rollout_pro: Collector.Rollout
        rollout_ant: Collector.Rollout
        collector_adv_x0, b_x0_data, info_collect = ppo_adv_x0.collect(
            collector_adv_x0, ppo
        )
        states_x0 = info_collect["x0"]

        # Set the collectors to use the sampled x0s.
        collector_pro_state = collector_pro.collect_state._replace(state=states_x0)
        collector_ant_state = collector_ant.collect_state._replace(state=states_x0)
        collector_pro = collector_pro.replace(collect_state=collector_pro_state)
        collector_ant = collector_ant.replace(collect_state=collector_ant_state)

        # 2. Collect rollouts for both pro and ant on the sampled x0s.
        pbar.set_description("Collect Pro and Ant...")
        collector_pro, rollout_pro, _ = ppo.collect_eval_w_col(collector_pro)
        collector_ant, rollout_ant, _ = ppo.collect_eval_w_col(collector_ant)
        jax.copy_to_host_async(rollout_pro)
        jax.copy_to_host_async(rollout_ant)

        # 3. Compute regrets = max_\tau^A U(\tau^A) - E_\tau^P [ U(\tau^P ]
        pbar.set_description("Compute Regret...")

        ret = compute_regret(
            ppo_pro=ppo,
            ppo_ant=ppo_ant,
            rollouts_pro=rollout_pro,
            rollouts_ant=rollout_ant,
        )
        regret = ret["regret"]
        max_U_A = ret["max_U_A"]
        E_U_P = ret["E_U_P"]
        dones_pro = ret["dones_pro"]
        dones_ant = ret["dones_ant"]

        # Update rollouts with the new rewards.
        match run_cfg.algo_cfg.pro_ant_obj:
            case "u":
                pass
            case "regret":
                regret_broadcast = regret[:, None]  # shape [256, 1]
                rollout_pro.T_rew = jnp.where(dones_pro == 1, -regret_broadcast, 0)
                rollout_ant.T_rew = jnp.where(dones_ant == 1, regret_broadcast, 0)

        # Truncate the rollouts to just the rollout_T size
        # TODO: TRUNCATE TO NEAREST DIVISIBLE NUMBER
        trunc_len = (rollout_pro.T_rew.shape[1] // rollout_T) * rollout_T
        rollout_pro_trunc = jtu.tree_map(lambda x: x[:, :trunc_len], rollout_pro)
        rollout_ant_trunc = jtu.tree_map(lambda x: x[:, :trunc_len], rollout_ant)

        # 4. Update PPO Protagonist, Antagonist, and Adversary
        pbar.set_description("Update PPO Protagonist...")
        ppo, info_loss_pro = ppo.update(rollout_pro_trunc)

        pbar.set_description("Update PPO Antagonist...")
        ppo_ant, info_loss_ant = ppo_ant.update(rollout_ant_trunc)

        b_x0_data.b_rew = regret.copy()
        pbar.set_description("Update PPO (x0)...")
        ppo_adv_x0, info_loss = ppo_adv_x0.update(b_x0_data)

        # # TODO
        # # 6. Save the collected trajectories.
        # #       We mainly just want to save 1 traj from each env, which is what the adversary collects.
        # pbar.set_description("Add rollout...")
        # rollout_pro = split_trajs(jax2np(rollout_pro))
        # rollout_pro = RolloutOutput.tree_concat(rollout_pro, axis=0)
        # rollout_pro = RolloutOutput.tree_stack([rollout_pro], axis=0)

        # TODO: comment out trajectory saving for now to save memory
        # pbar.set_description("Adding rollouts...")
        # trajsaver_pro.add_rollout(jax2np(rollout_pro))
        # trajsaver_ant.add_rollout(jax2np(rollout_ant))
        #
        # pbar.set_description("Logging")

        # 7. Log training progress
        if n_collects % train_cfg.log_every == 0:
            info_collect_pro = trajsaver_pro.get_stats(train_cfg.log_every)
            info_collect_ant = trajsaver_ant.get_stats(train_cfg.log_every)

            # Remove anything from info_loss that starts with nolog.
            info_loss_pro = {
                k: v for k, v in info_loss_pro.items() if not k.startswith("nolog/")
            }
            info_loss_ant = {
                k: v for k, v in info_loss_ant.items() if not k.startswith("nolog/")
            }
            info_loss = {
                k: v for k, v in info_loss.items() if not k.startswith("nolog/")
            }

            log_dict = {f"TrainPro/{k}": v for k, v in info_loss_pro.items()}
            log_dict = log_dict | {f"TrainAnt/{k}": v for k, v in info_loss_ant.items()}
            log_dict = log_dict | {
                f"CollectPro/{k}": v for k, v in info_collect_pro.items()
            }
            log_dict = log_dict | {
                f"CollectAnt/{k}": v for k, v in info_collect_ant.items()
            }
            log_dict = log_dict | {"n_collect": n_collects}
            log_dict = log_dict | {f"TrainAdvX0/{k}": v for k, v in info_loss.items()}
            log_dict_tb(writer, log_dict, global_step=n_collects)

        # 8. Save the model at specified intervals or at the final iteration.
        if (n_collects % train_cfg.save_every == 0) or final_iter:
            ckpt_manager.save_ez(
                n_collects,
                {
                    "alg": ppo,
                    # "alg_pro": ppo_pro,
                    "alg_ant": ppo_ant,
                    "alg_adv_x0": ppo_adv_x0,
                },
            )
            # ckpt_manager.save_ez(n_collects, {"alg": ppo_pro})

            if run_cfg.save_all:
                # Also save recent x0s.
                b_x0 = ppo_adv_x0.task.leaf_to_minstate(
                    leaf_stack(trajsaver_pro.all_x0s(), axis=0, which=np)
                )
                # Save b_x0 as pkl.
                pkl_dir = run_cfg.paths.run_dir / "pkls"
                pkl_dir.mkdir(exist_ok=True, parents=True)

                pkl_path = pkl_dir / "recent_x0s_{}.pkl".format(n_collects)
                with open(pkl_path, "wb") as f:
                    np.save(f, b_x0)

        # 9. Perform evaluation at specified intervals.
        if n_collects % train_cfg.eval_every == 0:
            pbar.set_description("Eval...")
            rollouts_eval = []
            rollouts_eval_pro = []
            rollouts_eval_ant = []
            for collector_eval in tqdm.tqdm(collector_evals):
                rollout_trunc = split_trajs(jax2np(ppo.collect_eval(collector_eval)[0]))
                rollouts_eval.append(rollout_trunc)
                rollouts_eval_pro.append(rollout_trunc)
                rollout_trunc_ant = split_trajs(
                    jax2np(ppo_ant.collect_eval(collector_eval)[0])
                )
                rollouts_eval_ant.append(rollout_trunc_ant)

            pbar.set_description("Eval Det...")
            rollout_eval_det = split_trajs(
                jax2np(ppo.collect_eval_det(collector_evals[0])[0])
            )

            _rollout_eval_pro = ppo.collect_eval(collector_evals[0])[0]
            _rollout_eval_ant = ppo_ant.collect_eval(collector_evals[0])[0]

            _rollout_eval_det_pro = ppo.collect_eval_det(collector_evals[0])[0]
            _rollout_eval_det_ant = ppo_ant.collect_eval_det(collector_evals[0])[0]
            rollout_eval_det_pro = split_trajs(jax2np(_rollout_eval_det_pro))
            rollout_eval_det_ant = split_trajs(jax2np(_rollout_eval_det_ant))

            ret = compute_regret(
                ppo_pro=ppo,
                ppo_ant=ppo_ant,
                rollouts_pro=_rollout_eval_pro,
                rollouts_ant=_rollout_eval_ant,
            )
            regret = ret["regret"]
            max_U_A = ret["max_U_A"]
            E_U_P = ret["E_U_P"]

            # Visualize the most recent train and eval trajectories.
            extra = dict(
                ppo_adv_x0=ppo_adv_x0,
                ppo_pro=ppo,
                ppo_ant=ppo_ant,
                trajsaver_pro=trajsaver_pro,
                trajsaver_ant=trajsaver_ant,
                rollout_eval_det_pro=rollout_eval_det_pro,
                rollout_eval_det_ant=rollout_eval_det_ant,
                regret=regret,
                max_U_A=max_U_A,
                E_U_P=E_U_P,
            )
            props = EvalProps(
                writer,
                ppo,
                trajsaver_pro,
                rollouts_eval,
                rollout_eval_det,
                eval_region_info,
                extra,
            )
            for cb in eval_cbs:
                pbar.set_description("Eval Plot... ({})".format(str(cb)))
                cb(n_collects, run_cfg, props)

            # Explicitly clear trajectories
            trajsaver_pro.clear_trajs()
            trajsaver_ant.clear_trajs()

            pbar.set_description("Eval... Done!")

        # Execute fast callbacks if specified.
        if (train_cfg.fastcb_every > 0) and (n_collects % train_cfg.fastcb_every == 0):
            pbar.set_description("Fast Eval...")
            props = EvalProps(writer, ppo, None, None, None, eval_region_info)
            for cb in fast_cbs:
                cb(n_collects, run_cfg, props)
            pbar.set_description("Fast Eval... Done!")

        logger.critical(f"RSS {rss_mb():.1f} MB")

    # Finally save everything again just in case
    ckpt_manager.save_ez(
        adj_n_steps,
        {
            "alg": ppo,
            # "alg_pro": ppo_pro,
            "alg_ant": ppo_ant,
            "alg_adv_x0": ppo_adv_x0,
        },
    )

    # Save the final model.
    ckpt_manager.wait_until_finished()
    time.sleep(10)
