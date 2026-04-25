import cyclopts
import jax
from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_accel import PPOAccelCfg, train_ppo_accel, AccelCfg
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.envs.mujoco.cheetah import cheetah
from fge.core.envs.mujoco.cheetah.cheetah_jax import CheetahJax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
    log_train,
)
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache
from loguru import logger

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-4,
        n_env: int = 1024,
        n_batch: int = 1000,
        # We increased rollout_T from 30 -> 1000, which is 33x. Increase the batch size similar amount.
        normalize_adv: bool = True,
        value_lims_epsilon: float | None = 1e-1,
        pol_lr: float = 4e-4,
        val_lr: float = 8e-4,
        n_update_epochs: int = 1,
        seed: int = 12345,
        save_all: bool = False,
):
    jax.config.update("jax_default_matmul_precision", "highest")
    enable_compilation_cache()

    # By design, this only works with rollout_T = max environment steps...
    # We increased rollout_T from 30 -> 1000, which is 33x. Increase the batch size
    max_env_steps = 1_000
    rollout_T_orig = 30
    rollout_T = max_env_steps

    hids = [256, 256]
    train_cfg = SumPPO.TrainCfg(
        gae_lambda=0.95,
        n_batches=n_batch,
        rollout_T=rollout_T,
        n_update_epochs=n_update_epochs,
        clip_ratio=0.1,
        normalize_adv=normalize_adv,
    )
    if value_lims_epsilon is None:
        value_lims = None
    else:
        value_lims = (-value_lims_epsilon, 1 + value_lims_epsilon)
    ppo_cfg = SumPPOCfg(
        pol_lr=pol_lr,
        val_lr=val_lr,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_cfg,
    )
    accel_cfg = AccelCfg()
    algo_cfg = PPOAccelCfg(ppo=ppo_cfg, accel=accel_cfg)

    task_cfg = cheetah.TaskCfg(max_steps=max_env_steps)

    train_cfg = TrainCfg(
        n_steps=50_000, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
    )
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=2)
    task_name = "CheetahJax"

    run_cfg = RunCfg.setup(
        seed,
        algo_cfg,
        task_cfg,
        train_cfg,
        col_cfg,
        eval_cfg,
        task_name,
        use_wandb=use_wandb,
        wandb_name=name,
    )

    total_env_steps = run_cfg.train_cfg.n_steps * n_env * rollout_T_orig

    run_cfg.save_all = save_all
    env = CheetahJax(run_cfg.task_cfg)
    train_ppo_accel(
        total_env_steps,
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,

            # plot_train,
            # plot_train_x0,
            # plot_train_x0_extended,
            # plot_eval,
            # plot_eval_det,
            # VizValue1D(env),

            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        app()
