import cyclopts

from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_vds import PPOVDSCfg, train_ppo_vds
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper import hopper
from fge.core.envs.mujoco.hopper.analyze_ppo import (
    plot_eval_det,
    plot_train,
    plot_train_x0,
)
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
    log_train,
)
from fge.core.envs.toylevels.viz_value import VizVDSValue1D
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-4,
        n_env: int = 1024,
        n_batch: int = 30,
        n_update_epoch: int = 1,
        normalize_adv: bool = True,
        value_lim_vds: bool = False,
        seed: int = 12345,
        pol_lr: float = 4e-4,
        val_lr: float = 8e-4,
        value_lims_epsilon: float | None = 1e-1,
        n_steps: int = 50_000,
        save_all: bool = False,
):
    enable_compilation_cache()

    hids = [256, 256]
    train_cfg = SumPPO.TrainCfg(
        gae_lambda=0.95,
        n_batches=n_batch,
        rollout_T=30,
        n_update_epochs=n_update_epoch,
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
    vds_cfg = VDSGAE.Cfg(
        lr=8e-4,
        gae_lambda=0.95,
        disc_gamma=0.995,
        n_batches=n_batch,
        n_update_epochs=n_update_epoch,
    )

    if value_lim_vds:
        if value_lims is None:
            vds_cfg.value_lims = (-1e-3, 1.0)
        else:
            vds_cfg.value_lims = value_lims
    else:
        vds_cfg.value_lims = None

    algo_cfg = PPOVDSCfg(ppo=ppo_cfg, vds=vds_cfg)

    task_cfg = hopper.TaskCfg(max_steps=1_000)

    train_cfg = TrainCfg(
        n_steps=n_steps, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
    )
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=2)
    task_name = "HopperJax"

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
    run_cfg.save_all = save_all
    env = HopperJax(run_cfg.task_cfg)
    train_ppo_vds(
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,
            plot_train,
            plot_train_x0,
            plot_eval_det,
            VizVDSValue1D(env),
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        app()
