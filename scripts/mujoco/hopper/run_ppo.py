import cyclopts
from loguru import logger

from fge.core.algos.onpol.ppo import EvalCfg, PPOOnlyCfg, RunCfg, TrainCfg, train_ppo
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.envs.mujoco.hopper import hopper
from fge.core.envs.mujoco.hopper.analyze_ppo import plot_eval_det, plot_train_x0, plot_train
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det, log_train,
)
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        px0: float | None = None,
        name: str | None = None,
        ent_cf: float = 2e-3,
        n_env: int = 1024,
        n_batch: int = 15,
        normalize_adv: bool = True,
        value_lims_epsilon: float | None = 1e-1,
        pol_lr: float = 4e-4,
        n_update_epochs: int = 1,
        seed: int = 12345,
        save_all: bool = False,
        px0_bounds: tuple[float, float] = (-1.5, 2.5),
):
    enable_compilation_cache()

    assert len(px0_bounds) == 2

    hids = [256, 256]
    train_cfg = SumPPO.TrainCfg(
        gae_lambda=0.95,
        n_batches=n_batch,
        rollout_T=30,
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
        val_lr=8e-4,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_cfg,
    )
    algo_cfg = PPOOnlyCfg(ppo=ppo_cfg)

    logger.info(f"Using px0 bounds: {px0_bounds}")
    task_cfg = hopper.TaskCfg(max_steps=1_000, px0_bounds=px0_bounds)

    if px0 is not None:
        task_cfg.px0 = px0

    train_cfg = TrainCfg(
        n_steps=50_000, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
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
    train_ppo(
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,
            plot_train,
            plot_train_x0,
            plot_eval_det,
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        app()
