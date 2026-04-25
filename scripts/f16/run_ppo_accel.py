import cyclopts
import ipdb

from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_accel import PPOAccelCfg, train_ppo_accel, AccelCfg
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.envs.f16_avoid import f16_avoid
from fge.core.envs.f16_avoid.f16_avoid_jax import F16AvoidJax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
    log_train,
)
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-4,
        n_env: int = 1024,
        n_batch: int = 500,
        normalize_adv: bool = True,
        value_lims_epsilon: float | None = 1e-1,
        pol_lr: float = 4e-4,
        n_update_epochs: int = 1,
        seed: int = 12345,
        eval_every: int = 200,
):
    enable_compilation_cache()

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
        val_lr=8e-4,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_cfg,
    )
    accel_cfg = AccelCfg()
    algo_cfg = PPOAccelCfg(ppo=ppo_cfg, accel=accel_cfg)

    task_cfg = f16_avoid.TaskCfg(max_steps=max_env_steps)

    train_cfg = TrainCfg(
        n_steps=50_000,
        log_every=10,
        fastcb_every=20,
        eval_every=eval_every,
        save_every=1_000,
    )
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=2)
    task_name = "F16Jax"

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

    env = F16AvoidJax(run_cfg.task_cfg)
    train_ppo_accel(
        total_env_steps,
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
