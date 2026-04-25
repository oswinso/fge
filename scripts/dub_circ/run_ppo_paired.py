import cyclopts

from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_paired import PPOPAIREDCfg, train_ppo_paired
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.bits.sumppo_x0 import SumPPOX0, SumPPOX0Cfg
from fge.core.envs.dub_circ import dub_circ_jax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
)
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 5e-3,
        ent_cf_x0: float = 5e-2,
        n_env: int = 1024,
        n_batch: int = 30,
        n_batch_x0: int = 16,
        pol_lr: float = 5e-5,
        val_lr: float = 8e-4,
        normalize_adv: bool = True,
        seed: int = 12345,
        value_lims_epsilon: float | None = 1e-1,
        n_update_epochs: int = 1,
        n_update_epochs_adv: int = 1,
        n_steps: int = 50_000,
        x0_rollout_T: (
                int | None
        ) = 1000,
        pro_ant_obj: str = "regret",  # "regret", "u" (utility)
):
    enable_compilation_cache()

    hids = [256, 256]
    train_pro_cfg = SumPPO.TrainCfg(
        gae_lambda=0.95,
        n_batches=n_batch,
        rollout_T=30,
        n_update_epochs=n_update_epochs,
        clip_ratio=0.1,
        normalize_adv=normalize_adv,
    )
    train_ant_cfg = SumPPO.TrainCfg(
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

    ppo_pro_cfg = SumPPOCfg(
        pol_lr=pol_lr,
        val_lr=val_lr,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_pro_cfg,
    )
    ppo_ant_cfg = SumPPOCfg(
        pol_lr=pol_lr,
        val_lr=val_lr,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_ant_cfg,
    )

    train_adv_cfg_x0 = SumPPOX0.TrainCfg(
        n_batches=n_batch_x0,
        n_update_epochs=n_update_epochs_adv,
        clip_ratio=0.1,
        normalize_adv=True,
        rollout_T_override=x0_rollout_T,
    )
    ppo_adv_x0_cfg = SumPPOX0Cfg(
        pol_lr=1e-3,
        val_lr=1e-2,
        entropy_cf=ent_cf_x0,
        disc_gamma=0.995,
        value_lims=value_lims,
        train_cfg=train_adv_cfg_x0,
    )
    algo_cfg = PPOPAIREDCfg(
        ppo=ppo_pro_cfg, ppo_ant=ppo_ant_cfg, ppo_adv_x0=ppo_adv_x0_cfg, pro_ant_obj=pro_ant_obj
    )

    task_cfg = dub_circ_jax.TaskCfg(max_timesteps=x0_rollout_T)

    train_cfg = TrainCfg(
        n_steps=n_steps, log_every=10, fastcb_every=10, eval_every=10, save_every=500
    )
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=2)
    task_name = "DubinsJax"

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
    train_ppo_paired(
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_eval,
            log_eval_det,
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        app()
