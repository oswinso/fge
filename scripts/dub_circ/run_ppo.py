import cyclopts
import ipdb
import jax_dataclasses as jdc
import numpy as np

from fge.core.algos.onpol.ppo import EvalCfg, PPOOnlyCfg, RunCfg, TrainCfg, train_ppo
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.envs.dub_circ import dub_circ_jax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
)
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        use_wandb: bool = False,
        pv0: tuple[float, float] | None = None,
        name: str | None = None,
        ent_cf: float = 5e-3,
        n_env: int = 1024,
        n_batch: int = 30,
        normalize_adv: bool = True,
        value_lims_epsilon: float | None = 1e-1,
        pol_lr: float = 5e-5,
        val_lr: float = 8e-4,
        n_update_epochs: int = 1,
        seed: int = 12345,
):
    enable_compilation_cache()

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
        val_lr=val_lr,
        entropy_cf=ent_cf,
        disc_gamma=0.995,
        pol_hids=hids,
        val_hids=hids,
        value_lims=value_lims,
        train_cfg=train_cfg,
    )
    algo_cfg = PPOOnlyCfg(ppo=ppo_cfg)

    task_cfg = dub_circ_jax.TaskCfg(max_timesteps=1_000)
    if pv0 is not None:
        task_cfg = jdc.replace(task_cfg, pv0=np.array(pv0))

    train_cfg = TrainCfg(
        n_steps=50_000, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
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
    train_ppo(
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
    with ipdb.launch_ipdb_on_exception():
        app()
