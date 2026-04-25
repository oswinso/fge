from typing import Annotated as Anno

import cyclopts
from cyclopts import Parameter
from loguru import logger

from fge.core.algos.buf_custom import BufCustom, RehearsalOPT, TrainNSFOPT, ValueOPT
from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_fge import PPOFGECfg, train_ppo_fge
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.common.reset_buf import ResetBuf
from fge.core.envs.mujoco.hopper import hopper
from fge.core.envs.mujoco.hopper.analyze_ppo import (
    plot_eval_det,
    plot_train,
    plot_train_x0, )
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
    log_train,
)
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        resetbuf_cfg: ResetBuf.Cfg = ResetBuf.Cfg(),
        ci_classify_cfg: Anno[
            CIClassifier.Cfg, Parameter(name="cicls", group="pol classifier")
        ] = CIClassifier.Cfg(n_sample=8192, n_batches=16),
        pol_classify_cfg: Anno[
            CIClassifier.Cfg, Parameter(name="polcls", group="pol classifier")
        ] = CIClassifier.Cfg(),
        nsf_cfg: Anno[NSF.Cfg, Parameter(name="nsf", group="nsf")] = NSF.Cfg(),
        nsf_ci_cfg: Anno[NSF.Cfg, Parameter(name="nsf", group="nsf_ci")] = NSF.Cfg(),
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-3,
        n_env: int = 1024,
        n_batch: int = 15,
        pol_lr: float = 3e-4,
        val_lr: float = 8e-4,
        normalize_adv_pre: bool = True,
        normalize_adv: bool = False,
        seed: int = 12345,
        value_opt: ValueOPT = "NSF",
        value_lims_epsilon: float | None = 1e-1,
        n_update_epochs: int = 2,
        train_nsf_mode: TrainNSFOPT = "b_x0",
        n_steps: int = 50_000,
        rehearsal_mode: RehearsalOPT = "polsafe",
        rehearsal_only_frac: float | None = 0.2,
        use_nsf_explore: bool = True,
        ci_clsfy_frac: float | None = None,
        save_all: bool = False,
        px0_bounds: tuple[float, float] = (-1.5, 2.5),
):
    enable_compilation_cache()

    hids = [256, 256]
    train_cfg = SumPPO.TrainCfg(
        gae_lambda=0.95,
        n_batches=n_batch,
        rollout_T=30,
        n_update_epochs=n_update_epochs,
        clip_ratio=0.1,
        normalize_adv_pre=normalize_adv_pre,
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
    buf_cfg = BufCustom.Cfg(
        resetbuf=resetbuf_cfg,
        n_resetbuf_init=4,
        value_opt=value_opt,
        train_nsf_mode=train_nsf_mode,
        rehearsal_mode=rehearsal_mode,
        rehearsal_only_frac=rehearsal_only_frac,
        use_nsf_explore=use_nsf_explore,
        ci_clsfy_fracci=ci_clsfy_frac,
    )

    algo_cfg = PPOFGECfg(
        ppo=ppo_cfg,
        nsf=nsf_cfg,
        buf=buf_cfg,
        ci_classify=ci_classify_cfg,
        pol_classify=pol_classify_cfg,
        nsf_ci=nsf_ci_cfg,
    )

    logger.info(f"Using px0 bounds: {px0_bounds}")
    task_cfg = hopper.TaskCfg(max_steps=1_000, px0_bounds=px0_bounds)

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

    train_ppo_fge(
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
