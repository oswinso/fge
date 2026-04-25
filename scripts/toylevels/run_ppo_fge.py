from typing import Annotated

import cyclopts
from cyclopts import Parameter

from fge.core.algos.buf_custom import BufCustom, RehearsalOPT, TrainNSFOPT, ValueOPT
from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_fge import PPOFGECfg, train_ppo_fge
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.common.reset_buf import ResetBuf
from fge.core.envs.toylevels import toylevels
from fge.core.envs.toylevels.analyze_ppo import (
    clear_trajs,
    log_eval,
    log_eval_det,
    log_train,
    plot_eval_det,
    plot_train_x0,
)
from fge.core.utils.debugging import launch_ipdb_on_exception
from fge.core.utils.jax_cache import enable_compilation_cache

app = cyclopts.App()


@app.default
def main(
        resetbuf_cfg: ResetBuf.Cfg = ResetBuf.Cfg(),
        ci_classify_cfg: Annotated[
            CIClassifier.Cfg, Parameter(name="cicls", group="pol classifier")] = CIClassifier.Cfg(),
        pol_classify_cfg: Annotated[
            CIClassifier.Cfg, Parameter(name="polcls", group="pol classifier")
        ] = CIClassifier.Cfg(),
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-3,
        n_env: int = 1024,
        n_batch: int = 30,
        pol_lr: float = 4e-4,
        val_lr: float = 1e-3,
        normalize_adv: bool = True,
        seed: int = 12345,
        value_opt: ValueOPT = "NSF",
        value_lims_epsilon: float | None = 1e-1,
        n_update_epochs: int = 1,
        train_nsf_mode: TrainNSFOPT = "b_x0",
        n_steps: int = 5_000,
        rehearsal_mode: RehearsalOPT = "polsafe",  # "polsafe", "Vl"
        rehearsal_only_frac: float | None = None,
        p_explore: float = 0.8,
        p_base: float = 0.02,
        p_predci: float = 0.08,
        use_nsf_explore: bool = True,
):
    enable_compilation_cache()

    resetbuf_cfg.p_explore = p_explore
    resetbuf_cfg.p_base = p_base
    resetbuf_cfg.p_predci = p_predci

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
    buf_cfg = BufCustom.Cfg(
        resetbuf=resetbuf_cfg,
        n_resetbuf_init=4,
        value_opt=value_opt,
        train_nsf_mode=train_nsf_mode,
        rehearsal_mode=rehearsal_mode,
        rehearsal_only_frac=rehearsal_only_frac,
        use_nsf_explore=use_nsf_explore,
    )
    nsf_cfg = NSF.Cfg()
    algo_cfg = PPOFGECfg(
        ppo=ppo_cfg,
        nsf=nsf_cfg,
        buf=buf_cfg,
        ci_classify=ci_classify_cfg,
        pol_classify=pol_classify_cfg,
    )

    task_cfg = toylevels.TaskCfg(max_steps=1_000)

    train_cfg = TrainCfg(n_steps=n_steps, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000)
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=2)
    task_name = "ToyLevelsJax"

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
    train_ppo_fge(
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,
            plot_train_x0,
            plot_eval_det,
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        app()
