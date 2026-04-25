from typing import Annotated

import cyclopts
import jax
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
        resetbuf_cfg: ResetBuf.Cfg = ResetBuf.Cfg(
            p_explore=0.8, p_base=0.02, p_predci=0.08
        ),
        ci_classify_cfg: Annotated[
            CIClassifier.Cfg, Parameter(name="cicls", group="pol classifier")
        ] = CIClassifier.Cfg(),
        pol_classify_cfg: Annotated[
            CIClassifier.Cfg, Parameter(name="polcls", group="pol classifier")
        ] = CIClassifier.Cfg(),
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 5e-3,
        n_env: int = 1024,
        n_batch: int = 30,
        pol_lr: float = 5e-5,
        val_lr: float = 8e-4,
        normalize_adv: bool = True,
        seed: int = 12345,
        value_opt: ValueOPT = "NSF",
        value_lims_epsilon: float | None = 1e-1,
        n_update_epochs: int = 1,
        train_nsf_mode: TrainNSFOPT = f"b_x0",
        n_steps: int = 50_000,
        pv0: tuple[float, float] | None = None,
        rehearsal_mode: RehearsalOPT = "polsafe",  # "polsafe", "Vl"
        rehearsal_only_frac: float | None = None,
        use_nsf_explore: int = 1,
        ci_clsfy_frac: float | None = None,
        save_all: bool = False,
        use_nsf_ci: bool = False,
        nsf_ci_quantile: float = 0.05,
):
    jax.config.update("jax_default_matmul_precision", "highest")
    enable_compilation_cache()

    logger.critical(
        "p_explore: {} | p_base: {} | p_predci: {}".format(
            resetbuf_cfg.p_explore, resetbuf_cfg.p_base, resetbuf_cfg.p_predci
        )
    )

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
        use_nsf_explore=bool(use_nsf_explore),
        ci_clsfy_fracci=ci_clsfy_frac,
        use_nsf_ci=use_nsf_ci,
        nsf_ci_quantile=nsf_ci_quantile,
    )
    nsf_cfg = NSF.Cfg()
    nsf_ci_cfg = NSF.Cfg(handle_std0=True)
    algo_cfg = PPOFGECfg(
        ppo=ppo_cfg,
        nsf=nsf_cfg,
        nsf_ci=nsf_ci_cfg,
        buf=buf_cfg,
        ci_classify=ci_classify_cfg,
        pol_classify=pol_classify_cfg,
    )

    task_cfg = dub_circ_jax.TaskCfg(max_timesteps=1_000, pv0=pv0)

    train_cfg = TrainCfg(
        n_steps=n_steps, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
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
    run_cfg.save_all = save_all
    train_ppo_fge(
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
