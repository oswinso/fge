from typing import Annotated as Anno

import cyclopts
import ipdb
from cyclopts import Parameter

from fge.core.algos.buf_custom import BufCustom, RehearsalOPT, TrainNSFOPT, ValueOPT
from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_fge import PPOFGECfg, train_ppo_fge
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.common.reset_buf import ResetBuf
from fge.core.envs.kinetix import lander
from fge.core.envs.kinetix.eval_fns import VizCustomLander, plot_train_x0_lander
from fge.core.envs.kinetix.viz_eval import save_traj_video
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
        resetbuf_cfg: ResetBuf.Cfg = ResetBuf.Cfg(p_explore=0.8, p_base=0.02, p_predci=0.08),
        ci_classify_cfg: Anno[CIClassifier.Cfg, Parameter(name="cicls", group="pol classifier")] = CIClassifier.Cfg(
            n_sample=8192, n_batches=8, epochs=1
        ),
        pol_classify_cfg: Anno[CIClassifier.Cfg, Parameter(name="polcls", group="pol classifier")] = CIClassifier.Cfg(
            n_sample=8192, n_batches=8, epochs=1
        ),
        nsf_cfg: Anno[NSF.Cfg, Parameter(name="nsf", group="nsf")] = NSF.Cfg(),
        nsf_ci_cfg: Anno[NSF.Cfg, Parameter(name="nsf", group="nsf_ci")] = NSF.Cfg(),
        use_wandb: bool = False,
        name: str | None = None,
        ent_cf: float = 1e-2,
        n_env: int = 512,
        n_batch: int = 8,
        rollout_T: int = 16,
        normalize_adv: bool = True,
        value_opt: ValueOPT = "NSF",
        value_lims_epsilon: float | None = 1e-1,
        pol_lr: float = 4e-4,
        val_lr: float = 8e-4,
        n_update_epochs: int = 1,
        # ---
        train_nsf_mode: TrainNSFOPT = "b_x0",
        use_nsf_explore: bool = True,
        ci_clsfy_frac: float | None = None,
        # -----
        n_steps: int = 5_500,
        rehearsal_mode: RehearsalOPT = "polsafe",
        rehearsal_only_frac: float | None = None,
        seed: int = 12345,
        eval_every: int = 100,
        log_every: int = 20,
        hids: list[int] = [256, 256],
):
    enable_compilation_cache()

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

    task_cfg = lander.TaskCfg()

    train_cfg = TrainCfg(
        n_steps=n_steps,
        log_every=log_every,
        fastcb_every=20,
        eval_every=eval_every,
        save_every=1_000,
    )
    col_cfg = CollectorCfg(n_envs=n_env)
    eval_cfg = EvalCfg(n_seeds=1)
    task_name = "Lander"

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
    env = lander.LanderJax(run_cfg.task_cfg)
    train_ppo_fge(
        run_cfg=run_cfg,
        seed=seed,
        eval_cbs=[
            log_train,
            log_eval,
            log_eval_det,
            plot_train_x0_lander,
            save_traj_video,
            VizCustomLander.create(env),
            clear_trajs,
        ],
        fast_cbs=[clear_trajs],
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
