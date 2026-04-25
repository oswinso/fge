import cyclopts
import jax

from fge.core.algos.onpol.ppo import EvalCfg, RunCfg, TrainCfg
from fge.core.algos.onpol.ppo_sfl import train_ppo_sfl, PPOSFLCfg
from fge.core.algos.sfl_sampler import SFLSampler
from fge.core.bits.collector import CollectorCfg
from fge.core.bits.ppo_core import SumPPO, SumPPOCfg
from fge.core.envs.mujoco.cheetah import cheetah
from fge.core.envs.mujoco.cheetah.cheetah_jax import CheetahJax
from fge.core.envs.toylevels.analyze_ppo import clear_trajs
from fge.core.envs.toylevels.analyze_ppo import (
    log_eval,
    log_eval_det,
    log_train,
)
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
        pol_lr: float = 4e-4,
        val_lr: float = 8e-4,
        normalize_adv: bool = True,
        seed: int = 12345,
        value_lims_epsilon: float | None = 1e-1,
        n_update_epochs: int = 1,
        n_steps: int = 50_000,
        px0_bounds: tuple[float, float] = (-1.5, 2.5),
        debug: bool = False,
):
    if debug:
        jax.config.update("jax_disable_jit", True)

    jax.config.update("jax_default_matmul_precision", "highest")
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
    sfl_cfg = SFLSampler.Cfg(
        update_buf_every=500,  # Update period T. How often to call "collect_learnable_levels".
        p_sample_base=0.5,  # 1 - rho, probability of sampling from the base distribution.
        init_sample_size=5_000,  # N. How many levels to sample before rolling out and getting top K
        sfl_buf_size=200,  # The K in Top K levels in B ranked by learnability
        T_coef=3,  # How many times to multiply with rollout_eval_T to get a non-0/1 success rate
    )
    algo_cfg = PPOSFLCfg(
        ppo=ppo_cfg,
        sfl=sfl_cfg,
    )

    task_cfg = cheetah.TaskCfg(max_steps=1_000)

    train_cfg = TrainCfg(
        n_steps=n_steps, log_every=10, fastcb_every=20, eval_every=200, save_every=1_000
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
    env = CheetahJax(run_cfg.task_cfg)

    train_ppo_sfl(
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
    with launch_ipdb_on_exception():
        app()
