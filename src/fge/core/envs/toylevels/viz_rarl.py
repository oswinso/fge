from typing import Self

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from loguru import logger
from matplotlib.colors import Normalize
from og.jax_utils import jax2np, jax_vmap
from og.register_sns_cmaps import register_cmaps
from og.tree_utils import tree_stack

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.sumppo_x0 import SumPPOX0, X0ResetBuf
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics
from fge.core.utils.jax_util import myjit


@struct.dataclass(frozen=False)
class VizRARL:
    task: HopperJax = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: ToyLevelsJax) -> Self:
        register_cmaps()
        return VizRARL(task)

    @myjit
    def eval_nn(self, ppo: IntrinsicSumPPO, ppo_x0: SumPPOX0):
        get_Vl = ppo.Vl_ext.apply
        b_x, b_obs, _ = self.task.get_eval_contour()
        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)

        # Sample a bunch of random states to visualize the distribution.
        b_key = jr.split(jr.PRNGKey(12345), 128)
        b_state = jax.vmap(ppo_x0.sample_x0)(b_key)
        b_ic_x0 = self.task.to_icval(b_state)

        return b_x, b_Vl, b_ic_x0

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        if "ppo_x0" not in props.extra:
            return

        ppo_x0: SumPPOX0 = props.extra["ppo_x0"]
        ppo = props.ppo
        task = props.task

        b_x, b_Vl, b_ic_x0 = jax2np(self.eval_nn(ppo, ppo_x0))

        b_x_rollouteval: np.ndarray = task.eval_ics()
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        # -------------------------------------------
        nrow = 3
        figsize = np.array([8.0, 2.5 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")
        [task.task_cpu.label_ic(ax) for ax in axes]

        axes_Vl = axes[1]
        axes_trajlen = axes[2]

        # -------------------------------
        ax = axes[0]
        ax.set_title("Policy")

        # Visualize the policy as histogram and scatter plot.
        bins = task.icval_bins(51)
        ax.hist(b_ic_x0, color="C1", bins=bins, histtype="bar", density=True)
        ylim = ax.get_ylim()

        def rescale(x):
            # [0, 1] -> [minval, maxval]
            return x * (ylim[1] - ylim[0]) + ylim[0]

        rng = np.random.default_rng(seed=12345)
        b_jitter = rng.uniform(low=-0.05, high=-0.01, size=len(b_ic_x0))
        ax.scatter(b_ic_x0, rescale(b_jitter), color="C1", s=3)

        # ---------------------------------
        ax = axes_Vl
        ax.set_title("Vl")

        ax.plot(b_x_rollouteval, b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, b_Vl_disc_det, color="C5", label="Vl MC (det)")
        ax.plot(b_x, b_Vl, color="C1", alpha=0.9, label="Vl")

        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ---------------------------------
        ax = axes_trajlen
        ax.set_title("Steps Alive")
        ax.plot(b_x_rollouteval, b_T_rng, marker="o", ms=3, lw=0.9, color="C4")
        ax.plot(b_x_rollouteval, b_T_det, marker="o", ms=3, lw=0.9, color="C5")

        # ------------------------------------------
        name = "rarl"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
