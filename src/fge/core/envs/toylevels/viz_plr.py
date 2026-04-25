from typing import Self

import ipdb
import jax
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
from fge.core.algos.plr_sampler import PLRSampler
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics
from fge.core.utils.jax_util import myjit


@struct.dataclass(frozen=False)
class VizPLR:
    task: HopperJax = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: ToyLevelsJax) -> Self:
        register_cmaps()
        return VizPLR(task)

    @myjit
    def eval_nn(self, ppo: IntrinsicSumPPO):
        get_Vl = ppo.Vl_ext.apply
        b_x, b_obs, _ = self.task.get_eval_contour()
        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        return b_x, b_Vl

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        if "plr" not in props.extra:
            return

        plr: PLRSampler = props.extra["plr"]
        ppo = props.ppo
        task = props.task

        b_x, b_Vl = jax2np(self.eval_nn(ppo))

        b_x_rollouteval: np.ndarray = task.eval_ics()
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        # -------------------------------------------
        nrow = 7
        figsize = np.array([8.0, 2.5 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")
        [task.task_cpu.label_ic(ax) for ax in axes]

        axes_Vl = axes[5]
        axes_trajlen = axes[6]

        cmap = plt.get_cmap("flare")

        b_plr_ic = plr.b_ic[plr.b_valid]
        b_score = plr.b_scores[plr.b_valid]
        b_score_weight = plr._score_weights(plr.b_scores)[plr.b_valid]
        b_staleness_weight = plr._staleness_weights(plr.b_staleness)[plr.b_valid]

        b_score_weight = b_score_weight / np.sum(b_score_weight)
        b_staleness_weight = b_staleness_weight / np.sum(b_staleness_weight)

        staleness_coef = plr.cfg.staleness_coef
        b_weight = (
            1 - staleness_coef
        ) * b_score_weight + staleness_coef * b_staleness_weight

        b_weight = b_weight / np.sum(b_weight)

        norm_score = Normalize(vmin=b_score.min(), vmax=b_score.max())
        b_score_colors = cmap(norm_score(b_score))

        norm_score_weight = Normalize(
            vmin=b_score_weight.min(), vmax=b_score_weight.max()
        )
        b_score_weight_colors = cmap(norm_score_weight(b_score_weight))

        norm_stale_weight = Normalize(
            vmin=b_staleness_weight.min(), vmax=b_staleness_weight.max()
        )
        b_stale_weight_colors = cmap(norm_stale_weight(b_staleness_weight))

        norm_weight = Normalize(vmin=b_weight.min(), vmax=b_weight.max())
        b_weight_colors = cmap(norm_weight(b_weight))

        vlines_opt = dict(linewidths=0.5, alpha=0.5)

        # -------------------------------
        ax = axes[0]
        ax.set_title("Score")

        # Lollipop plot.
        ax.vlines(b_plr_ic, 0, b_score, colors=b_score_colors, **vlines_opt)
        ax.scatter(b_plr_ic, b_score, color=b_score_colors, s=3)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_score, cmap=cmap), ax=ax)

        # ---------------------------------
        ax = axes[1]
        ax.set_title("Score weight")

        # Lollipop plot.
        ax.vlines(
            b_plr_ic, 0, b_score_weight, colors=b_score_weight_colors, **vlines_opt
        )
        ax.scatter(b_plr_ic, b_score_weight, color=b_score_weight_colors, s=3)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_score_weight, cmap=cmap), ax=ax
        )

        # ---------------------------------
        ax = axes[2]
        ax.set_title("Staleness weight")

        # Lollipop plot.
        ax.vlines(
            b_plr_ic, 0, b_staleness_weight, colors=b_stale_weight_colors, **vlines_opt
        )
        ax.scatter(b_plr_ic, b_staleness_weight, color=b_stale_weight_colors, s=3)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_stale_weight, cmap=cmap), ax=ax
        )

        # ---------------------------------
        ax = axes[3]
        ax.set_title("Total Weight")

        # Lollipop plot.
        ax.vlines(b_plr_ic, 0, b_weight, colors=b_weight_colors, **vlines_opt)
        ax.scatter(b_plr_ic, b_weight, color=b_weight_colors, s=3)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_weight, cmap=cmap), ax=ax)

        # ---------------------------------
        # Do a scatter plot + histogram of the most recent argmaxes (of the score).
        ax = axes[4]
        ax.set_title("Argmax ICs")

        b_argmax_ics = np.array(plr.argmax_ics)
        bins = task.icval_bins(51)
        ax.hist(b_argmax_ics, color="C1", bins=bins, histtype="bar")

        ylim = ax.get_ylim()

        def rescale(x):
            # [0, 1] -> [minval, maxval]
            return x * (ylim[1] - ylim[0]) + ylim[0]

        rng = np.random.default_rng(seed=12345)
        b_jitter = rng.uniform(low=-0.05, high=0.0, size=len(b_argmax_ics))
        ax.scatter(b_argmax_ics, rescale(b_jitter), color="C1", s=3)

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
        name = "plr"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)

        # ------------------------
        # Clear the argmax ics.
        plr.argmax_ics = []
