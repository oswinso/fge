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
from fge.core.algos.trajsaver import TrajSaver
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.f16_avoid.f16_avoid_jax import IC, F16AvoidJax
from fge.core.envs.f16_avoid.f16_plot_utils import plot_grid
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics
from fge.core.utils.jax_util import myjit


class VizCustomF16(struct.PyTreeNode):
    task: F16AvoidJax = struct.field(pytree_node=False)

    x0_ci_idx_last: int = struct.field(pytree_node=False)

    buf_ci_id_last: int = struct.field(pytree_node=False)
    buf_explore_id_last: int = struct.field(pytree_node=False)

    set_explore_zero: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: F16AvoidJax, set_explore_zero: bool = False) -> Self:
        return VizCustomF16(task, 0, 0, 0, set_explore_zero)

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        if "buf" not in props.extra:
            logger.error("No buf in props.extra")
            return
        if "ci_classify" not in props.extra:
            logger.error("No ci_classify in props.extra")
            return

        buf: BufCustom = props.extra["buf"]
        ppo = props.ppo
        nsf: NSF = props.extra["nsf"]
        vds: VDSGAE = props.extra["vds"]
        ci_classify: CIClassifier = props.extra["ci_classify"]
        pol_classify: CIClassifier = props.extra["pol_classify"]
        task = props.task
        info = props.extra["custom"]
        trajsaver = props.trajsaver

        eval_metrics = compute_rollout_metrics(props)

        # Plot the sampling distribution, colored by source.
        self.plot_sampling_dist(n_collects, run_cfg, trajsaver)

        # Plot the evaluation result.
        self.plot_eval_results(n_collects, run_cfg, eval_metrics)

    def plot_sampling_dist(
        self, n_collects: int, run_cfg: RunCfg, trajsaver: TrajSaver
    ):
        def grid_cb(xytup: tuple[int, int], vt: float, conedist: float, ax: plt.Axes):
            ii, jj = xytup
            vt_ = task.b_vt[ii]
            cd_ = task.b_cd[jj]

            assert np.allclose(vt_, vt)
            assert np.allclose(cd_, conedist)

            b_valid = (b_cd_idx == ii) & (b_vt_idx == jj)
            b_ic_ = b_ic[b_valid]
            assert b_ic_.shape[1:] == (4,)

            for s in Source:
                b_ic_s = b_ic_[b_source[b_valid] == s]

                b_conephi = b_ic_s[:, IC.CONEPHI]
                b_aspect = b_ic_s[:, IC.ASPECT]
                b_aspect_deg = np.rad2deg(b_aspect)

                color = colors_dict.get(s, "C4")

                ax.scatter(b_conephi, b_aspect_deg, c=color, s=5, alpha=0.9)

        def fig_cb(fig: plt.Figure):
            # Legend.
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    lw=0,
                    label=labels[ii],
                    mec=color,
                    mfc=color,
                    markersize=10,
                )
                for ii, color in enumerate(colors)
            ]
            fig.legend(
                handles=legend_handles,
                loc="outside right upper",
                borderaxespad=0,
                frameon=False,
            )

        source_labels = Source.labels_dict()
        colors_dict = Source.colors_dict()

        task = self.task

        b_x0 = task.leaf_to_minstate(tree_stack(trajsaver.all_x0s(), axis=0, which=np))
        b_ic = task.to_icval(b_x0)
        b_source = b_x0.source

        # # Separate b_ic by source.
        # b_ic_list = [b_ic[b_source == k] for k in Source]
        colors = [colors_dict.get(k, "C4") for k in Source]
        labels = [source_labels.get(k, "Unknown") for k in Source]

        # Separate b_ic by the closest (cd, vt) pair.
        b_cd_idx = np.digitize(b_ic[:, IC.CD], task.b_cd_edge) - 1
        b_vt_idx = np.digitize(b_ic[:, IC.VT], task.b_vt_edge) - 1

        fig, subfigs = plot_grid(grid_cb, fig_cb)
        fig.suptitle("Sampling Distribution")

        name = "sampledist"
        plot_dir = run_cfg.paths.train_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)

    def plot_eval_results(self, n_collects: int, run_cfg: RunCfg, eval_metrics):
        def grid_cb(xytup: tuple[int, int], vt: float, conedist: float, ax: plt.Axes):
            ii, jj = xytup

            idx = ii * self.task.n_cd_regions + jj
            s_T_det = bs_T_det[idx]
            assert s_T_det.shape == (self.task.n_eval_pts_per_slice,)

            # Separate into bins based on the length of the trajectory.
            s_bin = np.digitize(s_T_det, boundaries) - 1
            s_colors = [colors[b] for b in s_bin]

            ax.scatter(
                self.task.s_conephi, self.task.s_aspect_deg, c=s_colors, s=5, alpha=0.9
            )

        def fig_cb(fig: plt.Figure):
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    lw=0,
                    label=bin_names[ii],
                    mec=color,
                    mfc=color,
                    markersize=10,
                )
                for ii, color in enumerate(colors)
            ]
            fig.legend(
                handles=legend_handles,
                loc="outside right upper",
                borderaxespad=0,
                frameon=False,
            )

        boundaries = [0, 100, 800, 2000]
        bin_names = [
            f"{boundaries[i]} - {boundaries[i + 1] - 1}"
            for i in range(len(boundaries) - 1)
        ]
        colors = ["C0", "C2", "C1"]

        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = eval_metrics
        assert b_T_det.shape == (self.task.n_eval_pts_total,)

        # Reshape back to (n_slices, n_pts_per_slice).
        bs_T_det = b_T_det.reshape(self.task.n_slices, self.task.n_eval_pts_per_slice)

        fig, subfigs = plot_grid(grid_cb, fig_cb)
        fig.suptitle("Eval (Det)")

        name = "evaldet"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
