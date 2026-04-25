import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from og.jax_utils import jax2np
from og.register_sns_cmaps import register_cmaps

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.state_reset_id import Source
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.dub_circ.analyze_ppo import (
    _overlay_mask,
    _plot_eval_results_small,
    _setup_eval_dist_plot,
)
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics


class VizSFL(struct.PyTreeNode):
    """Visualize the ensemble from SFL."""

    task: DubinsJax = struct.field(pytree_node=False)

    def compute_learnability_landscape(self, run_cfg: RunCfg, props: EvalProps):
        ppo = props.ppo
        buf = props.extra["sfl"]
        collector_eval = props.extra['collector_evals'][0]
        # _, rollout_eval, _ = ppo.collect_eval_w_col(collector_eval)
        _, rollout_eval, _ = ppo.collect_eval_w_col_spec_T(collector_eval,
                                                                      T=buf.buffer.task.eval_rollout_T * buf.cfg.T_coef)
        jax.copy_to_host_async(rollout_eval)
        num_term = jnp.sum(rollout_eval.T_term, axis=-1)
        num_trunc = jnp.sum(rollout_eval.T_trunc, axis=-1)
        p = num_trunc / (num_term + num_trunc)
        learnability = p * (1 - p)
        print(f"{num_term.sum()=} ; {num_trunc.sum()=} ; {p=} ; {learnability.sum()=}")
        return learnability

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()

        buf = props.extra["sfl"]
        task: DubinsJax = props.task
        ppo: SumPPO = props.ppo
        trajsaver = props.trajsaver
        collector_sfl = props.extra["collector_sfl"]

        # ---------------------------------------------------
        # Common variables.
        b_ic_train = np.array(
            [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s]
        )
        b_ic_source = np.array(
            [DubinsJax.State(*_x0).source for _x0 in trajsaver.x0s], dtype=np.int32
        )
        # ---------------------------------------------------

        # Compute learnability landscape
        learnabilities = self.compute_learnability_landscape(run_cfg, props)

        nrow = 3
        ncol = 4
        figsize = np.array([6.0 * ncol, 4.0 * nrow])
        fig, axes = plt.subplots(
            nrow, ncol, sharex=True, sharey=True, figsize=figsize, layout="constrained"
        )
        axes = axes.flatten()
        for ax in axes:
            ax.set_aspect("equal")

        # ----- 1. Eval States -----
        ax = axes[0]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in props.rollout_eval_det], dtype=int
        )
        is_safes_grid = task.get_contour_grid(is_safes)
        cmap = plt.get_cmap("viridis")
        cbar_props = _overlay_mask(
            is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1
        )
        cbar_props["cbar"].set_ticks([0, 1])

        n_safe = np.sum(is_safes)
        n_total = len(is_safes)
        p_safe = n_safe / n_total
        ax.set_title(
            "Eval State Distribution ({}/{}, {:.1%})".format(n_safe, n_total, p_safe)
        )

        # ----- 2. Recent reset states (all) -----
        ax = axes[1]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Recent Reset States")
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in props.rollout_eval_det], dtype=int
        )
        is_safes_grid = task.get_contour_grid(is_safes)
        cbar_props = _overlay_mask(
            is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1
        )
        cbar_props["cbar"].set_ticks([0, 1])
        # Plot all reset train x0's
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]
        for marker, alpha, z_order in zip(["D", "x"], [0.7, 0.7], [2, 1]):
            idxs = np.where(np.array(markers) == marker)[0]
            color = "black" if marker == "D" else "red"
            facecolor = "none" if marker == "D" else color
            edgecolor = color if marker == "D" else None
            ax.scatter(
                b_ic_train[idxs, 0],
                b_ic_train[idxs, 1],
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                s=4**2,
                zorder=z_order,
                label="Success" if marker == "D" else "Collide",
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----- 3. Recent reset states (BASE) -----
        ax = axes[2]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Recent Reset States (BASE)")
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in props.rollout_eval_det], dtype=int
        )
        is_safes_grid = task.get_contour_grid(is_safes)
        cbar_props = _overlay_mask(
            is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1
        )
        cbar_props["cbar"].set_ticks([0, 1])
        # Plot all reset train x0's
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]

        # Only consider BASE sources
        markers = np.array(markers)[np.where(b_ic_source == Source.BASE)].tolist()
        for marker, alpha, z_order in zip(["D", "x"], [0.7, 0.7], [2, 1]):
            idxs = np.where(np.array(markers) == marker)[0]
            color = "black" if marker == "D" else "red"
            facecolor = "none" if marker == "D" else color
            edgecolor = color if marker == "D" else None
            train_ics = b_ic_train[np.where(b_ic_source == Source.BASE)]
            ax.scatter(
                train_ics[idxs, 0],
                train_ics[idxs, 1],
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                s=4**2,
                zorder=z_order,
                label="Success" if marker == "D" else "Collide",
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----- 3. Recent reset states (SFL Buffer) -----
        ax = axes[3]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Recent Reset States (SFL Buf)")
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in props.rollout_eval_det], dtype=int
        )
        is_safes_grid = task.get_contour_grid(is_safes)
        cbar_props = _overlay_mask(
            is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1
        )
        cbar_props["cbar"].set_ticks([0, 1])
        # Plot all reset train x0's
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]

        markers = np.array(markers)[np.where(b_ic_source == 2)].tolist()

        for marker, alpha, z_order in zip(["D", "x"], [0.7, 0.7], [2, 1]):
            idxs = np.where(np.array(markers) == marker)[0]
            color = "black" if marker == "D" else "red"
            facecolor = "none" if marker == "D" else color
            edgecolor = color if marker == "D" else None
            train_ics = b_ic_train[np.where(b_ic_source == 2)]
            ax.scatter(
                train_ics[idxs, 0],
                train_ics[idxs, 1],
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                s=4**2,
                zorder=z_order,
                label="Success" if marker == "D" else "Collide",
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----- 4. Learnability Landscape -----
        ax = axes[4]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Learnability landscape + Buf ICs")
        learnability_grid = task.get_contour_grid(learnabilities)
        cbar_props = _overlay_mask(
            learnability_grid, ax, run_cfg, cmap, label="Learnabilities",
        )
        # cbar_props["cbar"].set_ticks([0, 1])

        # Plot all reset train x0's
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]
        markers = np.array(markers)[np.where(b_ic_source == 2)].tolist()
        for marker, alpha, z_order in zip(["D", "x"], [0.7, 0.7], [2, 1]):
            idxs = np.where(np.array(markers) == marker)[0]
            color = "black" if marker == "D" else "red"
            facecolor = "none" if marker == "D" else color
            edgecolor = color if marker == "D" else None
            train_ics = b_ic_train[np.where(b_ic_source == 2)]
            ax.scatter(
                train_ics[idxs, 0],
                train_ics[idxs, 1],
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                s=4**2,
                zorder=z_order,
                label="Success" if marker == "D" else "Collide",
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )


        # -----------------------------------------------------
        name = "sfl"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.set_size_inches(20, 10)
        fig.savefig(fig_path, dpi=400)
        plt.close(fig)
