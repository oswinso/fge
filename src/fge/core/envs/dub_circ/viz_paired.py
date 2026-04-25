from typing import Self

import jax
import jax.random as jr
import numpy as np
from flax import struct
from matplotlib import pyplot as plt

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.sumppo_x0 import SumPPOX0
from fge.core.envs.dub_circ.analyze_ppo import _overlay_mask, _setup_eval_dist_plot
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax
from fge.core.utils.jax_util import myjit


@myjit
def eval_nn(task, ppo_x0: SumPPOX0):
    b_x, b_obs, _ = task.get_eval_contour()

    # Sample a bunch of random states to visualize the distribution.
    b_key = jr.split(jr.PRNGKey(12345), 128)
    b_state = jax.vmap(ppo_x0.sample_x0)(b_key)
    b_ic_x0 = task.to_icval(b_state)

    return b_x, b_ic_x0


@struct.dataclass(frozen=False)
class DubinsJaxVizPAIRED:
    task: DubinsJax = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: DubinsJax) -> Self:
        return DubinsJaxVizPAIRED(task)

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        task = props.task
        ppo_adv_x0 = props.extra["ppo_adv_x0"]
        ppo_pro = props.extra["ppo_pro"]
        ppo_ant = props.extra["ppo_ant"]
        trajsaver_pro = props.extra["trajsaver_pro"]
        trajsaver_ant = props.extra["trajsaver_ant"]
        rollout_eval_det_pro = props.extra["rollout_eval_det_pro"]
        rollout_eval_det_ant = props.extra["rollout_eval_det_ant"]
        regret = props.extra["regret"]
        max_U_A = props.extra["max_U_A"]
        E_U_P = props.extra["E_U_P"]

        # Common variables.
        b_ic_train = np.array(
            [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver_pro.x0s]
        )
        b_ic_source = np.array(
            [DubinsJax.State(*_x0).source for _x0 in trajsaver_pro.x0s], dtype=np.int32
        )

        b_x0_eval, b_x0_obs_eval, _ = task.get_eval_contour()
        b_x0_box_eval = task.box_from_x0(b_x0_eval)

        nrow = 3
        ncol = 3
        figsize = np.array([6.0 * ncol, 4.0 * nrow])
        fig, axes = plt.subplots(
            nrow, ncol, sharex=True, sharey=True, figsize=figsize, layout="constrained"
        )
        axes = axes.flatten()
        for ax in axes:
            ax.set_aspect("equal")

        # ----- 1. Protagonist Eval -----
        ax = axes[0]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in rollout_eval_det_pro], dtype=int
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
            "Pro Eval State Distribution ({}/{}, {:.1%})".format(
                n_safe, n_total, p_safe
            )
        )

        # ----- 2. Antagonist Eval -----
        ax = axes[1]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        is_safes = np.array(
            [traj.T_trunc[-1] for traj in rollout_eval_det_ant], dtype=int
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
            "Ant Eval State Distribution ({}/{}, {:.1%})".format(
                n_safe, n_total, p_safe
            )
        )

        # ----- 3. Regret -----
        ax = axes[2]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        regret_det_grid = task.get_contour_grid(regret)
        cmap = plt.get_cmap("viridis")
        # cbar_props = _overlay_mask(regret_det_grid, ax, run_cfg, cmap, label="regret", vmin=-1, vmax=1)
        # cbar_props["cbar"].set_ticks([-1, 1])
        cbar_props = _overlay_mask(regret_det_grid, ax, run_cfg, cmap, label="regret")
        ax.set_title("Regret ($\mu$={:.2f})".format(np.mean(regret)))

        # ----- 4. Max U_A Det -----
        ax = axes[3]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        max_U_A_det_grid = task.get_contour_grid(max_U_A)
        cmap = plt.get_cmap("viridis")
        # cbar_props = _overlay_mask(max_U_A_det_grid, ax, run_cfg, cmap, label="Max U_A Det", vmin=-1, vmax=0)
        # cbar_props["cbar"].set_ticks([-1, 0])
        cbar_props = _overlay_mask(
            max_U_A_det_grid, ax, run_cfg, cmap, label="Max U_A Det"
        )
        ax.set_title("Max U_A ($\mu$={:.2f})".format(np.mean(max_U_A)))

        # ----- 5. E_U_P Det -----
        ax = axes[4]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        E_U_P_det_grid = task.get_contour_grid(E_U_P)
        cmap = plt.get_cmap("viridis")
        # cbar_props = _overlay_mask(E_U_P_det_grid, ax, run_cfg, cmap, label="E_U_P Det", vmin=-1, vmax=0)
        cbar_props = _overlay_mask(E_U_P_det_grid, ax, run_cfg, cmap, label="E_U_P Det")
        # cbar_props["cbar"].set_ticks([-1, 0])
        ax.set_title("Expected U_P ($\mu$={:.2f})".format(np.mean(E_U_P)))

        # ----- 6. Action Distribution of Adversary -----
        ax = axes[5]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        ppo_adv_x0_control_probs = ppo_adv_x0.get_x0_control_grid_probs(
            mesh_grid=b_x0_box_eval
        )
        ppo_adv_x0_control_probs_grid = task.get_contour_grid(ppo_adv_x0_control_probs)
        cmap = plt.get_cmap("viridis")
        cbar_props = _overlay_mask(
            ppo_adv_x0_control_probs_grid, ax, run_cfg, cmap, label="Control Prob"
        )
        ax.set_title("Adversary Control Distribution (prob)")

        # ----- 7. Plot action distribution of Adversary with recent reset x0s overlayed -----
        # Assumes this comes after the Action Distribution of Adversary plot.
        ax = axes[6]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        cmap = plt.get_cmap("viridis")
        cbar_props = _overlay_mask(
            ppo_adv_x0_control_probs_grid,
            ax,
            run_cfg,
            cmap,
            label="Control Prob",
            vmin=0,
            vmax=1,
        )
        cbar_props["cbar"].set_ticks([0, 1])
        ax.set_title("Recent Reset x0s")
        # Plot all reset train x0's
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver_pro.trajs]
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

        # -----------------------------------------------------
        name = "paired"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.set_size_inches(20, 10)
        fig.savefig(fig_path, dpi=400)
        plt.close(fig)
