from typing import Self

import einops as ei
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from og.register_sns_cmaps import register_cmaps

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.level_sampler import LevelSampler, Sampler
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax


def _setup_eval_dist_plot(ax, run_cfg, task, props):
    opts = dict(linewidth=0.8, zorder=0.5, alpha=0.8)
    ax.axvline(run_cfg.task_cfg.other_min_ang_vel, color="green", **opts)
    ax.axvline(run_cfg.task_cfg.other_max_ang_vel, color="green", **opts)
    ax.axhline(run_cfg.task_cfg.other_min_ang_vel, color="green", **opts)
    ax.axhline(run_cfg.task_cfg.other_max_ang_vel, color="green", **opts)
    ax.axvline(
        x=task.task_cpu.reg_to_ang_v(run_cfg.task_cfg.ego_min_vel, task.task_cpu.lane_radii[0]),
        **opts,
    )
    ax.axvline(
        x=task.task_cpu.reg_to_ang_v(run_cfg.task_cfg.ego_max_vel, task.task_cpu.lane_radii[0]),
        **opts,
    )
    ax.axhline(
        y=task.task_cpu.reg_to_ang_v(run_cfg.task_cfg.ego_min_vel, task.task_cpu.lane_radii[1]),
        **opts,
    )
    ax.axhline(
        y=task.task_cpu.reg_to_ang_v(run_cfg.task_cfg.ego_max_vel, task.task_cpu.lane_radii[1]),
        **opts,
    )

    ax.set_xlabel("Car 1 (Inner) Ang. Vel.")
    ax.set_ylabel("Car 2 (Outer) Ang. Vel.")
    # ax.set_aspect('equal', adjustable='box')
    ax.set_aspect("equal")
    offset = (run_cfg.task_cfg.other_max_ang_vel - run_cfg.task_cfg.other_min_ang_vel) / 16
    ax.set_xlim(
        run_cfg.task_cfg.other_min_ang_vel - offset,
        run_cfg.task_cfg.other_max_ang_vel + offset,
    )
    ax.set_ylim(
        run_cfg.task_cfg.other_min_ang_vel - offset,
        run_cfg.task_cfg.other_max_ang_vel + offset,
    )


def _overlay_mask(
    mask,
    ax,
    run_cfg,
    cmap,
    alpha: float = 0.7,
    levels: list[float] | None = None,
    **kwargs,
):
    fig = ax.figure
    mask_imshow = ei.rearrange(mask, "y x -> x y")
    min_angvel, max_angvel = (
        run_cfg.task_cfg.other_min_ang_vel,
        run_cfg.task_cfg.other_max_ang_vel,
    )
    extent = [min_angvel, max_angvel, min_angvel, max_angvel]
    im = ax.imshow(
        mask_imshow,
        extent=extent,
        origin="lower",
        cmap=cmap,
        alpha=alpha,
    )
    cbar = fig.colorbar(im, ax=ax, label=kwargs["label"], location="left")

    if levels is not None:
        lines = ax.contour(
            mask_imshow,
            levels=levels,
            extent=extent,
            origin="lower",
            colors="k",
            linewidths=2,
            zorder=12,
        )
        cbar.add_lines(lines)

    ret_dict = {"cbar": cbar}
    return ret_dict


class VizAccelDubins:
    @classmethod
    def create(cls) -> Self:
        register_cmaps()
        return VizAccelDubins()

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        # Visualize the x0 in the buffer.
        sampler: Sampler = props.extra["sampler"]
        level_sampler: LevelSampler = props.extra["level_sampler"]
        task: DubinsJax = props.task

        import pdb; pdb.set_trace()

        trajsaver = props.trajsaver
        if trajsaver is None:
            return

        rollouts_eval = props.rollout_eval_det
        task: DubinsJax = props.task

        writer = props.writer
        writer.add_scalar(
            "Eval/SuccessRate",
            np.mean([int(traj.T_trunc[-1]) for traj in rollouts_eval]),
            n_collects,
        )
        writer.add_scalar(
            "Eval/TrajLenDet",
            np.mean([traj.T_rew.shape[0] for traj in rollouts_eval]),
            n_collects,
        )

        return

        # --------------------------------------------

        # Get the x0s from the buffer.
        size = int(sampler["size"])
        b_scores = np.array(sampler["scores"])[:size]

        # Manually compute the rank weights.
        b_weights = level_sampler.get_rank_weights_np(b_scores)

        b_leafs = jtu.tree_map(lambda x: np.array(x)[:size], sampler["levels"])
        b_state = task.leaf_to_state(b_leafs)
        b_ic = task.to_icval(b_state)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        axes = axes.flatten()

        # ----------------------------------------
        # 0: Recent reset states.
        ax = axes[0]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        ax.set_title("Recent Reset States")

        # Plot all reset train x0's
        train_x0s = np.array([DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s])
        if len(train_x0s) > 0:
            markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]
            for marker, alpha, z_order in zip(["D", "x"], [0.8, 0.8], [2, 1]):
                idxs = np.where(np.array(markers) == marker)[0]
                color = "black" if marker == "D" else "red"
                facecolor = "none" if marker == "D" else color
                edgecolor = color if marker == "D" else None
                ax.scatter(
                    train_x0s[idxs, 0],
                    train_x0s[idxs, 1],
                    marker=marker,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    s=4**2,
                    zorder=z_order,
                    label="Success" if marker == "D" else "Collide",
                )

        # Vis the eval results here too
        is_safes = np.array([traj.T_trunc[-1] for traj in props.rollout_eval_det], dtype=int)
        is_safes_grid = task.get_contour_grid(is_safes)
        cmap = plt.get_cmap("viridis")
        cbar_props = _overlay_mask(is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1)
        cbar_props["cbar"].set_ticks([0, 1])

        # ----------------------------------------
        # 2: Plot eval results.
        ax = axes[1]
        _setup_eval_dist_plot(ax, run_cfg, task, props)

        T_state_now = [task.leaf_to_state(traj.T_state_now) for traj in props.rollout_eval_det]

        is_safes = np.array([np.sum(traj.T_rew) == 0.0 for traj in props.rollout_eval_det], dtype=int)
        is_safes_grid = task.get_contour_grid(is_safes)
        cmap = plt.get_cmap("viridis")
        cbar_props = _overlay_mask(is_safes_grid, ax, run_cfg, cmap, label="Eval Safe", vmin=0, vmax=1)
        cbar_props["cbar"].set_ticks([0, 1])

        n_safe = np.sum(is_safes)
        n_total = len(is_safes)
        p_safe = n_safe / n_total
        ax.set_title("Eval State Distribution ({}/{}, {:.1%})".format(n_safe, n_total, p_safe))

        # ----------------------------------------
        # 3: Visualize the FARR probability distribution.
        cmap_scatter = sns.color_palette("crest", as_cmap=True)

        ax = axes[2]

        cbar_props = _overlay_mask(
            is_safes_grid,
            ax,
            run_cfg,
            cmap,
            label="Eval Safe",
            vmin=0,
            vmax=1,
            alpha=0.2,
        )

        # Scatter the x0s, color it depending on the probability.
        norm = Normalize(vmin=0, vmax=b_scores.max())
        sc = ax.scatter(b_ic[:, 0], b_ic[:, 1], c=b_scores, zorder=7, norm=norm, cmap=cmap_scatter)
        cbar = fig.colorbar(sc, ax=ax)

        ax.set_title("scores")

        # ----------------------------------------
        # 4: Visualize the FARR probability distribution.
        ax = axes[3]

        cbar_props = _overlay_mask(
            is_safes_grid,
            ax,
            run_cfg,
            cmap,
            label="Eval Safe",
            vmin=0,
            vmax=1,
            alpha=0.2,
        )

        # Scatter the x0s, color it depending on the probability.
        sc = ax.scatter(
            b_ic[:, 0],
            b_ic[:, 1],
            c=b_weights,
            zorder=7,
            cmap=cmap_scatter,
        )
        cbar = fig.colorbar(sc, ax=ax)

        ax.set_title("Max: {}".format(b_weights.max()))
        # ----------------------------------------

        plot_dir = run_cfg.paths.eval_plots / "accel"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"accel_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
