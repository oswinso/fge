import itertools
import math
import pickle

import colorcet as cc
import einops as ei
import ipdb
import matplotlib.colors as mcolors
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from tqdm import tqdm, trange

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax


def plot_base_reset_dist(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollout_eval_det
    trajsaver = props.trajsaver
    writer = props.writer
    task = props.task

    # Common variables.
    b_ic_train = np.array(
        [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s]
    )

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    ax.set_title("Base Reset Distribution")
    _setup_eval_dist_plot(ax, run_cfg, task, props)

    # Plot density of pvs over the grid
    lb, ub = run_cfg.task_cfg.other_min_ang_vel, run_cfg.task_cfg.other_max_ang_vel
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = (
        task.task_cpu.other_vel_dist.pdf(np.column_stack([X.ravel(), Y.ravel()]))
        .prod(-1)
        .reshape(X.shape)
    )
    ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.5)

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

    name = "base_reset_dist"
    plot_dir = run_cfg.paths.eval_plots / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
    fig.set_size_inches(20, 10)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


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
    norm = None
    if "vmin" and "vmax" in kwargs:
        vmin, vmax = kwargs["vmin"], kwargs["vmax"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        mask_imshow,
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
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


def _plot_eval_results_small(ax, run_cfg, task, props):
    """
    Assumes few enough iconds to visualize this way
    """
    rollouts_eval = props.rollout_eval_det
    b_px0 = task.eval_xs()  # The eval iconds

    if len(rollouts_eval) > 50:
        logger.info("Too many eval rollouts to plot. Skipping.")
        return

    for px0_o1, px0_o2 in b_px0:
        ax.axvline(x=px0_o1, color="orange", linestyle="--", alpha=0.1)
        ax.axhline(y=px0_o2, color="orange", linestyle="--", alpha=0.1)

    # Plot evaluation results (diamond if safe, x if unsafe)
    is_safes = [traj.T_trunc[-1] for traj in rollouts_eval]
    markers = ["D" if is_safe else "x" for is_safe in is_safes]
    colors = ["green" if is_safe else "red" for is_safe in is_safes]
    for m, c in list(itertools.product(["D", "x"], ["green", "red"])):
        idxs = np.where((np.array(markers) == m) & (np.array(colors) == c))[0]
        if len(idxs) > 0:
            ax.scatter(
                b_px0[idxs, 0],
                b_px0[idxs, 1],
                marker=m,
                color=c,
                s=100,
                linewidths=2,
                zorder=3,
            )


def _setup_eval_dist_plot(ax, run_cfg, task, props):
    opts = dict(linewidth=0.8, zorder=0.5, alpha=0.8)
    ax.axvline(run_cfg.task_cfg.other_min_ang_vel, color="green", **opts)
    ax.axvline(run_cfg.task_cfg.other_max_ang_vel, color="green", **opts)
    ax.axhline(run_cfg.task_cfg.other_min_ang_vel, color="green", **opts)
    ax.axhline(run_cfg.task_cfg.other_max_ang_vel, color="green", **opts)
    ax.axvline(
        x=task.task_cpu.reg_to_ang_v(
            run_cfg.task_cfg.ego_min_vel, task.task_cpu.lane_radii[0]
        ),
        **opts,
    )
    ax.axvline(
        x=task.task_cpu.reg_to_ang_v(
            run_cfg.task_cfg.ego_max_vel, task.task_cpu.lane_radii[0]
        ),
        **opts,
    )
    ax.axhline(
        y=task.task_cpu.reg_to_ang_v(
            run_cfg.task_cfg.ego_min_vel, task.task_cpu.lane_radii[1]
        ),
        **opts,
    )
    ax.axhline(
        y=task.task_cpu.reg_to_ang_v(
            run_cfg.task_cfg.ego_max_vel, task.task_cpu.lane_radii[1]
        ),
        **opts,
    )

    ax.set_xlabel("Car 1 (Inner) Ang. Vel.")
    ax.set_ylabel("Car 2 (Outer) Ang. Vel.")
    # ax.set_aspect('equal', adjustable='box')
    ax.set_aspect("equal")
    offset = (
        run_cfg.task_cfg.other_max_ang_vel - run_cfg.task_cfg.other_min_ang_vel
    ) / 16
    ax.set_xlim(
        run_cfg.task_cfg.other_min_ang_vel - offset,
        run_cfg.task_cfg.other_max_ang_vel + offset,
    )
    ax.set_ylim(
        run_cfg.task_cfg.other_min_ang_vel - offset,
        run_cfg.task_cfg.other_max_ang_vel + offset,
    )


def _plot_traj(n_collects: int, run_cfg: RunCfg, props: EvalProps, ax, rollout):
    task: DubinsJax = props.task

    t_state_now = rollout.T_state_now
    terms = np.concatenate([[False], rollout.T_term])
    truncs = np.concatenate([[False], rollout.T_trunc])
    steps, _, _, ego_states, other_states = t_state_now
    T = len(steps)
    plot_every = 6
    ts = np.unique([i for i in range(0, T, plot_every)] + [-1])
    ts = ts[-20:]  # Plot last few timesteps of interest

    e_cmap, o_cmap = cc.cm.rainbow, cc.cm.gray
    T_frac = np.linspace(0, 1, T)
    o_T_frac = np.linspace(0.2, 0.8, T)

    # If success, label and skip
    if truncs[-1]:
        return

    task.task_cpu.state = ego_states[-1]
    task.task_cpu.other_cars = other_states[-1]
    task.task_cpu.timestep = steps[-1]
    task.task_cpu.setup_trajplot(ax, term=rollout.T_term[-1])
    for t in tqdm(ts, leave=False, desc="Plotting trajs"):
        # Set up cars
        states = [ego_states[t]]
        if run_cfg.task_cfg.num_vehicles > 0:
            states = np.concatenate([states, other_states[t]])
        for i in trange(len(states), leave=False):
            x, y, theta, v = states[i]
            if i == 0:
                # Ego agent uses regular velocity
                dx = 0.4 * v * np.cos(theta)
                dy = 0.4 * v * np.sin(theta)
            else:
                # Other cars have angular velocity
                reg_v = (
                    v
                    * (
                        task.task_cpu.track_inner_radius
                        + task.task_cpu.track_outer_radius
                    )
                    / 2
                )
                dx = 0.4 * reg_v * np.cos(theta)
                dy = 0.4 * reg_v * np.sin(theta)

            color = e_cmap(T_frac[t]) if i == 0 else o_cmap(1 - o_T_frac[t])
            if terms[t]:
                if i == 0:
                    color = "purple"
                elif (
                    np.linalg.norm(states[i][:2] - states[0][:2])
                    < 2 * run_cfg.task_cfg.vehicle_radius
                ):
                    color = "maroon"

            body = Circle(
                (states[i][0], states[i][1]),
                run_cfg.task_cfg.vehicle_radius,
                color=color,
                fill=True,
                zorder=i,
                alpha=1 if terms[t] or truncs[t] else 0.25,
            )
            arrow = FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle="->",
                mutation_scale=10,
                color="black" if terms[t] else color,
                zorder=len(states) - i,
                alpha=1 if terms[t] or truncs[t] else 0.25,
            )
            ax.add_patch(body)
            ax.add_patch(arrow)


def plot_eval_rollouts(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    """
    Don't call if more than 25 eval rollouts
    """
    rollouts_eval = props.rollout_eval_det
    task: DubinsJax = props.task

    # Set up rendering
    task.task_cpu.reset()
    task.task_cpu.render()
    n_xs = len(rollouts_eval)
    b_px0 = task.eval_xs()  # The eval iconds
    assert len(b_px0) == n_xs

    # TODO: Which iconds do you want to plot?
    # import pdb; pdb.set_trace()

    nrow = ncol = math.isqrt(b_px0.shape[0])
    if nrow * ncol < n_xs:
        ncol = math.ceil(n_xs / nrow)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(nrow * 4, ncol * 4), layout="constrained"
    )
    if nrow * ncol == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_i, rollout in tqdm(
        enumerate(rollouts_eval), total=n_xs, desc="Plotting eval rollouts"
    ):
        ax = axes[ax_i]
        _plot_traj(n_collects, run_cfg, props, ax, rollout)

    # Save things
    plot_dir = run_cfg.paths.eval_plots / "eval_trajs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_{n_collects:05d}.jpg"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    data_dir = run_cfg.paths.data_dir / "eval_trajs"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"eval_{n_collects:05d}.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(rollouts_eval, f)


def plot_bottomleft_3x3_eval(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    """
    Just plot the bottom left 3x3 square
    """
    rollouts_eval = props.rollout_eval_det
    task: DubinsJax = props.task

    # Set up rendering
    task.task_cpu.reset()
    task.task_cpu.render()
    n_xs = len(rollouts_eval)
    b_px0 = task.eval_xs()  # The eval iconds
    assert len(b_px0) == n_xs

    nrow = ncol = 3
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(nrow * 4, ncol * 4), layout="constrained"
    )
    if nrow * ncol == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_i, rollout in tqdm(
        enumerate(rollouts_eval), total=n_xs, desc="Plotting eval rollouts"
    ):
        ax = axes[ax_i]
        _plot_traj(n_collects, run_cfg, props, ax, rollout)

    # Save things
    plot_dir = run_cfg.paths.eval_plots / "eval_trajs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_{n_collects:05d}.jpg"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    data_dir = run_cfg.paths.data_dir / "eval_trajs"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"eval_{n_collects:05d}.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(rollouts_eval, f)
