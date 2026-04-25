import colorcet as cc
import numpy as np
from jax import tree_util as jtu
from loguru import logger
from matplotlib import pyplot as plt

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.envs.mujoco.hopper.dm_hopper_artist import DMHopperArtist
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax


def plot_train(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: HopperJax = props.task

    if trajsaver is None:
        return

    n_trajs = len(trajsaver.trajs)
    if n_trajs == 0:
        return

    n_plot = min(n_trajs, 11)

    nrow = n_plot
    figsize = np.array([8.0, 2.0 * nrow])
    fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")

    draw_every = 4

    # Plot a few trajectories from the last time the buffer was cleared.
    for ii, ax in enumerate(axes):
        task.setup_trajplot(ax)

        # Plot from most recent.
        traj = trajsaver.trajs[-ii]
        Tp1_state: HopperJax.MinState = task.leaf_to_minstate(traj.Tp1_state)
        ic = Tp1_state.px0[0]
        Tp1_qpos = Tp1_state.qpos
        assert Tp1_qpos.ndim == 2

        T = len(Tp1_state.step)
        T_frac = np.linspace(0.0, 1.0, T)
        cmap = cc.cm.rainbow
        idxs = list(range(T))[::draw_every]
        if T - 1 not in idxs:
            idxs.append(T - 1)

        Tp1_px = Tp1_qpos[:, 0]
        Tp1_py = Tp1_qpos[:, 1]

        for jj in idxs:
            frac, qpos = T_frac[jj], Tp1_qpos[jj]
            color = cmap(frac)
            hopper = DMHopperArtist(qpos, facecolor=color, zorder=4)
            ax.add_artist(hopper)

        # Plot the CoM.
        ax.plot(Tp1_px, Tp1_py + 1, color="k", linewidth=1.0, alpha=0.7, zorder=5)

        rew_sum = np.sum(traj.T_rew)
        is_safe = rew_sum == 0.0

        title_color = "C5" if is_safe else "C0"
        ax.set_title("px0: {:.1f}  Length: {}".format(ic, T), color=title_color)
        ax.set_aspect("equal")

    plot_dir = run_cfg.paths.train_plots / "traj"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"train_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def plot_train_x0(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: HopperJax = ppo.task

    if trajsaver is None:
        return

    if len(trajsaver.trajs) == 0:
        return

    b_ic = []
    for traj in trajsaver.trajs:
        x0: HopperJax.MinState = task.leaf_to_minstate(
            jtu.tree_map(lambda x: x[0], traj.T_state_now)
        )
        b_ic.append(task.to_icval(x0))
    b_ic = np.array(b_ic)

    bins = task.icval_bins()
    fig, ax = plt.subplots(layout="constrained")
    task.task_cpu.label_ic(ax)
    ax.hist(b_ic, color="C1", bins=bins)

    plot_dir = run_cfg.paths.train_plots / "x0hist"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"x0hist_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)

def plot_train_x0_extended(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: HopperJax = ppo.task

    if trajsaver is None:
        return

    if len(trajsaver.trajs) == 0:
        return

    b_ic = []
    for traj in trajsaver.trajs:
        x0: HopperJax.MinState = task.leaf_to_minstate(
            jtu.tree_map(lambda x: x[0], traj.T_state_now)
        )
        b_ic.append(task.to_icval(x0))
    b_ic = np.array(b_ic)

    bins = task.icval_bins()

    # Smooth out the bins.
    # 2 bins per unit
    nbins: int = int((task.task_cfg.px0_bounds[1] - task.task_cfg.px0_bounds[0]) * 2 + 1)
    bins = np.linspace(
        task.task_cfg.px0_bounds[0],
        task.task_cfg.px0_bounds[1],
        num=nbins,
    )

    logger.critical(f"{b_ic.shape=}")

    fig, ax = plt.subplots(layout="constrained")
    task.task_cpu.label_ic(ax)
    ax.hist(b_ic, color="C1", bins=bins)

    plot_dir = run_cfg.paths.train_plots / "x0hist_extended"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"x0hist_extended_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


def plot_eval_det(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollout_eval_det
    task: HopperJax = props.task

    n_xs = len(rollouts_eval)

    b_ankle0_deg = task.eval_ics()
    assert len(b_ankle0_deg) == n_xs

    n_plot = min(11, n_xs)
    s_idxs = np.round(np.linspace(0, n_xs - 1, n_plot)).astype(int)

    nrow = n_plot
    figsize = np.array([8.0, 2.0 * nrow])
    fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")

    draw_every = 4

    for ii, ax in enumerate(axes):
        task.setup_trajplot(ax)

        eval_idx = s_idxs[ii]

        rollout = rollouts_eval[eval_idx]
        Tp1_state: HopperJax.MinState = task.leaf_to_minstate(rollout.Tp1_state)
        ic = Tp1_state.px0[0]

        T = len(Tp1_state.step)
        T_frac = np.linspace(0.0, 1.0, T)
        cmap = cc.cm.rainbow
        idxs = list(range(T))[::draw_every]
        if T - 1 not in idxs:
            idxs.append(T - 1)

        Tp1_qpos = Tp1_state.qpos
        assert Tp1_qpos.ndim == 2

        Tp1_px = Tp1_qpos[:, 0]
        Tp1_py = Tp1_qpos[:, 1]

        for jj in idxs:
            frac, qpos = T_frac[jj], Tp1_qpos[jj]
            color = cmap(frac)
            hopper = DMHopperArtist(qpos, facecolor=color, zorder=4)
            ax.add_artist(hopper)

        # Plot the CoM.
        ax.plot(Tp1_px, Tp1_py + 1, color="k", linewidth=1.0, alpha=0.7, zorder=5)

        rew_sum = np.sum(rollout.T_rew)
        is_safe = rew_sum == 0.0

        title_color = "C5" if is_safe else "C0"
        ax.set_title("px0: {:.1f}  Length: {}".format(ic, T), color=title_color)
        ax.set_aspect("equal")

    plot_dir = run_cfg.paths.eval_plots / "traj"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
