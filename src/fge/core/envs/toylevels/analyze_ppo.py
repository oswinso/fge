from collections import defaultdict

import ipdb
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.state_reset_id import Source
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.tb import log_dict_tb


def plot_train(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: ToyLevelsJax = ppo.task

    if len(trajsaver.trajs) == 0:
        return

    fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)

    # Plot the trajectories from the last time the buffer was cleared.
    for ii, traj in enumerate(trajsaver.trajs):
        Tp1_state: ToyLevelsJax.State = traj.Tp1_state
        Tp1_pos = Tp1_state.pos

        assert Tp1_pos.ndim == 2

        color = f"C{ii}"
        ax.plot(Tp1_pos[:, 0], Tp1_pos[:, 1], color=color, alpha=0.5)
        # Mark the start.
        ax.plot(
            Tp1_pos[0, 0],
            Tp1_pos[0, 1],
            marker="s",
            mfc=color,
            mec="none",
            markersize=1.8**2,
        )

    plot_dir = run_cfg.paths.train_plots / "traj"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"train_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    task.task_cpu._fig = None


def log_train(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: ToyLevelsJax = ppo.task

    if trajsaver is None:
        return

    if len(trajsaver.trajs) == 0:
        return

    log_dict = {}

    # Count the proportion of trajectories that were reset in each region.
    region_names = task.region_names

    if 0 < len(region_names) <= 20:
        # Don't log the region names if there are too many.
        region_count = defaultdict(int)

        for ii, traj in enumerate(trajsaver.trajs):
            x0: ToyLevelsJax.State = task.leaf_to_minstate(traj.x0)
            region = x0.reset_region
            region_name = region_names[region]
            region_count[region_name] += 1

        n_total = len(trajsaver.trajs)

        # Log.
        for region_name, count in region_count.items():
            key = "Train/resetFrac/{}".format(region_name)
            log_dict[key] = count / n_total

        # Log the metrics.
        log_dict_tb(props.writer, log_dict, global_step=n_collects)


def plot_train_x0(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: ToyLevelsJax = ppo.task

    if trajsaver is None:
        if not hasattr(plot_train_x0, "__printed_trajsaver_none_warning"):
            logger.warning("trajsaver is None, not plotting x0s.")
        return

    if len(trajsaver.x0s) == 0:
        if not hasattr(plot_train_x0, "__printed_x0_warning"):
            logger.warning("len(trajsaver.x0s) == 0, not plotting x0s.")
        return

    b_pos = []
    b_source = []
    for x0 in trajsaver.x0s:
        # x0: ToyLevelsJax.State = task.leaf_to_minstate(traj.x0)
        x0 = task.leaf_to_minstate(x0)
        b_pos.append(x0.pos)
        b_source.append(x0.source)
    b_pos = np.stack(b_pos, axis=0)
    b_source = np.array(b_source)

    # Plot by source.
    markers = ["x", "o", "^", "s", "D", "v"]

    bin_y_size = 5

    rng = np.random.default_rng(seed=1245)

    fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)
    for ii, source in enumerate(Source):
        b_is_source = b_source == source.value
        s_pos = b_pos[b_is_source]
        color = f"C{ii}"
        marker = markers[ii % len(markers)]

        offset = -(len(Source) * bin_y_size) + bin_y_size * ii  # Slight offset for visibility

        # Jitter the y-position for better visibility.
        s_pos[:, 1] += offset + rng.uniform(low=-bin_y_size / 2, high=bin_y_size / 2, size=len(s_pos))

        ax.scatter(s_pos[:, 0], s_pos[:, 1], alpha=0.5, color=color, marker=marker, label=source.name)

    # Legend outside plot.
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Source")

    plot_dir = run_cfg.paths.train_plots / "x0"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"x0_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    task.task_cpu._fig = None

    # ipdb.set_trace()


def clear_trajs(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    trajsaver = props.trajsaver
    if trajsaver is None:
        return
    trajsaver.clear_trajs()


def log_eval(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollouts_eval
    eval_regions_info = props.eval_region_info

    n_seeds = len(rollouts_eval)
    n_xs = len(rollouts_eval[0])

    # Get info for each run.
    b_metrics = defaultdict(list)

    for ii in range(n_xs):
        metrics = defaultdict(list)

        for jj in range(n_seeds):
            rollout = rollouts_eval[jj][ii]
            rew_sum = np.sum(rollout.T_rew)
            traj_len = len(rollout.T_rew)

            metrics["RewSum"] = rew_sum
            metrics["TrajLen"] = traj_len

        # Take the average.
        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}

        # Append to the batch metrics.
        for k, v in metrics_mean.items():
            b_metrics[k].append(v)

    log_dict = {}

    if len(eval_regions_info) <= 50:
        # Only log per region if there are not too many.
        start_idx = 0
        for region_name, n_eval in eval_regions_info:
            end_idx = start_idx + n_eval

            for k, v in b_metrics.items():
                mean_val = np.mean(v[start_idx:end_idx])
                key = "Eval/{}/{}".format(region_name, k)
                log_dict[key] = mean_val

            start_idx = end_idx

    # Also log the mean of everything.
    for k, v in b_metrics.items():
        mean_val = np.mean(v)
        key = "Eval/All/{}".format(k)
        log_dict[key] = mean_val

        mean_val = np.max(v)
        key = "Eval/All/{} Max".format(k)
        log_dict[key] = mean_val

    # Log the metrics.
    log_dict_tb(props.writer, log_dict, global_step=n_collects)


def log_eval_det(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollout_eval_det
    eval_regions_info = props.eval_region_info

    n_xs = len(rollouts_eval)

    # Get info for each run.
    b_metrics = defaultdict(list)

    for ii in range(n_xs):
        rollout = rollouts_eval[ii]
        rew_sum = np.sum(rollout.T_rew)
        traj_len = len(rollout.T_rew)

        metrics = dict(RewSum=rew_sum, TrajLen=traj_len)

        # Append to the batch metrics.
        for k, v in metrics.items():
            b_metrics[k].append(v)

    log_dict = {}

    if len(eval_regions_info) <= 50:
        # Only log per region if there are not too many.
        start_idx = 0
        for region_name, n_eval in eval_regions_info:
            end_idx = start_idx + n_eval

            for k, v in b_metrics.items():
                mean_val = np.mean(v[start_idx:end_idx])
                key = "Eval/{}/{} Det".format(region_name, k)
                log_dict[key] = mean_val

                min_val = np.max(v[start_idx:end_idx])
                key = "Eval/{}/{} Max Det".format(region_name, k)
                log_dict[key] = min_val

                min_val = np.min(v[start_idx:end_idx])
                key = "Eval/{}/{} Min Det".format(region_name, k)
                log_dict[key] = min_val

            start_idx = end_idx

    # Also log the mean of everything.
    for k, v in b_metrics.items():
        mean_val = np.mean(v)
        key = "Eval/All/{} Det".format(k)
        log_dict[key] = mean_val

        mean_val = np.mean(v)
        key = "Eval/All/{} Max Det".format(k)
        log_dict[key] = mean_val

    # Log the metrics.
    log_dict_tb(props.writer, log_dict, global_step=n_collects)


def plot_eval(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollouts_eval
    task: ToyLevelsJax = props.task

    n_seeds = len(rollouts_eval)
    n_xs = len(rollouts_eval[0])

    fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)

    for ii in range(n_xs):
        for jj in range(n_seeds):
            rollout = rollouts_eval[jj][ii]
            T_pos = rollout.Tp1_state.pos
            assert isinstance(T_pos, np.ndarray)
            rew_sum = np.sum(rollout.T_rew)
            is_safe = rew_sum == 0.0

            color = "C1" if is_safe else "C0"
            ax.plot(T_pos[:, 0], T_pos[:, 1], color=color, alpha=0.5)

            if jj == 0:
                # Mark the start.
                ax.plot(
                    T_pos[0, 0],
                    T_pos[0, 1],
                    marker="s",
                    mfc=color,
                    mec="none",
                    markersize=1.8**2,
                )

    plot_dir = run_cfg.paths.eval_plots / "traj"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    task.task_cpu._fig = None


def plot_eval_det(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    rollouts_eval = props.rollout_eval_det
    task: ToyLevelsJax = props.task

    n_xs = len(rollouts_eval)

    fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)

    for ii in range(n_xs):
        rollout = rollouts_eval[ii]
        # import pdb; pdb.set_trace()
        # T_pos = rollout.Tp1_state.pos
        T_pos = rollout.T_state_now[-3]

        rew_sum = np.sum(rollout.T_rew)
        is_safe = rew_sum == 0.0

        color = "C1" if is_safe else "C0"
        ax.plot(T_pos[:, 0], T_pos[:, 1], color=color, alpha=0.5)

        # Mark the start.
        ax.plot(
            T_pos[0, 0],
            T_pos[0, 1],
            marker="s",
            mfc=color,
            mec="none",
            markersize=1.8**2,
        )

    plot_dir = run_cfg.paths.eval_plots / "traj"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"eval_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    task.task_cpu._fig = None
