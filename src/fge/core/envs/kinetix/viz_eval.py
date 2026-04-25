import numpy as np

from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.envs.kinetix.lander import render_trajs_sidebyside


def save_traj_video(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    # rollouts_eval = props.rollouts_eval[0]
    rollouts_eval = props.rollout_eval_det

    n_rollouts = len(rollouts_eval)

    n_to_render = 9

    # Evenly spaced
    idxs_to_render = np.round(np.linspace(0, n_rollouts - 1, n_to_render)).astype(int)
    rollouts_to_render = [rollouts_eval[i] for i in idxs_to_render]

    plot_dir = run_cfg.paths.eval_plots / "traj_vids"
    plot_dir.mkdir(parents=True, exist_ok=True)
    vid_path = plot_dir / f"traj_{n_collects:05d}.mp4"
    render_trajs_sidebyside(rollouts_to_render, vid_path)

    # # Convert to flat index
    # # Original size: (n_goal_settings, n_px, n_py, n_theta) = (5, 5, 5, 3)
    # idx_to_render = (0, 2, 2, 1)
    # flat_idx = np.ravel_multi_index(idx_to_render, (5, 5, 5, 3))
    # assert flat_idx < len(rollouts_eval)
    #
    # rollout = rollouts_eval[flat_idx]
    #
    # plot_dir = run_cfg.paths.eval_plots / "traj_vids"
    # plot_dir.mkdir(parents=True, exist_ok=True)
    # vid_path = plot_dir / f"traj_{n_collects:05d}.mp4"
    # render_traj(rollout, vid_path)
