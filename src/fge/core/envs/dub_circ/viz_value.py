from typing import Self

import einops as ei
import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from loguru import logger
from matplotlib.colors import BoundaryNorm, CenteredNorm, ListedColormap
from matplotlib.patches import Patch
from og.jax_utils import jax2np, jax_vmap
from og.register_sns_cmaps import register_cmaps

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.dub_circ.analyze_ppo import (
    _overlay_mask,
    _plot_eval_results_small,
    _setup_eval_dist_plot,
)
from fge.core.envs.dub_circ.dub_circ_jax import DubinsJax
from fge.core.envs.toylevels.viz_value import compute_rollout_metrics
from fge.core.utils.jax_util import myjit


class DubinsJaxVizVDSValue2D(struct.PyTreeNode):
    """Visualize the ensemble from VDS."""

    task: DubinsJax = struct.field(pytree_node=False)

    @myjit
    def get_vds(self, ppo: IntrinsicSumPPO, vds: VDSGAE):
        """
        Get the VDS values for the evaluation contour.
        Args:
            ppo (IntrinsicSumPPO): The PPO model.
            vds (VDSGAE): The VDS model.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The evaluation contour points, the VDS values, and the ensemble values.
        """
        if isinstance(ppo, SumPPO):
            get_Vl = ppo.Vl.apply
        elif isinstance(ppo, IntrinsicSumPPO):
            get_Vl = ppo.Vl_ext.apply
        else:
            raise ValueError("")

        b_x, b_obs, _ = self.task.get_eval_contour()

        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        be_V = jax_vmap(vds.get_e_V)(b_obs)

        return b_x, b_Vl, be_V

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()

        vds: VDSGAE = props.extra["vds"]
        task: DubinsJax = props.task
        trajsaver = props.trajsaver

        b_x, b_Vl, be_vds = jax2np(self.get_vds(props.ppo, vds))

        b_std = np.std(be_vds, axis=1)
        b_V_max = np.max(be_vds, axis=1)
        b_V_min = np.min(be_vds, axis=1)

        b_x_rollouteval: np.ndarray = task.eval_xs()
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        b_ic_train = np.array(
            [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s]
        )

        nrow = 2
        ncol = 3
        figsize = np.array([6.0 * ncol, 4.0 * nrow])
        fig, axes = plt.subplots(
            nrow, ncol, sharex=True, figsize=figsize, layout="constrained"
        )
        # [task.task_cpu.label_ic(ax) for ax in axes]
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

        # ----- 2. Ensemble Std -----
        ax = axes[1]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Ensemble Std")
        b_std_grid = task.get_contour_grid(b_std)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_std_grid, ax, run_cfg, cmap, label="Ensemble Std")
        # Plot all reset train x0's
        train_x0s = np.array(
            [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s]
        )
        markers = ["D" if traj.T_trunc[-1] else "x" for traj in trajsaver.trajs]
        for marker, alpha, z_order in zip(["D", "x"], [0.8, 0.2], [2, 1]):
            idxs = np.where(np.array(markers) == marker)[0]
            ax.scatter(
                train_x0s[idxs, 0],
                train_x0s[idxs, 1],
                marker=marker,
                color="black",
                alpha=alpha,
                s=20,
                zorder=z_order,
                label="Success" if marker == "D" else "Collide",
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----- 3. Ensemble Max -----
        ax = axes[2]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("V Max")
        b_V_max_grid = task.get_contour_grid(b_V_max)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_V_max_grid, ax, run_cfg, cmap, label="Ensemble Max")

        # ----- 4. Ensemble Min -----
        ax = axes[3]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("V Min")
        b_V_min_grid = task.get_contour_grid(b_V_min)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_V_min_grid, ax, run_cfg, cmap, label="Ensemble Min")

        # # ----- 5. Show base distribution -----
        # ax = axes[4]
        # ax.set_title("Base Reset Distribution")
        # _setup_eval_dist_plot(ax, run_cfg, task, props)
        #
        # # Plot density of pvs over the grid
        # lb, ub = run_cfg.task_cfg.other_min_ang_vel, run_cfg.task_cfg.other_max_ang_vel
        # x = np.linspace(lb, ub, 100)
        # y = np.linspace(lb, ub, 100)
        # X, Y = np.meshgrid(x, y)
        # Z = task.task_cpu.other_vel_dist.pdf(np.column_stack([X.ravel(), Y.ravel()])).prod(-1).reshape(X.shape)
        # ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        #
        # # Plot all reset train x0's
        # markers = ['D' if traj.T_trunc[-1] else 'x' for traj in trajsaver.trajs]
        # for marker, alpha, z_order in zip(['D', 'x'], [0.7, 0.7], [2, 1]):
        #     idxs = np.where(np.array(markers) == marker)[0]
        #     color = "black" if marker == "D" else "red"
        #     facecolor = "none" if marker == "D" else color
        #     edgecolor = color if marker == "D" else None
        #     ax.scatter(b_ic_train[idxs, 0], b_ic_train[idxs, 1], marker=marker, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, s=4 ** 2,
        #                zorder=z_order, label="Success" if marker == 'D' else "Collide")
        # ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False)

        plot_dir = run_cfg.paths.eval_plots / "vds"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"vds_{n_collects:05d}.jpg"
        fig.set_size_inches(15, 10)
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)


@struct.dataclass(frozen=False)
class DubinsJaxVizCustom:
    task: DubinsJax = struct.field(pytree_node=False)

    x0_ci_idx_last: int = struct.field(pytree_node=False)

    buf_ci_id_last: int = struct.field(pytree_node=False)
    buf_explore_id_last: int = struct.field(pytree_node=False)

    set_explore_zero: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: DubinsJax, set_explore_zero: bool = False) -> Self:
        return DubinsJaxVizCustom(task, 0, 0, 0, set_explore_zero)

    @myjit
    def eval_nsf_ci(self, nsf_ci: NSF):
        b_x, b_obs, _ = self.task.get_eval_contour()
        b_ci_logprob = jax.vmap(nsf_ci.log_prob)(b_obs)
        return b_ci_logprob, b_obs

    @myjit
    def fuck(self, nsf_ci: NSF, obs):
        return nsf_ci.log_prob(obs)

    @myjit
    def eval_nn(
        self,
        ppo: IntrinsicSumPPO,
        nsf: NSF,
        vds: VDSGAE,
        rnd: RND,
        ci_classify: CIClassifier,
        pol_classify: CIClassifier,
    ):
        get_Vl = ppo.Vl_ext.apply

        b_x, b_obs, _ = self.task.get_eval_contour()

        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        b_logprob = jax.vmap(nsf.log_prob)(b_obs)
        be_V = jax_vmap(vds.get_e_V)(b_obs)
        b_rnd = jax.vmap(rnd.get_bonus)(b_obs)
        b_prob = jax.vmap(ci_classify.get_probs)(b_obs)
        b_prob_pol = jax.vmap(pol_classify.get_probs)(b_obs)

        return b_x, b_Vl, b_logprob, be_V, b_rnd, b_prob, b_prob_pol

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        if "buf" not in props.extra:
            logger.error("No buf in props.extra")
            return
        if "ci_classify" not in props.extra:
            logger.error("No ci_classify in props.extra")
            return

        nsf_ci = props.extra.get("nsf_ci", None)

        buf: BufCustom = props.extra["buf"]
        ppo = props.ppo
        nsf: NSF = props.extra["nsf"]
        vds: VDSGAE = props.extra["vds"]
        rnd: RND = props.extra["rnd"]
        ci_classify: CIClassifier = props.extra["ci_classify"]
        pol_classify: CIClassifier = props.extra["pol_classify"]
        task = props.task
        info = props.extra["custom"]
        trajsaver = props.trajsaver

        # b_x.shape=(25, 2)
        # b_Vl.shape=(25,)
        # b_logprob.shape=(25,)
        # be_V.shape=(25, 3)
        # b_prob.shape=(25,)
        # b_prob_polsafe.shape=(25,)
        b_x, b_Vl, b_logprob, be_V, b_rnd, b_prob_ci, b_prob_polsafe = jax2np(
            self.eval_nn(ppo, nsf, vds, rnd, ci_classify, pol_classify)
        )
        b_inci_nsf = None
        if nsf_ci is not None:
            b_ci_logprob_nsf, b_obs = jax2np(self.eval_nsf_ci(nsf_ci))
            b_inci_nsf = b_ci_logprob_nsf >= buf.logprob_thresh_unbiased

            idx_middle = len(b_obs) // 2
            obs_dbg = b_obs[idx_middle]

            # # Find the closest obs to obs_dbg in the CI buffer.
            # if buf.n_ci_buf > 0:
            #     b_obs_ci = buf.get_b_obs_ci()
            #     assert obs_dbg.shape == b_obs_ci.shape[1:]
            #     ii_argmin = np.argmin(np.linalg.norm(b_obs_ci - obs_dbg[None, :], axis=-1))
            #     obs_ci_closest = b_obs_ci[ii_argmin]
            #
            #     # Evaluate NSF CI on obs_ci_closest.
            #     logprob_wtf = np.array(self.fuck(nsf_ci, obs_dbg))
            #     logprob_ci_closest = np.array(self.fuck(nsf_ci, obs_ci_closest))

        # ---------------------------------------------------
        p_explore = float(buf.buffer.p_explore)
        if self.set_explore_zero and (n_collects >= 1000) and (p_explore > 0):
            # If during eval the entire region from -0.5 to 0.5 is predicted in CI, then set p_explore to 0.
            b_valid_region = (-0.5 <= b_x) & (b_x <= 0.5)
            b_inci = b_prob_ci >= 0.5
            inci_region_all = np.all(b_inci[b_valid_region])

            if inci_region_all:
                logger.critical("The entire region is in CI! Setting p_explore to 0.")
                buf.buffer.set_probs_inplace(
                    0.0, buf.buffer.p_base, buf.buffer.p_predci
                )
        # ---------------------------------------------------
        # Common variables.
        b_ic_train = np.array(
            [DubinsJax.State(*_x0).o_states[:, -1] for _x0 in trajsaver.x0s]
        )
        b_ic_source = np.array(
            [DubinsJax.State(*_x0).source for _x0 in trajsaver.x0s], dtype=np.int32
        )

        # ---------------------------------------------------

        # Set up plot
        rng = np.random.default_rng(seed=12345)
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

        # ----- 2. Recent reset states -----
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

        # ----- 3. Explore Buffer -----
        ax = axes[2]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Explore Buffer States")
        if buf.buffer.buf_explore.size > 0:
            buf_explore = buf.buffer.buf_explore
            buf_expl_pts: StateResetId = buf_explore.all_valid()
            b_buf_expl_ic = np.array(self.task.to_icval(buf_expl_pts.state))

            id_now = int(buf_explore.id)
            ids_new = np.arange(self.buf_explore_id_last, id_now)
            idxs_new = ids_new % buf_explore.capacity
            # Remove duplicate points in idxs_new.
            idxs_new = np.unique(idxs_new)

            b_expl_ic_new = b_buf_expl_ic[idxs_new]

            ax.scatter(
                b_expl_ic_new[:, 0],
                b_expl_ic_new[:, 1],
                marker="o",
                color="C0",
                alpha=0.4,
                s=20,
                zorder=1,
                label="Explore (New)",
            )
            ax.scatter(
                b_buf_expl_ic[:, 0],
                b_buf_expl_ic[:, 1],
                marker="o",
                color="C1",
                alpha=0.4,
                s=20,
                zorder=1,
                label="Explore",
            )

            ax.legend(
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )

            self.buf_explore_id_last = id_now

        # Plot NSF mask
        logprob_grid = task.get_contour_grid(b_logprob)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(logprob_grid, ax, run_cfg, cmap, label="NSF Log Prob")

        # ----- 4. CI Buffer -----
        ax = axes[3]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("CI Buffer States")
        if buf.n_ci_buf > 0:
            # Scatter CI buffer points
            b_x0_ci = buf.get_b_x0_ci()
            b_ic = task.to_icval(task.leaf_to_minstate(b_x0_ci))
            n_pts = len(b_ic)

            ##
            b_ic_ci_sampled = info["CI/ic"]
            b_Vl_sampled = info["CI/Vl"]

            ##
            buf_ci = buf.buffer.buf_ci
            buf_ci_pts: StateResetId = buf_ci.all_valid()
            b_buf_ci_ic = np.array(self.task.to_icval(buf_ci_pts.state))

            id_now = int(buf_ci.id)
            ids_new = np.arange(self.buf_ci_id_last, id_now)
            idxs_new = ids_new % buf_ci.capacity
            # Remove duplicate points in idxs_new.
            idxs_new = np.unique(idxs_new)
            b_ci_ic_new = b_buf_ci_ic[idxs_new]
            b_ic_new = b_ic[self.x0_ci_idx_last :]

            ax.scatter(
                b_ic_ci_sampled[:, 0],
                b_ic_ci_sampled[:, 1],
                marker="$A$",
                color="magenta",
                alpha=0.4,
                s=20,
                zorder=1,
                label="CI Argmax Cand",
            )
            ax.scatter(
                b_ic_new[:, 0],
                b_ic_new[:, 1],
                marker="$N$",
                color="C0",
                alpha=0.4,
                s=20,
                zorder=1,
                label="CI (New)",
            )
            ax.scatter(
                b_ic[:, 0],
                b_ic[:, 1],
                marker="o",
                edgecolor="C2",
                facecolor="none",
                alpha=0.4,
                s=20,
                label="CI",
            )

            ax.legend(
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )

        # Plot CI Probs mask
        b_probs = task.get_contour_grid(b_prob_ci)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_probs, ax, run_cfg, cmap, label="CI Classify Prob")

        # ----- 5. CI Dist -----
        # Replotting CI dist again since the CI blocks it all out
        ax = axes[4]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        ax.set_title("CI Probs")
        # Plot CI Probs mask
        b_probs = task.get_contour_grid(b_prob_ci)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_probs, ax, run_cfg, cmap, label="CI Classify Prob")

        # ----- 6. Rehearsal Buffer -----
        ax = axes[5]
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        ax.set_title("Rehearsal Buffer States")

        if buf.n_ci_buf > 0:
            ax.scatter(
                b_buf_ci_ic[:, 0],
                b_buf_ci_ic[:, 1],
                marker="d",
                color="C3",
                alpha=0.4,
                s=20,
                zorder=1,
                label="CI Rehearsal",
            )
            ax.scatter(
                b_ci_ic_new[:, 0],
                b_ci_ic_new[:, 1],
                marker="$N$",
                color="C1",
                alpha=0.4,
                s=20,
                zorder=1,
                label="CI Rehearsal (New)",
            )

            self.buf_ci_id_last = id_now
            self.x0_ci_idx_last = len(b_ic) - 1

            ax.legend(
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )

        # ----- 7. VDS -----
        ax = axes[6]
        ax.set_title("VDS Ensemble Std")
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        b_std = np.std(be_V, axis=1)
        b_V_max = np.max(be_V, axis=1)
        b_V_min = np.min(be_V, axis=1)

        b_std_grid = task.get_contour_grid(b_std)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_std_grid, ax, run_cfg, cmap, label="Ensemble Std")

        ax = axes[7]
        if nsf_ci is None:
            # ----- 8. RND -----
            ax.set_title("RND")
            _setup_eval_dist_plot(ax, run_cfg, task, props)
            _plot_eval_results_small(ax, run_cfg, task, props)
            b_rnd_grid = task.get_contour_grid(b_rnd)
            cmap = plt.get_cmap("viridis")
            _overlay_mask(b_rnd_grid, ax, run_cfg, cmap, label="RND")
        else:
            # ----- 8. NSF CI -----
            ax.set_title("NSF CI (thresh={:.1e})".format(buf.logprob_thresh_unbiased))
            _setup_eval_dist_plot(ax, run_cfg, task, props)
            _plot_eval_results_small(ax, run_cfg, task, props)
            bb_ci_logprob_nsf = task.get_contour_grid(b_ci_logprob_nsf)
            cmap = plt.get_cmap("RdBu_r")
            mid = buf.logprob_thresh_unbiased
            halfrange = np.abs(bb_ci_logprob_nsf - mid).max()
            norm = CenteredNorm(vcenter=mid, halfrange=halfrange)
            _overlay_mask(
                bb_ci_logprob_nsf,
                ax,
                run_cfg,
                cmap,
                norm=norm,
                label="Logprob",
                levels=[buf.logprob_thresh_unbiased],
            )

        # ----- 9. Value -----
        ax = axes[8]
        ax.set_title("V Value")
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        b_Vl_grid = task.get_contour_grid(b_Vl)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_Vl_grid, ax, run_cfg, cmap, label="V Value")

        # ----- 10. Pol Classify -----
        ax = axes[9]
        ax.set_title("Pol Classify")
        _setup_eval_dist_plot(ax, run_cfg, task, props)
        _plot_eval_results_small(ax, run_cfg, task, props)
        b_probs = task.get_contour_grid(b_prob_polsafe)
        cmap = plt.get_cmap("viridis")
        _overlay_mask(b_probs, ax, run_cfg, cmap, label="Pol Classify Prob")

        # ----- 11. Recent Reset States, but by source -----
        ax = axes[10]
        ax.set_title("Recent Reset States (Source)")
        _setup_eval_dist_plot(ax, run_cfg, task, props)

        source_labels = Source.labels_dict()
        colors_dict = Source.colors_dict()
        zorder_dict = Source.zorder_dict()

        for ii, source in enumerate(Source):
            color = colors_dict[source]
            zorder = zorder_dict[source]

            b_issource = b_ic_source == source

            b_ic_source_ = b_ic_train[b_issource]

            frac_source = np.mean(b_issource)

            ax.scatter(
                b_ic_source_[:, 0],
                b_ic_source_[:, 1],
                marker="o",
                edgecolor=color,
                facecolor="none",
                alpha=0.95,
                s=4**2,
                label="{} {:.0%}".format(source_labels[source], frac_source),
                zorder=zorder,
            )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----- 12. Show Classifier vs NSF -----
        ax = axes[11]
        if nsf_ci is not None:
            ax.set_title("Classifier vs NSF")

            # If nsf_ci is not None, then compare nsf_ci with ci classifier.
            b_inci = b_prob_ci >= 0.5

            b_in_neither = (~b_inci) & (~b_inci_nsf)
            b_in_clsfy_only = b_inci & (~b_inci_nsf)
            b_in_nsf_only = (~b_inci) & b_inci_nsf
            b_in_both = b_inci & b_inci_nsf

            b_values = np.where(
                b_in_neither,
                0,
                np.where(
                    b_in_clsfy_only,
                    1,
                    np.where(b_in_nsf_only, 2, np.where(b_in_both, 3, 4)),
                ),
            )
            # There should be no values greater than 3.
            assert np.all(b_values <= 3), f"b_values max: {b_values.max()}"

            bb_values = task.get_contour_grid(b_values)

            colors = ["C0", "C1", "C4", "C5"]
            labels = ["None", "Clsfy", "NSF", "Both"]
            cmap = ListedColormap(colors)
            bounds = np.arange(-0.5, 4, 1)  # boundaries between values
            norm = BoundaryNorm(bounds, cmap.N)

            im = ax.imshow(
                ei.rearrange(bb_values, "y x -> x y"),
                extent=[
                    run_cfg.task_cfg.other_min_ang_vel,
                    run_cfg.task_cfg.other_max_ang_vel,
                    run_cfg.task_cfg.other_min_ang_vel,
                    run_cfg.task_cfg.other_max_ang_vel,
                ],
                origin="lower",
                cmap=cmap,
                norm=norm,
                alpha=0.95,
            )
            # create a legend with proxy artists
            patches = [
                Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))
            ]
            ax.legend(
                handles=patches,
                title="Category",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

        # -----------------------------------------------------
        name = "custom"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.set_size_inches(20, 10)
        fig.savefig(fig_path, dpi=400)
        plt.close(fig)
