import functools as ft
from typing import Self, Union

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from loguru import logger
from matplotlib.colors import Normalize
from og.dyn_types import BObs
from og.jax_utils import jax2np, jax_vmap
from og.register_sns_cmaps import register_cmaps
from og.rng import PRNGKey
from og.tree_utils import tree_stack

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.collector import RolloutOutput
from fge.core.bits.intrinsic_ppo import IntrinsicSumPPO
from fge.core.bits.nsf import NSF
from fge.core.bits.ppo_core import SumPPO
from fge.core.bits.rnd import RND
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.bits.sumppo_lag import SumPPOLag
from fge.core.bits.sumppo_saute import SumPPOSaute
from fge.core.bits.vds_gae import VDSGAE
from fge.core.envs.mujoco.hopper.hopper_jax import HopperJax
from fge.core.envs.toylevels.toylevels_jax import ToyLevelsJax
from fge.core.utils.jax_util import myjit


class VizValue(struct.PyTreeNode):
    """Visualize the value function."""

    task: ToyLevelsJax = struct.field(pytree_node=False)

    @myjit
    def get_value(self, ppo: SumPPO):
        bb_X, bb_Y, bb_obs, _ = self.task.get_contour_grid()
        if isinstance(ppo, Union[SumPPO, SumPPOSaute, SumPPOLag]):
            get_Vl = ppo.Vl.apply
        elif isinstance(ppo, IntrinsicSumPPO):
            get_Vl = ppo.Vl_ext.apply
        else:
            raise ValueError("")

        bb_V = jax_vmap(get_Vl, rep=2)(bb_obs).squeeze(-1)
        return bb_X, bb_Y, bb_V

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()

        bb_X, bb_Y, bb_V = jax2np(self.get_value(props.ppo))

        ppo = props.ppo
        task: ToyLevelsJax = ppo.task

        cmap = "vlag"

        fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)

        cm = ax.contourf(
            bb_X, bb_Y, bb_V, cmap=cmap, alpha=0.9, vmin=0.0, vmax=1.0, zorder=100
        )
        cbar = fig.colorbar(cm, ax=ax)

        plot_dir = run_cfg.paths.eval_plots / "V"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"V_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
        task.task_cpu._fig = None


class VizRNDBonus(struct.PyTreeNode):
    """Visualize the RND bonus."""

    task: ToyLevelsJax = struct.field(pytree_node=False)

    @myjit
    def get_rnd_bonus(self, rnd: RND):
        bb_X, bb_Y, bb_obs, _ = self.task.get_contour_grid()
        bb_rnd_bonus = jax_vmap(rnd.get_bonus, rep=2)(bb_obs)
        return bb_X, bb_Y, bb_rnd_bonus

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()

        rnd: RND = props.extra["rnd"]

        bb_X, bb_Y, bb_rnd_bonus = jax2np(self.get_rnd_bonus(rnd))

        ppo = props.ppo
        task: ToyLevelsJax = ppo.task

        norm = Normalize(vmin=0.0, vmax=bb_rnd_bonus.max())

        cmap = "rocket"
        fig, ax = task.task_cpu._get_fig_and_ax(draw_agent=False)

        cm = ax.contourf(
            bb_X, bb_Y, bb_rnd_bonus, cmap=cmap, alpha=0.95, norm=norm, zorder=100
        )
        cbar = fig.colorbar(cm, ax=ax)
        ax.set_title("RND Bonus")

        plot_dir = run_cfg.paths.eval_plots / "rnd"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"rnd_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
        task.task_cpu._fig = None


def compute_rollout_metrics_rng(rollouts_eval: list[list[RolloutOutput]], gamma: float):
    n_seeds = len(rollouts_eval)
    n_xs = len(rollouts_eval[0])

    b_Vl_disc = []
    b_T = []
    for ii in range(n_xs):
        values = []
        Ts = []
        for jj in range(n_seeds):
            rollout = rollouts_eval[jj][ii]
            T = len(rollout.T_rew)
            rew_sum = np.sum(rollout.T_rew)
            is_safe = rew_sum == 0.0
            value = 0.0 if is_safe else gamma**T
            values.append(value)
            Ts.append(T)
        value_mean = np.mean(values)
        T_mean = np.mean(Ts)

        b_Vl_disc.append(value_mean)
        b_T.append(T_mean)

    b_Vl_disc = np.array(b_Vl_disc)
    b_T = np.array(b_T)

    return b_Vl_disc, b_T


def compute_rollout_metrics_det(rollout_eval: list[RolloutOutput], gamma: float):
    n_xs = len(rollout_eval)

    b_Vl_disc = []
    b_T = []
    for ii in range(n_xs):
        values = []
        Ts = []

        rollout = rollout_eval[ii]
        T = len(rollout.T_rew)
        rew_sum = np.sum(rollout.T_rew)
        is_safe = rew_sum == 0.0
        value = 0.0 if is_safe else gamma**T

        b_Vl_disc.append(value)
        b_T.append(T)

    b_Vl_disc = np.array(b_Vl_disc)
    b_T = np.array(b_T)

    return b_Vl_disc, b_T


def compute_rollout_metrics(props: EvalProps):
    gamma = props.ppo.disc_gamma
    b_Vl_disc_rng, b_T_rng = compute_rollout_metrics_rng(props.rollouts_eval, gamma)
    b_Vl_disc_det, b_T_det = compute_rollout_metrics_det(props.rollout_eval_det, gamma)
    return b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det


class VizValue1D(struct.PyTreeNode):
    """Visualize cthe value function over the initial conditions only."""

    task: ToyLevelsJax = struct.field(pytree_node=False)

    @myjit
    def get_value(self, ppo: SumPPO | IntrinsicSumPPO):
        if isinstance(ppo, Union[SumPPO, SumPPOLag, SumPPOSaute]):
            get_Vl = ppo.Vl.apply
        elif isinstance(ppo, IntrinsicSumPPO):
            get_Vl = ppo.Vl_ext.apply
        else:
            raise ValueError("")
        b_x, b_obs, _ = self.task.get_eval_contour()

        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        return b_x, b_Vl

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()
        task: HopperJax = props.task

        b_x_rollouteval: np.ndarray = task.eval_ics()

        # Compute the "true" value function from both rng and det.
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        b_x, b_Vl = jax2np(self.get_value(props.ppo))

        # ----------------------------
        fig, axes = plt.subplots(2, sharex=True, layout="constrained")
        [task.task_cpu.label_ic(ax) for ax in axes]
        ax = axes[0]
        ax.set_title("Vl")

        ax.plot(b_x_rollouteval, b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, b_Vl_disc_det, color="C5", label="Vl MC (det)")
        ax.plot(b_x, b_Vl, color="C1", label="Vl")
        fig.legend(loc="outside right upper")

        # ----------------------------
        ax = axes[1]
        ax.set_title("Steps Alive")
        ax.plot(b_x_rollouteval, b_T_rng, marker="o", ms=3, lw=0.9, color="C4")
        ax.plot(b_x_rollouteval, b_T_det, marker="o", ms=3, lw=0.9, color="C5")

        # ----------------------------
        plot_dir = run_cfg.paths.eval_plots / "value_gt"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"value_gt_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)


class VizRNDValue1D(struct.PyTreeNode):
    """Visualize both the value function and RND bonus over the initial conditions only."""

    task: ToyLevelsJax = struct.field(pytree_node=False)

    @myjit
    def get_rnd_value(self, ppo: IntrinsicSumPPO, rnd: RND, nsf: NSF):
        get_Vl = ppo.Vl_ext.apply

        b_x, b_obs, _ = self.task.get_eval_contour()

        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        b_rnd_bonus = jax_vmap(rnd.get_bonus)(b_obs)

        b_logprob = jax.vmap(nsf.log_prob)(b_obs)
        b_nsf_bonus = -b_logprob

        return b_x, b_rnd_bonus, b_nsf_bonus, b_Vl

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        register_cmaps()

        if "rnd" not in props.extra:
            return

        rnd: RND = props.extra["rnd"]
        nsf: NSF = props.extra["nsf"]
        # buf: BufCustom = props.extra["buf"]
        task: ToyLevelsJax = props.task

        info: dict = props.extra["custom"]
        trajsaver = props.trajsaver

        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)
        b_x, b_rnd_bonus, b_nsf_bonus, b_Vl = jax2np(
            self.get_rnd_value(props.ppo, rnd, nsf)
        )

        if not np.isfinite(b_nsf_bonus).all():
            logger.error("nsf not finite")
            ipdb.set_trace()

        # Visualize the argmax of RND.
        idx_argmax_rnd = np.argmax(b_rnd_bonus)
        x_argmax_rnd = b_x[idx_argmax_rnd]

        idx_argmax_nsf = np.argmax(b_nsf_bonus)
        x_argmax_nsf = b_x[idx_argmax_nsf]

        nrow = 5
        figsize = np.array([8.0, 1.5 * nrow])
        fig, axes = plt.subplots(
            nrow, sharex=True, figsize=figsize, layout="constrained"
        )
        [task.task_cpu.label_ic(ax) for ax in axes]

        if "x" in info:
            # Overlay the sampled points in the custom buffer, if we had a new argmax.
            b_ic_sampled = task.to_icval(info["x"])
            b_rnd_sampled = info["rnd"]
            b_nsf_sampled = info["nsf"]

        # ----------------------------
        ax = axes[0]
        ax.set_title("RND Bonus [Custom Sampled Pts]")

        ax.plot(b_x, b_rnd_bonus, color="C1")
        ax.plot(x_argmax_rnd, b_rnd_bonus[idx_argmax_rnd], "*", color="C0", zorder=3)

        if "x" in info:
            # Overlay the sampled points in the custom buffer, if we had a new argmax.
            argmax_sampled = np.argmax(b_rnd_sampled)

            ax.scatter(b_ic_sampled, b_rnd_sampled, color="C2", zorder=4)
            ax.plot(
                b_ic_sampled[argmax_sampled],
                b_rnd_sampled[argmax_sampled],
                "*",
                color="C5",
                zorder=5,
            )

        # Overlay the initial conditions of the most recently finished trajs.
        if len(trajsaver.trajs) > 0:
            b_x0_trajsaver = tree_stack(
                [t.x0 for t in trajsaver.trajs], axis=0, which=np
            )
            b_ic_trajsaver = task.to_icval(task.leaf_to_minstate(b_x0_trajsaver))

            assert b_ic_trajsaver.ndim == 1
            for ic in b_ic_trajsaver:
                ax.axvline(ic, color="C3", lw=0.6)

        # ----------------------------
        ax = axes[1]
        ax.set_title("NSF Bonus [Custom Sampled Pts]")
        ax.plot(b_x, b_nsf_bonus, color="C1")
        ax.plot(x_argmax_nsf, b_nsf_bonus[idx_argmax_nsf], "*", color="C0", zorder=3)

        if "x" in info:
            # Overlay the sampled points in the custom buffer, if we had a new argmax.
            argmax_sampled = np.argmax(b_nsf_sampled)

            ax.scatter(b_ic_sampled, b_nsf_sampled, color="C2", zorder=4)
            ax.plot(
                b_ic_sampled[argmax_sampled],
                b_nsf_sampled[argmax_sampled],
                "*",
                color="C5",
                zorder=5,
            )

        # ----------------------------
        ax = axes[2]

        ax.set_title("Vl [recent rehearsal]".format())

        # # Visualize the most recently added points to the rehearsal buffer.
        # if "buf" in props.extra:
        #     buf: BufCustom = props.extra["buf"]
        #     b_x0_recent: ToyLevelsJax.State = jax2np(buf.buffer.most_recent(6)).state
        #
        #     b_ic = task.to_icval(b_x0_recent)
        #     for ii, ic in enumerate(b_ic):
        #         axes[1].axvline(ic, color=f"C{ii}", lw=0.8)

        b_x_rollouteval: np.ndarray = task.eval_ics()
        ax.plot(b_x_rollouteval, b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, b_Vl_disc_det, color="C5", label="Vl MC (det)")
        ax.plot(b_x, b_Vl, color="C1")
        fig.legend(loc="outside right upper")

        # ----------------------------
        ax = axes[3]
        ax.set_title("Steps Alive")
        ax.plot(b_x_rollouteval, b_T_rng, marker="o", ms=3, lw=0.9, color="C4")
        ax.plot(b_x_rollouteval, b_T_det, marker="o", ms=3, lw=0.9, color="C5")

        # ----------------------------
        plot_dir = run_cfg.paths.eval_plots / "rndvalue_ic"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"rndvalue_ic_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)


class VizVDSValue1D(struct.PyTreeNode):
    """Visualize the ensemble from VDS."""

    task: ToyLevelsJax = struct.field(pytree_node=False)

    @myjit
    def get_vds(self, ppo: IntrinsicSumPPO, vds: VDSGAE):
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
        task: ToyLevelsJax = props.task

        b_x, b_Vl, be_vds = jax2np(self.get_vds(props.ppo, vds))

        b_std = np.std(be_vds, axis=1)
        b_V_max = np.max(be_vds, axis=1)
        b_V_min = np.min(be_vds, axis=1)

        b_x_rollouteval: np.ndarray = task.eval_ics()
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        nrow = 3
        figsize = np.array([8.0, 2.5 * nrow])
        fig, axes = plt.subplots(
            nrow, sharex=True, figsize=figsize, layout="constrained"
        )
        [task.task_cpu.label_ic(ax) for ax in axes]

        # ----------------------------
        ax = axes[0]
        ax.set_title("Ensemble Std")
        ax.plot(b_x, b_std, color="C1")

        # ----------------------------
        ax = axes[1]
        ax.set_title("Ensemble")
        ax.fill_between(b_x, b_V_max, b_V_min, color="C2", alpha=0.2)
        ax.plot(b_x, b_V_max, color="C2", alpha=0.5)
        ax.plot(b_x, b_V_min, color="C2", alpha=0.5)

        ax.plot(b_x, b_Vl, color="C1", alpha=0.9)

        ax.plot(b_x_rollouteval, b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, b_Vl_disc_det, color="C5", label="Vl MC (det)")
        fig.legend(loc="outside right upper")

        # ----------------------------
        ax = axes[2]
        ax.set_title("Steps Alive")
        ax.plot(b_x_rollouteval, b_T_rng, marker="o", ms=3, lw=0.9, color="C4")
        ax.plot(b_x_rollouteval, b_T_det, marker="o", ms=3, lw=0.9, color="C5")

        plot_dir = run_cfg.paths.eval_plots / "vds"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"vds_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)


# def plot_rehearsal_buf(n_collects: int, run_cfg: RunCfg, props: EvalProps):
#     """Visualize the most recently added points to the rehearsal buffer."""
#     if "buf" not in props.extra:
#         return
#
#     buf: BufCustom = props.extra["buf"]
#     task = props.task
#     b_x0 = jax2np(buf.buffer.all_valid())
#     b_x0_ic = task.to_icval(b_x0.state)
#
#     bins = task.icval_bins
#     fig, ax = plt.subplots(layout="constrained")
#     task.task_cpu.label_ic(ax)
#     ax.hist(b_x0_ic, color="C1", bins=bins)
#
#     plot_dir = run_cfg.paths.train_plots / "rehearsal_hist"
#     plot_dir.mkdir(parents=True, exist_ok=True)
#     fig_path = plot_dir / f"rehearsal_hist_{n_collects:05d}.jpg"
#     fig.savefig(fig_path, bbox_inches="tight", dpi=250)
#     plt.close(fig)


class VizCIDensity(struct.PyTreeNode):
    task: ToyLevelsJax = struct.field(pytree_node=False)

    # @myjit
    # def get_logprob_ci(self, nsf_ci: NSF):
    #     b_x, b_obs, _ = self.task.get_eval_contour()
    #     b_logprob = jax.vmap(nsf_ci.log_prob)(b_obs)
    #
    #     return b_x, b_logprob

    @myjit
    def get_logprob_ci(self, ci_classify: CIClassifier):
        b_x, b_obs, _ = self.task.get_eval_contour()
        b_prob = jax.vmap(ci_classify.get_probs)(b_obs)

        return b_x, b_prob

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        """Visualize the most recently added points to the rehearsal buffer."""
        if "buf" not in props.extra:
            return
        # if "nsf_ci" not in props.extra:
        #     return
        if "ci_classify" not in props.extra:
            return

        buf: BufCustom = props.extra["buf"]
        # nsf_ci: NSF = props.extra["nsf_ci"]
        ci_classify: CIClassifier = props.extra["ci_classify"]
        task = props.task

        # b_x, b_logprob_ci = jax2np(self.get_logprob_ci(nsf_ci))
        # b_prob_ci = np.exp(b_logprob_ci)

        b_x, b_prob_ci = jax2np(self.get_logprob_ci(ci_classify))

        # ------------------------------------------------------
        fig, ax = plt.subplots(layout="constrained")
        task.task_cpu.label_ic(ax)
        ax.plot(b_x, b_prob_ci, color="C1")

        # Highlight regions where b_prob_ci >= 0.5 using axvspan.
        ax.fill_between(
            b_x,
            0,
            1,
            where=b_prob_ci >= 0.5,
            color="C5",
            alpha=0.3,
            transform=ax.get_xaxis_transform(),
        )

        if buf.n_ci_buf > 0:
            b_x0_ci = buf.get_b_x0_ci()
            b_ic = task.to_icval(task.leaf_to_minstate(b_x0_ci))
            n_pts = len(b_ic)

            # Jitter around the bottom of the plot.
            jitter_scale = 0.1

            jitter_max = 0.0
            jitter_min = -jitter_scale
            rng = np.random.default_rng(seed=12345)
            b_jitter = rng.uniform(low=jitter_min, high=jitter_max, size=n_pts)
            ax.scatter(b_ic, b_jitter, color="C0", s=3)
        # ------------------------------------------------------

        name = "ci_prob"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)


@struct.dataclass(frozen=False)
class VizCustom:
    task: HopperJax = struct.field(pytree_node=False)

    x0_ci_idx_last: int = struct.field(pytree_node=False)

    buf_ci_id_last: int = struct.field(pytree_node=False)
    buf_explore_id_last: int = struct.field(pytree_node=False)

    set_explore_zero: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, task: ToyLevelsJax, set_explore_zero: bool = False) -> Self:
        return VizCustom(task, 0, 0, 0, set_explore_zero)

    @myjit
    def eval_nn(
        self,
        ppo: IntrinsicSumPPO,
        nsf: NSF,
        vds: VDSGAE,
        ci_classify: CIClassifier,
        pol_classify: CIClassifier,
    ):
        get_Vl = ppo.Vl_ext.apply

        b_x, b_obs, _ = self.task.get_eval_contour()

        b_Vl = jax_vmap(get_Vl)(b_obs).squeeze(-1)
        b_logprob = jax.vmap(nsf.log_prob)(b_obs)
        be_V = jax_vmap(vds.get_e_V)(b_obs)
        b_prob = jax.vmap(ci_classify.get_probs)(b_obs)
        b_prob_pol = jax.vmap(pol_classify.get_probs)(b_obs)

        return b_x, b_Vl, b_logprob, be_V, b_prob, b_prob_pol

    @myjit
    def eval_entropy(self, ppo: IntrinsicSumPPO, b_obs: BObs):
        def entropy(dist_, key: PRNGKey):
            return dist_.entropy(seed=key)

        def eval_entropy(obs_):
            dist = ppo.policy.apply(obs_)
            e_entropy = jax.vmap(ft.partial(entropy, dist))(b_key)
            return jnp.mean(e_entropy)

        n_entropy_samples = 32
        b_key = jr.split(jr.PRNGKey(12345), n_entropy_samples)

        b_entropy = jax.vmap(eval_entropy)(b_obs)
        return b_entropy

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

        # Concatenate all the observations together for the eval_rng. Pad it to get a constant length
        # so we don't need to recompile.
        lens = [len(r.T_obs_now) for r in props.rollout_eval_det]
        b_obs_flat = np.concatenate(
            [r.T_obs_now for r in props.rollout_eval_det], axis=0
        )
        n_obs = b_obs_flat.shape[1]

        # n_env * max_len
        b_max = len(props.rollout_eval_det) * (task.eval_rollout_T + 1)
        # pad it to (b_max, n_obs)
        n_pad = b_max - len(b_obs_flat)
        b_obs_pad = np.concatenate([b_obs_flat, np.zeros((n_pad, n_obs))], axis=0)
        assert b_obs_pad.shape == (b_max, n_obs)

        out = jax.device_get(self.eval_nn(ppo, nsf, vds, ci_classify, pol_classify))
        out2 = jax.device_get(self.eval_entropy(ppo, b_obs_pad))

        b_x, b_Vl, b_logprob, be_V, b_prob_ci, b_prob_polsafe = jax2np(out)
        b_entropy = jax2np(out2)

        # Split b_entropy back into the original lengths.
        b_entropy = b_entropy[: len(b_obs_flat)]
        split_at = np.cumsum(lens)[:-1]
        bT_entropy: list[np.ndarray] = np.split(b_entropy, split_at)

        # Compute the average entropy for each rollout, over each initial condition.
        b_entropy0 = np.array([T_entropy[0] for T_entropy in bT_entropy])
        b_meanentropy = np.array([np.mean(T_entropy) for T_entropy in bT_entropy])
        b_entropy_max = np.array([np.max(T_entropy) for T_entropy in bT_entropy])
        b_entropy_min = np.array([np.min(T_entropy) for T_entropy in bT_entropy])

        entropy_mean_flat = np.mean(b_entropy)

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

        b_vds = np.std(be_V, axis=1)
        b_V_max = np.max(be_V, axis=1)
        b_V_min = np.min(be_V, axis=1)

        b_x_rollouteval: np.ndarray = task.eval_ics()
        b_Vl_disc_rng, b_T_rng, b_Vl_disc_det, b_T_det = compute_rollout_metrics(props)

        # ----
        # Array for the boundaries of the jitter. Starting from zero,
        width = 0.04
        margin = 0.01
        hs_end = -np.arange(6) * (width + margin)
        hs_start = hs_end - width

        # ----
        rng = np.random.default_rng(seed=12345)

        nrow = 9
        figsize = np.array([8.0, 2.5 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")
        [task.label_ic(ax) for ax in axes]

        axes_Vl = axes[3]

        # -------------------------------
        ax = axes[0]
        ax.set_title("CI (New) origin")

        if buf.n_ci_buf > 0:
            b_x0_ci = task.leaf_to_minstate(buf.get_b_x0_ci())
            b_ic = task.to_icval(b_x0_ci)[self.x0_ci_idx_last :]
            b_source = b_x0_ci.source[self.x0_ci_idx_last :]

            source_labels = Source.labels_dict()
            colors_dict = Source.colors_dict()

            # Separate b_ic by source.
            b_ic_list = [b_ic[b_source == k] for k in Source]
            colors = [colors_dict.get(k, "C4") for k in Source]
            labels = [source_labels.get(k, "Unknown") for k in Source]

            # Stacked histogram + scatter at the bottom.
            bins = task.icval_bins(51)
            ax.hist(
                b_ic_list,
                color=colors,
                bins=bins,
                histtype="bar",
                stacked=True,
                label=labels,
            )
            ax.legend(
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )

            ylim = ax.get_ylim()

            def rescale(x):
                # [0, 1] -> [minval, maxval]
                return x * (ylim[1] - ylim[0]) + ylim[0]

            for ii, source in enumerate(Source):
                color = colors_dict[source]
                b_ic_source = b_ic[b_source == source]
                b_jitter = rng.uniform(
                    low=hs_start[ii], high=hs_end[ii], size=len(b_ic_source)
                )
                ax.scatter(b_ic_source, rescale(b_jitter), color=color, s=3)

        # -------------------------------
        ax = axes[1]
        ax.set_title("CI Classifier")

        ax.plot(b_x, b_prob_ci, color="C1")

        if buf.n_ci_buf > 0:
            # Scatter CI buffer points
            b_x0_ci = buf.get_b_x0_ci()
            b_ic = task.to_icval(task.leaf_to_minstate(b_x0_ci))
            n_pts = len(b_ic)

            ##
            b_ic_ci_sampled = info["CI/ic"]
            b_Vl_sampled = info["CI/Vl"]

            color_ci_candidates = "magenta"

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

            for ax in [axes[1], axes[2]]:
                b_jitter = rng.uniform(
                    low=hs_start[0], high=hs_end[0], size=len(b_ic_ci_sampled)
                )
                ax.scatter(
                    b_ic_ci_sampled,
                    b_jitter,
                    color=color_ci_candidates,
                    s=3,
                    label="CI Argmax Candidates",
                )

                # Also visualize the points on the Vl plot.
                axes_Vl.scatter(
                    b_ic_ci_sampled,
                    b_Vl_sampled,
                    color=color_ci_candidates,
                    s=3,
                    zorder=10,
                )

                b_jitter = rng.uniform(
                    low=hs_start[1], high=hs_end[1], size=len(b_ci_ic_new)
                )
                ax.scatter(
                    b_ci_ic_new, b_jitter, color="C1", s=3, label="CI Rehearsal (New)"
                )

                b_jitter = rng.uniform(
                    low=hs_start[2], high=hs_end[2], size=len(b_buf_ci_ic)
                )
                ax.scatter(b_buf_ci_ic, b_jitter, color="C3", s=3, label="CI Rehearsal")

                b_jitter = rng.uniform(
                    low=hs_start[3], high=hs_end[3], size=len(b_ic_new)
                )
                ax.scatter(b_ic_new, b_jitter, color="C0", s=3, label="CI (New)")

                b_jitter = rng.uniform(low=hs_start[4], high=hs_end[4], size=n_pts)
                ax.scatter(b_ic, b_jitter, color="C2", s=3, label="CI")

            self.buf_ci_id_last = id_now
            self.x0_ci_idx_last = len(b_ic) - 1

        axes[1].legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # -------------------------------
        ax = axes[2]
        ax.set_title("Policy Classifier")

        ax.plot(b_x, b_prob_polsafe, color="C1")
        # 1 minus, since low cost = high prob of safety.
        ax.plot(b_x_rollouteval, 1 - b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, 1 - b_Vl_disc_det, color="C5", label="Vl MC (det)")

        # -------------------------------
        ax = axes[3]
        ax.set_title("Vl")

        ax.plot(b_x_rollouteval, b_Vl_disc_rng, color="C4", label="Vl MC (rng)")
        ax.plot(b_x_rollouteval, b_Vl_disc_det, color="C5", label="Vl MC (det)")

        ax.plot(b_x, b_Vl, color="C1", alpha=0.9, label="Vl")

        ax.fill_between(b_x, b_V_max, b_V_min, color="C2", alpha=0.2)
        ax.plot(b_x, b_V_max, color="C2", alpha=0.5, label="VDS")
        ax.plot(b_x, b_V_min, color="C2", alpha=0.5)

        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # -------------------------------
        ax = axes[4]
        ax.set_title("VDS")
        ax.plot(b_x, b_vds)

        vds_min, vds_max = b_vds.min(), b_vds.max()

        # -------------------------------
        ax = axes[5]
        ax.set_title("Log Prob")
        ax.plot(b_x, b_logprob)

        logprob_min, logprob_max = b_logprob.min(), b_logprob.max()

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

            for minval, maxval, ax in [
                (-0.05, 1.0, axes[3]),
                (vds_min, vds_max, axes[4]),
                (logprob_min, logprob_max, axes[5]),
            ]:

                def rescale(x):
                    # [0, 1] -> [minval, maxval]
                    return x * (maxval - minval) + minval

                b_jitter = rng.uniform(
                    low=hs_start[0], high=hs_end[0], size=len(b_expl_ic_new)
                )
                ax.scatter(
                    b_expl_ic_new,
                    rescale(b_jitter),
                    color="C0",
                    s=3,
                    label="Explore (New)",
                )

                b_jitter = rng.uniform(
                    low=hs_start[1], high=hs_end[1], size=len(b_buf_expl_ic)
                )
                ax.scatter(
                    b_buf_expl_ic, rescale(b_jitter), color="C1", s=3, label="Explore"
                )

            self.buf_explore_id_last = id_now

        # Vl max is 1, so fix the ymax so we can more easily compare.
        ylim = np.array(axes[3].get_ylim())
        ylim[1] = 1.05
        axes[3].set_ylim(*ylim)

        axes[5].legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ----------------------------
        ax = axes[6]
        ax.set_title("Sampling distribution")

        source_labels = Source.labels_dict()
        colors_dict = Source.colors_dict()

        b_x0 = task.leaf_to_minstate(tree_stack(trajsaver.all_x0s(), axis=0, which=np))
        b_ic = task.to_icval(b_x0)
        b_source = b_x0.source

        # Separate b_ic by source.
        b_ic_list = [b_ic[b_source == k] for k in Source]
        colors = [colors_dict.get(k, "C4") for k in Source]
        labels = [source_labels.get(k, "Unknown") for k in Source]

        # Stacked histogram + scatter at the bottom.
        bins = task.icval_bins(51)
        ax.hist(
            b_ic_list,
            color=colors,
            bins=bins,
            histtype="bar",
            stacked=True,
            label=labels,
        )
        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        ylim = ax.get_ylim()

        # Jitter around the bottom of the plot.
        def rescale(x):
            # [0, 1] -> [minval, maxval]
            return x * (ylim[1] - ylim[0]) + ylim[0]

        b_jitter = rng.uniform(low=hs_start[0], high=hs_end[0], size=len(b_ic))
        ax.scatter(b_ic, rescale(b_jitter), color="C3", s=3)

        for ii, ic in enumerate(b_ic_list):
            b_jitter = rng.uniform(
                low=hs_start[ii + 1], high=hs_end[ii + 1], size=len(ic)
            )
            ax.scatter(ic, rescale(b_jitter), color=colors[ii], s=3)

        # -------------------------------
        ax = axes[7]
        ax.set_title("Steps Alive")
        ax.plot(b_x_rollouteval, b_T_rng, marker="o", ms=3, lw=0.9, color="C4")
        ax.plot(b_x_rollouteval, b_T_det, marker="o", ms=3, lw=0.9, color="C5")

        # -------------------------------
        ax = axes[8]
        ax.set_title("Entropy (Mean={:+.2e})".format(entropy_mean_flat))
        ax.plot(
            b_x_rollouteval,
            b_entropy0,
            marker="o",
            ms=3,
            lw=0.9,
            color="C1",
            label="k=0",
        )
        ax.plot(
            b_x_rollouteval, b_meanentropy, ms=3, lw=0.9, color="C2", label="Traj Mean"
        )
        ax.fill_between(
            b_x_rollouteval, b_entropy_min, b_entropy_max, color="C2", alpha=0.2
        )

        # Add a horizontal line to indicate the mean entropy (over the entire batch)
        ax.axhline(entropy_mean_flat, color="C0", label="Mean over flat")

        ax.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False
        )

        # ------------------------------------------
        for ax in axes:
            ylim = ax.get_ylim()
            # Highlight regions where b_prob_ci >= 0.5 using axvspan.
            ax.fill_between(
                b_x,
                0,
                1,
                where=b_prob_ci >= 0.5,
                color="C5",
                alpha=0.3,
                transform=ax.get_xaxis_transform(),
                zorder=0.7,
            )
            ax.set_ylim(ylim)

        # ------------------------------------------
        name = "custom"
        plot_dir = run_cfg.paths.eval_plots / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"{name}_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
