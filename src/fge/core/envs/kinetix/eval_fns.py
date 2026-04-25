import ipdb
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, ListedColormap
from matplotlib.patches import Patch
from og.jax_utils import jax2np, jax_vmap

from fge.core.algos.buf_custom import BufCustom
from fge.core.algos.onpol.ppo import EvalProps, RunCfg
from fge.core.bits.classifier import CIClassifier
from fge.core.bits.state_reset_id import Source, StateResetId
from fge.core.envs.kinetix import lander


def plot_train_x0_lander(n_collects: int, run_cfg: RunCfg, props: EvalProps):
    ppo, trajsaver = props.ppo, props.trajsaver
    task: lander.LanderJax = ppo.task

    if trajsaver is None:
        if not hasattr(plot_train_x0_lander, "__printed_trajsaver_none_warning"):
            logger.warning("trajsaver is None, not plotting x0s.")
        return

    if len(trajsaver.x0s) == 0:
        if not hasattr(plot_train_x0_lander, "__printed_x0_warning"):
            logger.warning("len(trajsaver.x0s) == 0, not plotting x0s.")
        return

    b_pos = []
    b_source = []
    for x0 in trajsaver.x0s:
        # x0: ToyLevelsJax.State = task.leaf_to_minstate(traj.x0)
        x0: lander.LanderJax.State = task.leaf_to_state(x0)
        b_pos.append(x0.state_kinetix.polygon.position[1])
        b_source.append(x0.source)
    b_pos = np.stack(b_pos, axis=0)
    b_source = np.array(b_source)

    markers = ["x", "o", "^", "s", "D", "v"]

    figsize = (6, 6)
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    task: lander.LanderJax = ppo.task
    task.setup_plot(ax)

    for ii, source in enumerate(Source):
        b_is_source = b_source == source.value

        if not np.any(b_is_source):
            continue

        s_pos = b_pos[b_is_source]

        color = f"C{ii}"
        marker = markers[ii % len(markers)]

        ax.scatter(s_pos[:, 0], s_pos[:, 1], alpha=0.5, color=color, marker=marker, label=source.name)

    # Legend outside plot.
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Source")

    plot_dir = run_cfg.paths.train_plots / "x0"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"x0_{n_collects:05d}.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=250)
    plt.close(fig)


@struct.dataclass(frozen=False)
class VizCustomLander:
    env: lander.LanderJax = struct.field(pytree_node=False)

    x0_ci_idx_last: int = struct.field(pytree_node=False)
    buf_ci_id_last: int = struct.field(pytree_node=False)
    buf_explore_id_last: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, env: lander.LanderJax) -> "VizCustomLander":
        return cls(env, 0, 0, 0)

    @jax.jit
    def eval_classifiers(self, ci_clsfy: CIClassifier, pol_clsfy: CIClassifier):
        num = 32

        b_x = jnp.linspace(-1, 1, num)
        b_y = jnp.linspace(-1, 1, num)
        bb_X, bb_Y = jnp.meshgrid(b_x, b_y)
        bb_ic = jnp.stack([bb_X, bb_Y], axis=-1)

        bb_state = jax_vmap(self.env.reset_from_box, rep=2)(bb_ic)
        bb_obs = jax_vmap(self.env.get_obs, rep=2)(bb_state)

        bb_predci = jax_vmap(ci_clsfy.get_probs, rep=2)(bb_obs)
        bb_predpol = jax_vmap(pol_clsfy.get_probs, rep=2)(bb_obs)

        bb_X = bb_state.state_kinetix.polygon.position[:, :, 1, 0]
        bb_Y = bb_state.state_kinetix.polygon.position[:, :, 1, 1]

        return bb_X, bb_Y, bb_predci, bb_predpol

    def __call__(self, n_collects: int, run_cfg: RunCfg, props: EvalProps):
        task: lander.LanderJax = props.task

        ci_clsfy: CIClassifier = props.extra["ci_classify"]
        pol_clsfy: CIClassifier = props.extra["pol_classify"]

        # Visualize the CI classifiers.
        bb_X, bb_Y, bb_p_predci, bb_p_predpol = jax2np(self.eval_classifiers(ci_clsfy, pol_clsfy))

        # Take the points inside the CI buffer, scatter their positions.
        buf: BufCustom = props.extra["buf"]

        thresh = 0.5
        bb_predci = bb_p_predci >= thresh
        bb_predpol = bb_p_predpol >= thresh

        ncol = 3
        figsize = (5.2 * ncol, 5)
        fig, axes = plt.subplots(1, ncol, figsize=figsize)
        [self.env.setup_plot(ax) for ax in axes]

        # ------------------------------
        # Plot different colors for [neither, CI only, Pol only, both]
        ax = axes[0]
        ax.set_title("Summary")

        # Encode condition combinations as integers:
        # 0 = none true
        # 1 = cond1 only
        # 2 = cond2 only
        # 3 = both
        bb_Z = (bb_predci.astype(int)) + 2 * (bb_predpol.astype(int))

        # Colors for each combination
        colors = ["C0", "C1", "C2", "C3"]
        labels = [
            "!predCI, !predPol",
            "predCI, !predPol",
            "!predCI, predPol",
            "predCI, predPol",
        ]

        cmap = ListedColormap(colors)
        # im = ax.contourf(bb_X, bb_Y, bb_Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.5)
        ax.pcolormesh(bb_X, bb_Y, bb_Z, cmap=cmap, shading="auto", alpha=0.5)
        # cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])

        patches = [Patch(color=colors[i], label=labels[i]) for i in range(4)]
        # Legend at bottom.
        ax.legend(handles=patches, ncol=4, bbox_to_anchor=(0.5, -0.15), loc="upper center")

        # ------------------------------
        ax = axes[1]
        ax.set_title("CI Classifier (n_ci={})".format(buf.n_ci_buf))
        levels = 11
        im = ax.contourf(bb_X, bb_Y, bb_p_predci, levels, cmap="RdBu_r", norm=CenteredNorm(thresh), alpha=0.5)
        lines = ax.contour(bb_X, bb_Y, bb_p_predci, levels=[thresh], colors="k", linewidths=2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.add_lines(lines)

        # ------------------------------
        ax = axes[2]
        ax.set_title("Pol Classifier")
        im = ax.contourf(bb_X, bb_Y, bb_p_predpol, levels, cmap="RdBu_r", norm=CenteredNorm(thresh), alpha=0.5)
        lines = ax.contour(bb_X, bb_Y, bb_p_predpol, levels=[thresh], colors="k", linewidths=2)
        cbar = plt.colorbar(im, ax=ax)
        try:
            cbar.add_lines(lines)
        except:
            pass

        # ------------------------------
        if buf.n_ci_buf > 0:
            b_x0_ci = buf.get_b_x0_ci()
            b_state: lander.LanderJax.State = task.leaf_to_state(b_x0_ci)
            b_ic_pos = b_state.state_kinetix.polygon.position[:, 1, :]
            assert b_ic_pos.ndim == 2

            # Scatter.
            buf_ci = buf.buffer.buf_ci
            buf_ci_pts: StateResetId = buf_ci.all_valid()
            b_buf_ci_ic = np.array(task.to_icval(buf_ci_pts.state))

            id_now = int(buf_ci.id)
            ids_new = np.arange(self.buf_ci_id_last, id_now)
            idxs_new = ids_new % buf_ci.capacity
            # Remove duplicate points in idxs_new.
            idxs_new = np.unique(idxs_new)
            b_ci_ic_new = b_buf_ci_ic[idxs_new]
            b_ic_pos_old = b_ic_pos[: self.x0_ci_idx_last]
            b_ic_pos_new = b_ic_pos[self.x0_ci_idx_last :]

            # -------------------------------------
            ax = axes[1]
            ax.scatter(
                b_ic_pos_old[:, 0],
                b_ic_pos_old[:, 1],
                marker="o",
                edgecolor="C5",
                facecolor="none",
                alpha=0.7,
                s=20,
                zorder=9,
                label="CI",
            )
            ax.scatter(
                b_ic_pos_new[:, 0],
                b_ic_pos_new[:, 1],
                marker="$N$",
                color="C0",
                alpha=0.7,
                s=20,
                zorder=10,
                label="CI (New)",
            )

            self.buf_ci_id_last = id_now
            self.x0_ci_idx_last = len(b_ic_pos) - 1

        # -------------------------------------
        # wtf is going on with the policy classifier training?
        # Visualize the polcond distribution...
        rng = np.random.default_rng(seed=12334)
        n_viz = min(buf.polcond_obs0.size, 500)
        b_ic, b_label = buf.polcond_obs0.sample_ics(rng, size=n_viz)

        # Convert b_ic to positions.
        b_pos = task.pos_from_ic(b_ic)

        b_pos_safe = b_pos[b_label]
        b_pos_unsafe = b_pos[~b_label]

        ax = axes[2]
        ax.scatter(
            b_pos_unsafe[:, 0],
            b_pos_unsafe[:, 1],
            marker="x",
            color="C0",
            alpha=0.7,
            s=20,
            zorder=9,
            label="Pol Unsafe",
        )
        ax.scatter(
            b_pos_safe[:, 0],
            b_pos_safe[:, 1],
            marker="o",
            edgecolor="C1",
            facecolor="none",
            alpha=0.7,
            s=20,
            zorder=10,
            label="Pol Safe",
        )

        # ------------------------------
        plot_dir = run_cfg.paths.eval_plots / "clsfy"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"clsfy_{n_collects:05d}.jpg"
        fig.savefig(fig_path, bbox_inches="tight", dpi=250)
        plt.close(fig)
