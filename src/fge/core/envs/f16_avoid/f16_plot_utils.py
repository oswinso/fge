from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from fge.core.envs.f16_avoid.f16_avoid_jax import get_bins


class PlotGridCallback(Protocol):
    def __call__(
        self, xytup: tuple[int, int], vt: float, conedist: float, ax: plt.Axes
    ) -> None:
        """vt: rows, conedist: cols."""
        pass


class PlotFigCallback(Protocol):
    def __call__(self, fig: plt.Figure) -> None:
        pass


def plot_grid(grid_cb: PlotGridCallback, fig_cb: PlotFigCallback):
    # cd: conedist

    vt_lo, vt_hi = 5.52724563e02, 7.26505260e02
    cd_lo, cd_hi = 500, 1000

    n_vt = n_cd = 4

    b_vt, b_vt_edge = get_bins(vt_lo, vt_hi, n_vt)
    b_cd, b_cd_edge = get_bins(cd_lo, cd_hi, n_cd)

    nrow = n_vt
    ncol = n_cd
    figsize = np.array([ncol * 3, nrow * 3])
    fig = plt.figure(layout="constrained", figsize=figsize)
    subfigs = fig.subfigures(nrow, ncol)

    for ii in range(n_vt):
        vt = b_vt[ii]

        for jj in range(n_cd):
            conedist = b_cd[jj]

            subfig: plt.SubFigure = subfigs[ii, jj]
            # ax = axes[ii, jj]
            ax = subfig.add_subplot(111, polar=True)

            # On the first column, add a supylabel for the vT.
            if jj == 0:
                subfig.supylabel("{:.0f}".format(vt))
            else:
                subfig.supylabel(".", color="white")

            # On the last row, add a supxlabel for the conedist.
            if ii == n_vt - 1:
                subfig.supxlabel("{:.0f}".format(conedist))
            else:
                subfig.supxlabel(".", color="white")

            # b_trajlen, b_ic = data[(vt, conedist)]
            # b_bin = np.digitize(b_trajlen, boundaries) - 1
            # b_colors = [colors[b] for b in b_bin]
            #
            # # Scatter plot.
            # ax.scatter(b_conephi, b_aspect_deg, c=b_colors, s=5, alpha=0.9)

            grid_cb((ii, jj), vt, conedist, ax)

            def formatter(x, pos):
                x_deg = np.rad2deg(x)
                if x_deg == 0.0:
                    suffix = "\n(R)"
                elif np.allclose(x_deg, 90.0):
                    suffix = "\n(U)"
                elif np.allclose(x_deg, 180.0):
                    suffix = "\n(L)"
                elif np.allclose(x_deg, 270.0):
                    suffix = "\n(D)"
                else:
                    suffix = ""

                return rf"{x_deg:.0f}$\degree$" + suffix

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(r"{x:.0f}$\degree$")
            ax.tick_params(axis="y", which="major", labelsize=4)

            # Label up down left right to make the plot easier to understand.
            label: plt.Text
            for label in ax.get_xticklabels():
                text = label.get_text()
                if text[:-1] == "0":
                    label.set_horizontalalignment("left")
                if text[:-1] == "180":
                    label.set_horizontalalignment("right")

            aspect_lim_lo, aspect_lim_hi = 22, 45.8
            ax.set_ylim(aspect_lim_lo, aspect_lim_hi)

    fig_cb(fig)
    # # Create a figure legend.
    # legend_handles = [
    #     plt.Line2D([0], [0], marker="o", lw=0, label=bin_names[ii], mec=color, mfc=color, markersize=10)
    #     for ii, color in enumerate(colors)
    # ]
    # fig.legend(handles=legend_handles, loc="outside right upper", borderaxespad=0, frameon=False)

    fig.supxlabel("Cone Dist")
    fig.supylabel("VT")

    return fig, subfigs
