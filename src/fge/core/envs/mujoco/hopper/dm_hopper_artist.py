import functools as ft

import ipdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as mtransforms
from shapely import geometry as sg
from shapely import ops as so

from fge.core.envs.mujoco.mj_plot_utils import CapsuleArtist


class DMHopperArtist(plt.Artist):
    def __init__(
        self, qpos: np.ndarray, *, zorder: float = 4, alpha: float = 0.7, **kwargs
    ):
        super().__init__()

        self.nose_from = np.array([0.08, 0.13])
        self.nose_to = np.array([0.15, 0.14])
        self.nose_halfheight = np.linalg.norm(self.nose_to - self.nose_from) / 2
        self.nose_midpt = (self.nose_from + self.nose_to) / 2
        self.nose_rotangle = np.arctan2(
            self.nose_to[1] - self.nose_from[1], self.nose_to[0] - self.nose_from[0]
        )
        self.nose_rotangle_deg = np.degrees(self.nose_rotangle) - 90

        self.artists_ = self._get_artists(alpha, **kwargs)
        (
            self.torso,
            self.pelvis,
            self.thigh,
            self.calf,
            self.foot,
            self.root,
            self.nose,
        ) = self.artists_
        self.qpos = qpos

        # self.pelvis_jnt = plt.Line2D([0], [0], marker="o", ms=2.0, color="black")
        # self.thigh_jnt = plt.Line2D([0], [0], marker="o", ms=2.0, color="black")
        # self.calf_jnt = plt.Line2D([0], [0], marker="o", ms=2.0, color="black")
        # self.foot_jnt = plt.Line2D([0], [0], marker="o", ms=2.0, color="black")
        # self.jnts = [self.pelvis_jnt, self.thigh_jnt, self.calf_jnt, self.foot_jnt]
        self.jnts = []

        self.last_t = mtransforms.Affine2D()

        self.set_zorder(zorder)

    @property
    def artists(self) -> list[plt.Artist]:
        return [*self.artists_, *self.jnts]

    def set_figure(self, fig: plt.Figure):
        [artist.set_figure(fig) for artist in self.artists]

    def update_state(self, qpos: np.ndarray):
        self.qpos = qpos
        self.set_transform(self.last_t)
        self.stale = True

    def set_transform(self, t):
        qpos = self.qpos
        assert qpos.shape == (7,), "qpos wrong shape"

        X_W_torso = (
            mtransforms.Affine2D().rotate(-qpos[2]).translate(qpos[0], 1 + qpos[1])
        )
        X_W_torsogeom = (
            mtransforms.Affine2D().translate(0, (-0.05 + 0.2) / 2) + X_W_torso
        )
        X_W_nosegeom = (
            mtransforms.Affine2D().rotate(0.0).translate(*self.nose_midpt) + X_W_torso
        )

        pelvis_offset = -0.05
        pelvis_midpt = (0.0 + -0.15) / 2
        X_W_pelvis = (
            mtransforms.Affine2D().rotate(-qpos[3]).translate(0.0, pelvis_offset)
            + X_W_torso
        )
        X_W_pelvisgeom = mtransforms.Affine2D().translate(0, pelvis_midpt) + X_W_pelvis

        thigh_offset = -0.2
        thigh_midpt = (0.0 + -0.33) / 2
        X_W_thigh = (
            mtransforms.Affine2D().rotate(-qpos[4]).translate(0, thigh_offset)
            + X_W_pelvis
        )
        X_W_thighgeom = mtransforms.Affine2D().translate(0, thigh_midpt) + X_W_thigh

        calf_offset = -0.33
        calf_midpt = (0.0 + -0.32) / 2
        X_W_calf = (
            mtransforms.Affine2D().rotate(-qpos[5]).translate(0, calf_offset)
            + X_W_thigh
        )
        X_W_calfgeom = mtransforms.Affine2D().translate(0, calf_midpt) + X_W_calf

        foot_offset = -0.32
        X_W_foot = (
            mtransforms.Affine2D().rotate(-qpos[6]).translate(0, foot_offset) + X_W_calf
        )
        foot_midpt = (0.17 + -0.08) / 2
        X_W_footgeom = mtransforms.Affine2D().translate(foot_midpt, 0.0) + X_W_foot

        self.last_t = t
        # self.pelvis_jnt.set_transform(X_W_pelvis + t)
        # self.thigh_jnt.set_transform(X_W_thigh + t)
        # self.calf_jnt.set_transform(X_W_calf + t)
        # self.foot_jnt.set_transform(X_W_foot + t)

        self.torso.set_transform(X_W_torsogeom + t)
        self.nose.set_transform(X_W_nosegeom + t)
        self.pelvis.set_transform(X_W_pelvisgeom + t)
        self.thigh.set_transform(X_W_thighgeom + t)
        self.calf.set_transform(X_W_calfgeom + t)
        self.foot.set_transform(X_W_footgeom + t)
        self.root.set_transform(X_W_torso + t)

    def draw(self, renderer):
        [artist.draw(renderer) for artist in self.artists]

    def _get_artists(self, alpha: float = 0.7, **kwargs):
        c0, c1, c2, c3, c4 = "C0", "C1", "C2", "C4", "C6"
        if "facecolor" in kwargs:
            facecolor = kwargs.pop("facecolor")
            c0, c1, c2, c3, c4 = facecolor, facecolor, facecolor, facecolor, facecolor

        torso = CapsuleArtist(
            0.0653, (0.2 - -0.05) / 2, facecolor=c0, alpha=alpha, **kwargs
        )
        pelvis = CapsuleArtist(
            0.065, (0 - -0.15) / 2, facecolor=c1, alpha=alpha, **kwargs
        )
        thigh = CapsuleArtist(
            0.04, (0 - -0.33) / 2, facecolor=c2, alpha=alpha, **kwargs
        )
        calf = CapsuleArtist(0.03, (0 - -0.32) / 2, facecolor=c3, alpha=alpha, **kwargs)
        foot = CapsuleArtist(
            0.04, (0.17 - -0.08) / 2, rot=90, facecolor=c4, alpha=alpha, **kwargs
        )
        nose = CapsuleArtist(
            0.03,
            self.nose_halfheight,
            rot=self.nose_rotangle_deg,
            facecolor=c0,
            alpha=alpha,
            **kwargs
        )
        root = plt.Line2D([0], [0], marker="o", ms=2.0, color="black")

        return torso, pelvis, thigh, calf, foot, root, nose


def DMHopperArtistWhole(qpos: np.ndarray, **kwargs):
    """
    Instead of having a DMHopperArtist composed of multiple polygons, have just a single polygon.
    """
    artist = DMHopperArtist(qpos)
    artist.update_state(qpos)

    # Recreate the geometry of the hopper using Shapely and the transforms in artist.
    polys = []
    for geom in artist.artists:
        if not isinstance(geom, CapsuleArtist):
            continue

        geom: CapsuleArtist
        orig_xy = geom.get_xy()
        trans = geom.get_transform()
        # (N, 2)
        new_pts = trans.transform(orig_xy)

        # Create a polygon using shapely.
        poly = sg.Polygon(new_pts)
        polys.append(poly)

    # Merge the polygons using shapely.
    merged_poly = so.unary_union(polys)

    # Get the exterior of the merged polygon.
    xs, ys = merged_poly.exterior.xy
    xs, ys = np.asarray(xs), np.asarray(ys)
    b_points = np.stack([xs, ys], axis=1)

    # Create a new artist using the merged polygon.
    merged_artist = plt.Polygon(b_points, **kwargs)

    return merged_artist
