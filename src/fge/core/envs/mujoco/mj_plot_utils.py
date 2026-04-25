import functools as ft

import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry as sg
from shapely import ops as so


class CapsuleArtist(plt.Polygon):
    @staticmethod
    def fromto(radius: float, fr: tuple, to: tuple, offset: tuple = (0, 0), **kwargs):
        fr = np.array(fr)
        to = np.array(to)
        halfheight = np.linalg.norm(to - fr) / 2
        rot = np.pi / 2 - np.arctan2(to[1] - fr[1], to[0] - fr[0])
        # rot = np.pi/2 - np.deg2rad(10)
        offset = np.array(offset) + (fr + to) / 2

        return CapsuleArtist(
            radius, halfheight, rot=-np.rad2deg(rot), offset=tuple(offset), **kwargs
        )

    def __init__(
        self,
        radius: float,
        halfheight: float,
        rot: float | None = None,
        offset: tuple | None = None,
        **kwargs
    ):
        """
        :param rot: Rotation angle in degrees.
        """
        self.radius = radius
        self.halfheight = halfheight

        b_points = get_b_points(radius, halfheight, rot, offset)

        super().__init__(b_points, **kwargs)


@ft.lru_cache
def get_b_points(
    radius: float,
    halfheight: float,
    rot: float | None = None,
    offset: tuple | None = None,
):
    rect = sg.box(-radius, -halfheight, radius, halfheight)
    circle_d = sg.Point(0, -halfheight).buffer(radius, quad_segs=4)
    circle_u = sg.Point(0, halfheight).buffer(radius, quad_segs=4)
    capsule_shape = so.unary_union([rect, circle_d, circle_u])
    # Rotate capsule_shape by rot if provided
    xs, ys = capsule_shape.exterior.xy
    xs, ys = np.asarray(xs), np.asarray(ys)
    b_points = np.stack([xs, ys], axis=1)

    # Rotate the points if rot is provided
    if rot is not None:
        rot_rad = np.deg2rad(rot)
        rotation_matrix = np.array(
            [[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]]
        )
        b_points = b_points @ rotation_matrix.T

    # Offset if offset is provided.
    if offset is not None:
        b_points += np.array(offset)

    return b_points


class CircleArtist(plt.Circle):
    def __init__(self, radius: float, **kwargs):
        super().__init__((0, 0), radius, **kwargs)