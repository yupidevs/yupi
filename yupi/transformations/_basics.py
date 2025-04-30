from typing import Sequence

import numpy as np

from yupi.trajectory import Trajectory
from yupi.vector import Vector


def add_polar_offset(traj: Trajectory, radius: float, angle: float) -> None:
    """
    Adds an offset given a point in polar coordinates.

    Parameters
    ----------
    radius : float
        Point's radius.
    angle : float
        Point's angle.

    Raises
    ------
    TypeError
        If the trajectory is not 2 dimensional.
    """
    if traj.dim != 2:
        raise TypeError(
            "Polar offsets can only be applied on 2 dimensional trajectories"
        )

    # From cartesian to polar
    x, y = traj.r.x, traj.r.y
    rad, ang = np.hypot(x, y), np.arctan2(y, x)

    rad += radius
    ang += angle

    # From polar to cartesian
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    traj.r = Vector([x, y]).T


def rotate_2d(traj: Trajectory, angle: float) -> None:
    """
    Rotates a trajectory around the center coordinates [0,0]

    Parameters
    ----------
    angle : float
        Angle in radians to rotate the trajectory.
    """
    add_polar_offset(traj, 0, angle)


def rotate_3d(
    traj: Trajectory, angle: float, vector: Sequence[float] | np.ndarray
) -> None:
    """
    Rotates a trajectory around a given vector.

    Parameters
    ----------
    vector : Collection[float]
        Vector to rotate the trajectory around.
    angle : float
        Angle in radians to rotate the trajectory.

    Raises
    ------
    TypeError
        If the trajectory is not 3 dimensional.
    ValueError
        If the vector has shape different than (3,).
    """
    if traj.dim != 3:
        raise TypeError(
            "3D rotations can only be applied on 3 dimensional trajectories"
        )

    vec: Vector = Vector(vector)
    if vec.shape != (3,):
        raise ValueError("The vector must have shape (3,)")

    vec = Vector(vec / vec.norm)
    v_x, v_y, v_z = vec[0], vec[1], vec[2]
    a_cos, a_sin = np.cos(angle), np.sin(angle)

    rot_matrix = np.array(
        [
            [
                v_x * v_x * (1 - a_cos) + a_cos,
                v_x * v_y * (1 - a_cos) - v_z * a_sin,
                v_x * v_z * (1 - a_cos) + v_y * a_sin,
            ],
            [
                v_x * v_y * (1 - a_cos) + v_z * a_sin,
                v_y * v_y * (1 - a_cos) + a_cos,
                v_y * v_z * (1 - a_cos) - v_x * a_sin,
            ],
            [
                v_x * v_z * (1 - a_cos) - v_y * a_sin,
                v_y * v_z * (1 - a_cos) + v_x * a_sin,
                v_z * v_z * (1 - a_cos) + a_cos,
            ],
        ]
    )
    traj.r = Vector(np.dot(traj.r, rot_matrix))
