from typing import Tuple
import numpy as np
from yupi import Trajectory
from yupi.affine_estimator import affine_matrix


def add_dynamic_reference(traj: Trajectory,
                          reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
                          start_at_origin=True):
    """
    This function fuses the information of a trajectory with an
    external reference of the motion of the Frame of Reference
    (FoR).

    It allows to remap the information gathered in local SoRs
    to a more general FoR.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    reference : Tuple[np.ndarray,np.ndarray,np.ndarray]
        Angular and translational parameters of the form
        ``(ang:np.ndarray, tx:np.ndarray, ty:np.ndarray)`` that
        accounts for the orientation and displacement of the reference.
    start_at_origin : bool, optional
        If True, set initial position at the origin. By default True.

    Returns
    -------
    Trajectory
        Output trajectory in the lab frame of reference.
    """

    def affine2camera(theta, tx, ty):
        x_cl, y_cl, theta_cl = np.zeros((3, theta.size + 1))
        theta_cl[1:] = np.cumsum(theta)

        for i in range(theta.size):
            A = affine_matrix(theta_cl[i + 1], x_cl[i], y_cl[i], R_inv=True)
            x_cl[i + 1], y_cl[i + 1] = A @ [-tx[i], -ty[i], 1]

        x_cl, y_cl, theta_cl = x_cl[1:], y_cl[1:], theta_cl[1:]
        return x_cl, y_cl, theta_cl

    def camera2obj(x_ac, y_ac, x_cl, y_cl, theta_cl):
        x_al, y_al = np.empty((2, x_ac.size))

        for i in range(x_ac.size):
            A = affine_matrix(theta_cl[i], x_cl[i], y_cl[i], R_inv=True)
            x_al[i], y_al[i] = A @ [x_ac[i], y_ac[i], 1]

        return x_al, y_al

    def affine2obj(theta, tx, ty, x_ac, y_ac):
        x_cl, y_cl, theta_cl = affine2camera(theta, tx, ty)
        x_al, y_al = camera2obj(x_ac, y_ac, x_cl, y_cl, theta_cl)
        return x_al, y_al

    theta, tx, ty = reference

    x_al, y_al = affine2obj(theta, tx, ty, traj.r.x, traj.r.y)

    if start_at_origin:
        x_al = x_al - x_al[0]
        y_al = y_al - y_al[0]

    traj.x = x_al
    traj.y = y_al

    return Trajectory(x=x_al, y=y_al, ang=traj.ang, t=traj.t, dt=traj.dt,
                      traj_id=traj.id)


def subsample_trajectory(traj: Trajectory, step=1, step_in_seconds=False):
    """
    Sample the trajectory ``traj`` by removing evenly spaced
    points according to ``step``.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    step : int, optional
        Number of sample points or period, depending on the value
        of ``step_in_seconds``. By default 1.
    step_in_seconds : bool, optional
        If True, ``step`` is considered as the number of sample
        points. Otherwise, ``step`` is interpreted as the sample
        period, in seconds. By default False.

    Returns
    -------
    Trajectory
        Output trajectory.
    """

    if step_in_seconds:
        step = int(step / traj.dt)

    points = traj.r[::step]
    ang = traj.ang[::step] if traj.ang is not None else None
    t = traj.t[::step] if traj.t is not None else None
    return Trajectory(points=points, t=t, ang=ang, dt=step*traj.dt)


def wrap_theta(ang: np.ndarray, degrees=False, centered=False):
    """
    Wrap angles by removing more than one lap to the
    trigonometic circle.

    Parameters
    ----------
    ang : np.ndarray
        Input array of angles.
    degrees : bool, optional
        If True, angles are given in degrees. Otherwise, the units
        are radians. By default False.
    centered : bool, optional
        If True, angles are wrapped on the interval ``[-pi, pi]``.
        Otherwise, the interval ``[0, 2*pi]`` is chosen. By default
        False.

    Returns
    -------
    np.ndarray
        A wrapped copy of ``ang``.
    """

    discont = 360 if degrees else 2 * np.pi
    if not centered:
        return ang % discont

    discont_half = discont / 2
    return -((discont_half - ang) % discont - discont_half)
