import numpy as np
from yupi import Trajectory


def _wrap_ang(ang: np.ndarray, degrees=False, centered=False):
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


def turning_angles(traj: Trajectory, accumulate=False, degrees=False,
                   centered=False, wrap=True):
    """
    Return the sequence of turning angles that forms the trajectory.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    accumulate : bool, optional
        If True, turning angles are measured with respect to an axis
        defined by the initial velocity (i.e., angles between initial
        and current velocity). Otherwise, relative turning angles
        are computed (i.e., angles between succesive velocity
        vectors). By default False.
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
        Turning angles where each position in the array correspond
        to a given time instant.
    """

    dx = traj.delta_r.x
    dy = traj.delta_r.y
    theta = np.arctan2(dy, dx)

    if not accumulate:
        theta = np.ediff1d(theta)  # Relative turning angles
    else:
        theta -= theta[0]          # Accumulative turning angles

    if degrees:
        theta = np.rad2deg(theta)

    if wrap:
        return _wrap_ang(theta, degrees, centered)
    return theta
