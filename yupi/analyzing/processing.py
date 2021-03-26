import numpy as np
from yupi.analyzing.transformations import wrap_theta

def turning_angles(traj, accumulate=False, degrees=False, centered=False):
    """Return the sequence of turning angles that form the trajectory.

    Parameters
    ----------
    traj : Trajectory
        Trajectory object.
    accumulate : bool, optional
        If True, turning angles are measured with respect to an axis define by the \\
        initial velocity (i.e., angles between initial and current velocity). \\
        Otherwise, relative turning angles are computed (i.e., angles between \\
        succesive velocity vectors). By default False.
    degrees : bool, optional
        If True, angles are given in degrees. Otherwise, the units are radians. \\
        By default False.
    centered : bool, optional
        If True, angles are wrapped on the interval ``[-pi, pi]``. Otherwise, \\
        the interval ``[0, 2*pi]`` is chosen. By default False.

    Returns
    -------
    np.ndarray
        Turning angles where each position in the array correspond to a given \\
        time instant.
    """

    dx = traj.x_diff()
    dy = traj.y_diff()
    theta = np.arctan2(dy, dx)

    if not accumulate:
        theta = np.ediff1d(theta)  # relative turning angles
    else:
        theta -= theta[0]          # cumulative turning angles

    return wrap_theta(theta, degrees, centered)