import numpy as np
from yupi.analyzing.transformations import wrap_theta

def turning_angles(traj, accumulate=False, degrees=False, centered=False):
    """Docstring for turning angles

    Parameters
    ----------
    traj : Trajectory
        [description]
    accumulate : bool, optional
        [description], by default False
    degrees : bool, optional
        [description], by default False
    centered : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    dx = traj.x_diff()
    dy = traj.y_diff()
    theta = np.arctan2(dy, dx)

    if not accumulate:
        theta = np.ediff1d(theta)  # relative turning angles
    else:
        theta -= theta[0]          # cumulative turning angles

    return wrap_theta(theta, degrees, centered)