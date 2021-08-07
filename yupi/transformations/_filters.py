import numpy as np
from yupi.trajectory import Trajectory


def exp_convolutional_filter(traj: Trajectory, gamma: float,
                             new_traj_id: str = None):
    """
    Returns a smoothed version of the trajectory `traj`
    by taking a weighted average over past values.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    gamma : float
        Inverse of the characteristic time window of
        the average.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """

    r = traj.r
    dt = np.ediff1d(traj.t)
    R = np.zeros_like(r)
    for i in range(len(traj) - 1):
        R[i + 1] = R[i] - gamma * (R[i] - r[i]) * dt[i]

    smooth_traj = Trajectory(points=R, t=traj.t, traj_id=new_traj_id)
    return smooth_traj
