"""
This contains filtering functions for the trajectories.
"""

from typing import Optional

import numpy as np

from yupi.trajectory import Trajectory


def exp_convolutional_filter(
    traj: Trajectory, gamma: float, new_traj_id: Optional[str] = None
):
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
    new_traj_id : Optional[str]
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """

    r = traj.r
    dt = np.ediff1d(traj.t)
    new_r = np.zeros_like(r)
    for i in range(len(traj) - 1):
        new_r[i + 1] = new_r[i] - gamma * (new_r[i] - r[i]) * dt[i]

    smooth_traj = Trajectory(
        points=new_r, t=traj.t, traj_id=new_traj_id, diff_est=traj.diff_est
    )
    return smooth_traj
