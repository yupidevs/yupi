"""
This constains resampling functions for trajectories.
"""

from typing import Optional

from yupi import Trajectory


def subsample(traj: Trajectory, step: int = 1, new_traj_id: Optional[str] = None):
    """
    Sample the trajectory ``traj`` by removing evenly spaced
    points according to ``step``.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    step : int, optional
        Number of sample points or period. By default 1.
    new_traj_id : Optional[str]
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Output trajectory.
    """

    points = traj.r[::step]
    t = traj.t[::step] if traj.t is not None else None

    subsampled_traj = Trajectory(
        points=points,
        t=t,
        dt=step * traj.dt,
        traj_id=new_traj_id,
        diff_est=traj.diff_est,
    )
    return subsampled_traj
