"""
This contains filtering functions for the trajectories.
"""

from typing import Optional

import numpy as np

from yupi.trajectory import Trajectory

from yupi.trajectory import _THRESHOLD, Trajectory

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

    track_origin = traj.r[0]
    r = (traj-track_origin).r
    dt = np.ediff1d(traj.t)
    new_r = np.zeros_like(r)
    for i in range(len(traj) - 1):
        new_r[i + 1] = new_r[i] - gamma * (new_r[i] - r[i]) * dt[i]

    smooth_traj = Trajectory(
        points=new_r, t=traj.t, traj_id=new_traj_id, diff_est=traj.diff_est
    )
    return smooth_traj + track_origin



def exp_moving_average_filter(
        traj: Trajectory, alpha: float,tau:Optional[float] = None, new_traj_id: Optional[str] = None
):
    """
    Returns a smoothed version of the trajectory `traj`
    using the exponential moving average defined as

    s(0) = x(0)
    s(t_n) = alpha x(t_{n-1})  + (1-alpha) s(t_{n-1})

    If the the trajectory times are non-uniform then tau must be provided. The non-uniform time filter is
    computed as

    s(0) = x(0)
    alpha(t_n) = 1 - exp(-(t_n - t_{n-1}) / tau))
    s(t_n) = alpha(t_n) x(t_{n-1})  + (1-alpha(t_n)) s(t_{n-1})
    
    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    alpha : float
        Exponential smoothing paramter.
    tau: float [optional, default=None]
        Smoothing factor that must be provided if the trajectory timeseries is non-uniform.
    new_traj_id : Optional[str]
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """
    data = traj.r
    times = traj.t
    if tau is None and abs(traj.dt_std - 0) > _THRESHOLD:
        raise ValueError("All trajectories must be uniformly time spaced if tau is not provided")        
    n_times, _ = data.shape
    ema = np.zeros_like(data)    
    ema[0] = data[0]
    # The uniform time smoother can likely be simplified with a convolution
    # using scipy but I didn't want to bring in that dependency here
    for i in range(1, n_times):
        dt = times[i] - times[i - 1]
        if tau is not None:
            alpha = 1 - np.exp(-dt / tau)  # Adaptive smoothing factor
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    smooth_traj = Trajectory(
        points=ema, t=traj.t, traj_id=new_traj_id, diff_est=traj.diff_est
    )
    return smooth_traj
