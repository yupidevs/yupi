import numpy as np
import scipy
from yupi.analyzing import wrap_theta

# relative and cumulative turning angles
def estimate_turning_angles(trajectory, accumulate=False, 
                    degrees=False, centered=False):
    dx = trajectory.get_x_diff()
    dy = trajectory.get_y_diff()
    theta = np.arctan2(dy, dx)

    if not accumulate:
        theta = np.ediff1d(theta)  # relative turning angles
    else:
        theta -= theta[0]          # cumulative turning angles

    return wrap_theta(theta, degrees, centered)


# mean square displacement
# TODO: Fix this implementation for dim != 2 Traj
def estimate_msd(trajectories, time_avg=True, lag=None):
    dr2 = []
    for trajectory in trajectories:
        # ensemble average
        if not time_avg:
            dx_n = (trajectory.x - trajectory.x[0])**2
            dy_n = (trajectory.y - trajectory.y[0])**2
            dr_n = (dx_n + dy_n)
        # time average
        else:
            dr_n = np.empty(lag)
            for lag_ in range(1, lag + 1):
                dx_n = (trajectory.x[lag_:] - trajectory.x[:-lag_])**2
                dy_n = (trajectory.y[lag_:] - trajectory.y[:-lag_])**2
                dr_n[lag_ - 1] = np.mean(dx_n + dy_n)    
        dr2.append(dr_n)
    return np.transpose(dr2)


# get displacements for ensemble average and
# kurtosis for time average
# TODO: Fix this implementation for dim != 2 Traj
def estimate_kurtosis(trajectories, time_avg=True, lag=None):
    dr_k = []
    for trajectory in trajectories:
        if not time_avg:
            dx = trajectory.x - trajectory.x[0]
            dy = trajectory.y - trajectory.y[0]
            kurt = np.sqrt(dx**2 + dy**2)

        # time average
        else:
            kurt = np.empty(lag)
            for lag_ in range(1, lag + 1):
                dx = trajectory.x[lag_:] - trajectory.x[:-lag_]
                dy = trajectory.y[lag_:] - trajectory.y[:-lag_]
                dr = np.sqrt(dx**2 + dy**2)
                kurt[lag_ - 1] = scipy.stats.kurtosis(dr, fisher=False)

    if not time_avg:
        return scipy.stats.kurtosis(dr_k, axis=0, fisher=False)
    else:
        return np.mean(dr_k, axis=0)
