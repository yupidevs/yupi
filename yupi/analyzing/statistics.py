import numpy as np
import scipy.stats
from yupi.analyzing import turning_angles, subsample_trajectory

# relative and cumulative turning angles
def estimate_turning_angles(trajectories, accumulate=False, 
                    degrees=False, centered=False):
    theta = [turning_angles(traj) for traj in trajectories]
    return np.concatenate(theta)


# Returns measured velocity samples on all the trajectories
# subsampling them at a given stem
def estimate_velocity_samples(trajectories, step):
    step = 1
    trajs_ = [subsample_trajectory(traj, step) for traj in trajectories]
    return np.concatenate([traj.velocity() for traj in trajs_])


# mean square displacement
# TODO: Fix this implementation for dim != 2 Traj
def estimate_msd(trajs, time_avg=True, lag=None):
    msd = []
    for traj in trajs:
        # ensemble average
        if not time_avg:
            dx = traj.x - traj.x[0]
            dy = traj.y - traj.y[0]
            dr_2 = dx**2 + dy**2

        # time average
        else:
            dr_2 = np.empty(lag)
            for lag_ in range(1, lag + 1):
                dx = traj.x[lag_:] - traj.x[:-lag_]
                dy = traj.y[lag_:] - traj.y[:-lag_]
                dr_2[lag_ - 1] = np.mean(dx**2 + dy**2)

        # append all squared displacements
        msd.append(dr_2)
    
    msd = np.transpose(msd)
    msd_mean = np.mean(msd, axis=1)
    msd_std = np.std(msd, axis=1)
    return msd_mean, msd_std


# get displacements for ensemble average and
# kurtosis for time average
# TODO: Fix this implementation for dim != 2 Traj
def estimate_kurtosis(trajs, time_avg=True, lag=None):
    kurtosis = []
    for traj in trajs:
        if not time_avg:
            dx = traj.x - traj.x[0]
            dy = traj.y - traj.y[0]
            kurt = np.sqrt(dx**2 + dy**2)

        # time average
        else:
            kurt = np.empty(lag)
            for lag_ in range(1, lag + 1):
                dx = traj.x[lag_:] - traj.x[:-lag_]
                dy = traj.y[lag_:] - traj.y[:-lag_]
                dr = np.sqrt(dx**2 + dy**2)
                kurt[lag_ - 1] = scipy.stats.kurtosis(dr, fisher=False)

        kurtosis.append(kurt)

    if not time_avg:
        return scipy.stats.kurtosis(kurtosis, axis=0, fisher=False)
    else:
        return np.mean(kurtosis, axis=0)


# velocity autocorrelation function
# TODO: Fix this implementation for dim != 2
def estimate_vacf(trajs, time_avg=True, lag=None):
    vacf = []
    for traj in trajs:
        vx = traj.x_velocity()
        vy = traj.y_velocity()

        # ensemble average
        if not time_avg:
            v1v2x = vx[0] * vx
            v1v2y = vy[0] * vy
            v1v2 = v1v2x + v1v2y

        # time average
        else:
            v1v2 = np.empty(lag)
            for lag_ in range(1, lag + 1):
                v1v2x = vx[:-lag_] * vx[lag_:]
                v1v2y = vy[:-lag_] * vy[lag_:]
                v1v2[lag_ - 1] = np.mean(v1v2x + v1v2y)
        
        # append all pair-wise veloctiy dot products
        vacf.append(v1v2)
    
    vacf = np.transpose(vacf)
    vacf_mean = np.mean(vacf, axis=1)
    vacf_std = np.std(vacf, axis=1)
    return vacf_mean, vacf_std
