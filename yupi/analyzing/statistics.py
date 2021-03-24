import numpy as np
import scipy.stats
from yupi.analyzing import turning_angles, subsample_trajectory

# relative and cumulative turning angles
def estimate_turning_angles(trajs, accumulate=False, 
                    degrees=False, centered=False):
    theta = [turning_angles(traj) for traj in trajs]
    return np.concatenate(theta)


# Returns measured velocity samples on all the trajectories
# subsampling them at a given stem
def estimate_velocity_samples(trajs, step):
    step = 1
    trajs_ = [subsample_trajectory(traj, step) for traj in trajs]
    return np.concatenate([traj.velocity() for traj in trajs_])


# get position vector by components
def get_position_vector(traj):
    # get the components of the position
    r = traj.data[:traj.dim]

    # transpose to have time/dimension as first/second axis
    r = np.transpose(r)
    return r


# get velocity vector by components
def get_velocity_vector(traj):
    v = []
    if traj.dim <= 1:
        vx = traj.x_velocity()
        v.append(vx)  # append velocity x-component

    if traj.dim <= 2:
        vy = traj.y_velocity()
        v.append(vy)  # append velocity y-component

    if traj.dim <= 3:
        vz = traj.z_velocity()
        v.append(vz)  # append velocity z-component

    # transpose to have time/dimension as first/second axis
    v = np.transpose(v)
    return v


# mean square displacement (ensemble average)
def estimate_msd_ensemble(trajs):
    msd = []
    for traj in trajs:
        r = get_position_vector(traj)  # position vectors
        r2 = (r - r[0])**2             # square distances
        msd.append(r2)                 # append square distances
    
    # transpose to have time/trials as first/second axis
    msd = np.transpose(msd)
    return msd

# mean square displacement (time average)
def estimate_msd_time(trajs, lag):
    msd = []
    for traj in trajs:
        dr2 = np.empty(lag)
        dr2_ = 0
        for lag_ in range(1, lag + 1):
            # iterating over all dimensions
            for dim in traj.dim:
                r_i = traj.data[dim - 1]         # position coordinates
                dr_i = r_i[lag_:] - r_i[:-lag_]  # lag displacements
                dr2_ += dr_i**2                  # sum of square displacements
            
            # averaging over a single realization
            dr2[lag_ - 1] = np.mean(dr2_)
        
        # append all square displacements
        msd.append(dr2)
    
    # transpose to have time/trials as first/second axis
    msd = np.transpose(msd)
    return msd


# mean square displacement
def estimate_msd(trajs, time_avg=True, lag=None):
    if not time_avg:
        msd = estimate_msd_ensemble(trajs)   # ensemble average
    else:
        msd = estimate_msd_time(trajs, lag)  # time average

    msd_mean = np.mean(msd, axis=1)  # mean
    msd_std = np.std(msd, axis=1)    # standard deviation
    return msd_mean, msd_std


# velocity autocorrelation function (ensemble average)
def estimate_vacf_ensemble(trajs):
    vacf = []
    for traj in trajs:
        # cartesian velocity components
        v = get_velocity_vector(traj)

        # pair-wise dot product between velocities at t0 and t
        v0_dot_v = np.sum(v[0] * v, axis=1)
        
        # append all veloctiy dot products
        vacf.append(v0_dot_v)

    # transpose to have time/trials as first/second axis
    vacf = np.transpose(vacf)
    return vacf


# velocity autocorrelation function (time average)
def estimate_vacf_time(trajs, lag):
    vacf = []
    for traj in trajs:
        # cartesian velocity components
        v = get_velocity_vector(traj)

        # compute vacf for a single trajectory
        vacf_ = np.empty(lag)
        for lag_ in range(1, lag + 1):
            v1v2 = v[:-lag_] * v[lag_:]           # multiply components given lag
            v1_dot_v2 = np.sum(v1v2, axis=1)      # dot product for a given lag time
            vacf_[lag_ - 1] = np.mean(v1_dot_v2)  # averaging over a single realization

        # append the vacf for a every single realization
        vacf.append(vacf_)

    # transpose to have time/trials as first/second axis
    vacf = np.transpose(vacf)
    return vacf


# velocity autocorrelation function
def estimate_vacf(trajs, time_avg=True, lag=None):
    if not time_avg:
        vacf = estimate_vacf_ensemble(trajs)   # ensemble average
    else:
        vacf = estimate_vacf_time(trajs, lag)  # time average

    vacf_mean = np.mean(vacf, axis=1)  # mean
    vacf_std = np.std(vacf, axis=1)    # standard deviation
    return vacf_mean, vacf_std


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
