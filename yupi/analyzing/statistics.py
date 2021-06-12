from typing import List
import scipy.stats
import numpy as np
from yupi.trajectory import Trajectory
from yupi.analyzing import turning_angles, subsample_trajectory


def estimate_turning_angles(trajs: List[Trajectory], accumulate=False,
                            degrees=False, centered=False):
    """Return a concatenation of all the turning angles that forms
    a set of trajectories.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    accumulate : bool, optional
        If True, turning angles are measured with respect to an axis
        define by the initial velocity (i.e., angles between initial
        and current velocity). Otherwise, relative turning angles
        are computed (i.e., angles between succesive velocity vectors).
        By default False.
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
        Concatenated array of turning angles for a list of Trajectory
        objects.
    """

    theta = [turning_angles(traj) for traj in trajs]
    return np.concatenate(theta)


def estimate_velocity_samples(trajs: List[Trajectory], step: int = 1):
    """
    Estimate speeds of the list of trajectories, ``trajs``,
    by computing displacements according to a certain sample
    frequency given by ``step``.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    step : int
        Numer of sample points.

    Returns
    -------
    np.array
        Concatenated array of speeds.
    """

    trajs_ = [subsample_trajectory(traj, step) for traj in trajs]
    return np.concatenate([traj.v.norm for traj in trajs_])


def estimate_msd_ensemble(trajs: List[Trajectory]):
    """
    Compute the square displacements for every Trajectory object
    stored in ``trajs`` as the square of the current position vector
    that has been subtracted the initial position.

    Trajectories should have the same length.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.

    Returns
    -------
    np.ndarray
        Array of square displacements with shape ``(n, N)``, where
        ``n`` is the total number of time steps and ``N`` the number
        of trajectories.
    """

    msd = []
    for traj in trajs:
        # Position vectors
        r = traj.r

        # Square displacements
        r_2 = (r - r[0])**2            # Square coordinates
        r2 = np.sum(r_2, axis=1)       # Square distances
        msd.append(r2)                 # Append square distances

    # Transpose to have time/trials as first/second axis
    msd = np.transpose(msd)
    return msd


def estimate_msd_time(trajs: List[Trajectory], lag: int):
    """
    Estimate the mean square displacement for every Trajectory
    object stored in ``trajs`` as the average of the square of
    dispacement vectors as a function of the lag time.

    This is a convenience estimator specially when trajectories
    do not have equal lengths.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.

    Returns
    -------
    np.ndarray
        Array of mean square displacements with shape ``(lag, N)``,
        where ``N`` the number of trajectories.
    """

    msd = []
    for traj in trajs:
        # Position vectors
        r = traj.r

        # Compute msd for a single trajectory
        msd_ = np.empty(lag)
        for lag_ in range(1, lag + 1):
            # Lag displacement vectors
            dr = r[lag_:] - r[:-lag_]
            # Lag displacement
            dr2 = np.sum(dr**2, axis=1)
            # Averaging over a single realization
            msd_[lag_ - 1] = np.mean(dr2)

        # Append all square displacements
        msd.append(msd_)

    # Transpose to have time/trials as first/second axis
    msd = np.transpose(msd)
    return msd


def estimate_msd(trajs: List[Trajectory], time_avg=True, lag=None):
    """
    Estimate the mean square displacement of the list of Trajectory
    objects, ``trajs``, providing the options of averaging over the
    ensemble of realizations or over time.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    time_avg : bool, optional
        If True, mean square displacement is estimated averaging over
        time. Otherwise, an ensemble average will be performed and all
        Trajectory objects will have to have the same length. By default
        True.
    lag : None, optional
        If None, ``time_avg`` should be set to ``False`` indicating
        ensemble average. Otherwise, ``lag`` is taken as the number
        of steps that multiplied by ``dt`` defines the lag time. By
        default None.

    Returns
    -------
    np.ndarray
        Array of mean square displacements.
    np.ndarray
        Array of standard deviations.
    """

    if not time_avg:
        msd = estimate_msd_ensemble(trajs)   # Ensemble average
    else:
        msd = estimate_msd_time(trajs, lag)  # Time average

    msd_mean = np.mean(msd, axis=1)  # Mean
    msd_std = np.std(msd, axis=1)    # Standard deviation
    return msd_mean, msd_std


def estimate_vacf_ensemble(trajs: List[Trajectory]):
    """
    Compute the pair-wise dot product between initial and current
    velocity vectors for every Trajectory object stored in ``trajs``.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.

    Returns
    -------
    np.ndarray
        Array of velocity dot products with shape ``(n, N)``, where
        ``n`` is the total number of time steps and ``N`` the number
        of trajectories.
    """

    vacf = []
    for traj in trajs:
        # Cartesian velocity components
        v = traj.v

        # Pair-wise dot product between velocities at t0 and t
        v0_dot_v = np.sum(v[0] * v, axis=1)

        # Append all veloctiy dot products
        vacf.append(v0_dot_v)

    # Transpose to have time/trials as first/second axis
    vacf = np.transpose(vacf)
    return vacf


def estimate_vacf_time(trajs: List[Trajectory], lag: int):
    """
    Estimate the velocity autocorrelation function for every
    Trajectory object stored in ``trajs`` as the average of the
    dot product between velocity vectors that are distant a certain
    lag time.

    This is a convenience estimator specially when trajectories do
    not have equal lengths.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.

    Returns
    -------
    np.ndarray
        Array of velocity autocorrelation function with shape
        ``(lag, N)``, where ``N`` is the number of trajectories.
    """

    vacf = []
    for traj in trajs:
        # Cartesian velocity components
        v = traj.v

        # Compute vacf for a single trajectory
        vacf_ = np.empty(lag)
        for lag_ in range(1, lag + 1):
            # Multiply components given lag
            v1v2 = v[:-lag_] * v[lag_:]

            # Dot product for a given lag time
            v1_dot_v2 = np.sum(v1v2, axis=1)

            # Averaging over a single realization
            vacf_[lag_ - 1] = np.mean(v1_dot_v2)

        # Append the vacf for a every single realization
        vacf.append(vacf_)

    # Aranspose to have time/trials as first/second axis
    vacf = np.transpose(vacf)
    return vacf


def estimate_vacf(trajs: List[Trajectory], time_avg=True, lag: int = None):
    """
    Estimate the velocity autocorrelation function of the list of
    Trajectory objects, ``trajs``, providing the options of averaging
    over the ensemble of realizations or over time.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    time_avg : bool, optional
        If True, velocity autocorrelation function is estimated
        averaging over time. Otherwise, an ensemble average will be
        performed and all Trajectory objects will have to have the
        same length. By default True.
    lag : int, optional
        If None, ``time_avg`` should be set to ``False`` indicating
        ensemble average. Otherwise, ``lag`` is taken as the number
        of steps that multiplied by ``dt`` defines the lag time.
        By default None.

    Returns
    -------
    np.ndarray
        Array of velocity autocorrelation function.
    np.ndarray
        Array of standard deviations.
    """

    if not time_avg:
        vacf = estimate_vacf_ensemble(trajs)   # Ensemble average
    else:
        vacf = estimate_vacf_time(trajs, lag)  # Time average

    vacf_mean = np.mean(vacf, axis=1)  # Mean
    vacf_std = np.std(vacf, axis=1)    # Standard deviation
    return vacf_mean, vacf_std


# kurtosis (ensemble average)
# TODO: Fix this implementation for dim != 2 Traj
def estimate_kurtosis_ensemble(trajs: List[Trajectory]):
    """[summary]

    Parameters
    ----------
    trajs : List[Trajectory]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    kurtosis = []
    for traj in trajs:
        dx = traj.delta_r.x
        dy = traj.delta_r.y
        kurt = np.sqrt(dx**2 + dy**2)
        kurtosis.append(kurt)
    return scipy.stats.kurtosis(kurtosis, axis=0, fisher=False)


# kurtosis (time average)
# TODO: Fix this implementation for dim != 2 Traj
def estimate_kurtosis_time(trajs: List[Trajectory], lag):
    """[summary]

    Parameters
    ----------
    trajs : List[Trajectory]
        [description]
    lag : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    kurtosis = []
    for traj in trajs:
        kurt = np.empty(lag)
        for lag_ in range(1, lag + 1):
            dx = traj.r.x[lag_:] - traj.r.x[:-lag_]
            dy = traj.r.y[lag_:] - traj.r.y[:-lag_]
            dr = np.sqrt(dx**2 + dy**2)
            kurt[lag_ - 1] = scipy.stats.kurtosis(dr, fisher=False)
        kurtosis.append(kurt)
    return np.mean(kurtosis, axis=0)


# get displacements for ensemble average and
# kurtosis for time average
def estimate_kurtosis(trajs: List[Trajectory], time_avg=True, lag=None):
    """[summary]

    Parameters
    ----------
    trajs : List[Trajectory]
        [description]
    time_avg : bool, optional
        [description], by default True
    lag : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    if not time_avg:
        return estimate_kurtosis_ensemble(trajs)
    return estimate_kurtosis_time(trajs, lag)
