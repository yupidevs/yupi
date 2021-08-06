from typing import List
import numpy as np
from yupi.trajectory import Trajectory, _threshold
from yupi.analyzing import turning_angles, subsample_trajectory


def _check_uniform_time_spaced(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if any((abs(t.dt_std - 0) > _threshold for t in trajs)):
            raise ValueError('All trajectories must be uniformly time spaced')
        return func(trajs, *args, **kwargs)
    return wrapper

def _check_same_dt(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dt = trajs[0].dt
            if any((abs(t.dt - dt) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same 'dt'")
        return func(trajs, *args, **kwargs)
    return wrapper

def _check_same_dim(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dim = trajs[0].dim
            if any((t.dim != dim for t in trajs)):
                raise ValueError("All trajectories must have the same dimensions")
        return func(trajs, *args, **kwargs)
    return wrapper

def _check_exact_dim(dim):
    def _check_exact_dim_decorator(func):
        def wrapper(trajs: List[Trajectory], *args, dim=dim, **kwargs):
            if any((t.dim != dim for t in trajs)):
                raise ValueError(f"All trajectories must be {dim}-dimensional")
            return func(trajs, *args, **kwargs)
        return wrapper
    return _check_exact_dim_decorator

def _check_same_r0(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            r0 = trajs[0].r[0]
            if any((abs(t.r[0] - r0) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same initial position")
        return func(trajs, *args, **kwargs)
    return wrapper

def _check_same_lenght(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            length = len(trajs)
            if any((abs(len(t) - length) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same length")
        return func(trajs, *args, **kwargs)
    return wrapper

def _check_same_t(func):
    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            length = len(trajs)
            if any((abs(len(t) - length) > _threshold for t in trajs)):
                raise ValueError("All trajectories must have the same length")
        return func(trajs, *args, **kwargs)
    return wrapper

@_check_same_dt
@_check_exact_dim(2)
@_check_uniform_time_spaced
def estimate_turning_angles(trajs: List[Trajectory], accumulate=False,
                            degrees=False, centered=False, wrap=True):
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

    theta = [turning_angles(t, accumulate, degrees, centered, wrap) for t in trajs]
    return np.concatenate(theta)


@_check_same_dim
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


@_check_same_t
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


@_check_same_dt
@_check_uniform_time_spaced
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


@_check_same_dim
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


@_check_same_t
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


@_check_same_dt
@_check_uniform_time_spaced
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


@_check_same_dim
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


def _estimate_kurtosis(arr):
    """
    Compute the kurtosis of the array, `arr`.

    If `arr` is not a one-dimensional array, it should
    be a horizontal collection of column vectors.

    Parameters
    ----------
    arr : np.adarray
        Data for which the kurtosis is calculated.

    Returns
    -------
    float
        Kurtosis of the data set.
    """

    arr = np.squeeze(arr)

    # ONE-DIMENSIONAL CASE
    if len(arr.shape) == 1:
        # Subtract the mean position at every time instant
        arr_zm = arr - arr.mean()

        # Second and fourth central moments averaging
        # over repetitions
        m2 = np.mean(arr_zm**2)
        m4 = np.mean(arr_zm**4)

        # Compute kurtosis for those cases in which the
        # second moment is different from zero
        if m2 == 0:
            return 0
        kurtosis = m4 / m2**2
        return kurtosis

    # MULTIDIMENSIONAL CASE
    # arr should have shape (dim, trials)
    # (i.e., a horizontal sequence of column vectors)

    # Subtract the mean position
    arr_zm = arr - arr.mean(1)[:,None]

    try:
        # Inverse of the estimated covariance matrix
        cov_inv = np.linalg.inv(np.cov(arr))
    except np.linalg.LinAlgError:
        # Exception for the case of singular matrices
        return 0

    # Kurtosis definition for multivariate r.v.'s
    k = np.sum(arr_zm * (cov_inv @ arr_zm), axis=0)
    kurtosis = np.mean(k**2)

    return kurtosis

@_check_same_t
def estimate_kurtosis_ensemble(trajs: List[Trajectory]):
    """Estimate kurtosis as a function of time of the
    list of Trajectory objects, ``trajs``. The average
    is perform over the ensemble of realizations.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.

    Returns
    -------
    np.ndarray
        Kurtosis at every time instant.
    """
    # Get ensemble positions where axis 0/1/2 are 
    # in the order trials/time/dim
    r = [traj.r for traj in trajs]

    # Set trials as the last axis
    r = np.moveaxis(r, 0, 2)

    # Compute kurtosis at every time instant (loop over time)
    kurtosis = [_estimate_kurtosis(r_) for r_ in r]

    return np.array(kurtosis)


@_check_same_dt
@_check_uniform_time_spaced
def estimate_kurtosis_time(trajs: List[Trajectory], lag):
    """
    Estimate the kurtosis for every Trajectory object stored 
    in ``trajs``.

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

    kurtosis = []
    for traj in trajs:
        kurt = np.empty(lag)
        for lag_ in range(lag):
            try:
                dr = traj.r[lag_:] - traj.r[:-lag_]
            except ValueError:
                kurt[lag_] = 0
                continue
            kurt[lag_] = _estimate_kurtosis(dr.T)
        kurtosis.append(kurt)
    kurtosis = np.transpose(kurtosis)
    return kurtosis


@_check_same_dim
def estimate_kurtosis(trajs: List[Trajectory], time_avg=True, lag=None):
    """
    Estimate the kurtosis of the list of Trajectory objects, ``trajs``,
    providing the options of averaging over the ensemble of realizations
    or over time.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    time_avg : bool, optional
        If True, kurtosis is estimated averaging over time. Otherwise,
        an ensemble average will be performed and all Trajectory objects
        will have to have the same length. By default True.
    lag : int, optional
        If None, ``time_avg`` should be set to ``False`` indicating
        ensemble average. Otherwise, ``lag`` is taken as the number
        of steps that multiplied by ``dt`` defines the lag time.
        By default None.

    Returns
    -------
    np.ndarray
        Kurtosis.
    np.ndarray
        Standard deviations.
    """

    if not time_avg:
        kurt = estimate_kurtosis_ensemble(trajs)
        return kurt

    kurt = estimate_kurtosis_time(trajs, lag)
    kurt_mean = np.mean(kurt, axis=1)
    kurt_std = np.std(kurt, axis=1)
    return kurt_mean, kurt_std
