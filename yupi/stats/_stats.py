from typing import Callable, List, Tuple, Any, Union
import numpy as np
import logging
from yupi.trajectory import Trajectory
from yupi.vector import Vector
from yupi.transformations import subsample
from yupi._checkers import (
    _check_uniform_time_spaced,
    _check_same_dt,
    _check_same_dim,
    _check_exact_dim,
    _check_same_t
)


def collect_at_step(trajs: List[Trajectory], step: int, warnings: bool = True,
                    velocity: bool = False,
                    func: Callable[[Vector], Vector] = None) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory at a given
    step.

    Parameters
    ----------
    trajs : List[Trajectory]
        List of trajectories.
    step : int
        Index of the collected vector of each trajectory.
    warnings : bool
        If True, warns if the trajectory is shorter than the step, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    func : Callable[[Vector], Vector]
        Function to apply to the collected vector of each trajectory.
        By default, the identity function.

    Returns
    -------
    np.ndarray
        Array of collected data.

    See Also
    --------
    collect_at_time, collect_step_lagged, collect_time_lagged, collect
    """
    return collect(trajs, at=int(step), warnings=warnings, velocity=velocity,
                   func=func)

def collect_at_time(trajs: List[Trajectory], time: float, warnings: bool = True,
                    velocity: bool = False,
                    func: Callable[[Vector], Vector] = None) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory at a given
    time.

    Parameters
    ----------
    trajs : List[Trajectory]
        List of trajectories.
    time : float
        Time of the collected vector of each trajectory.

        It is calculated using the trajectory's dt.
    warnings : bool
        If True, warns if the trajectory is shorter than the time, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    func : Callable[[Vector], Vector]
        Function to apply to the collected vector of each trajectory.
        By default, the identity function.

    Returns
    -------
    np.ndarray
        Array of collected data.

    See Also
    --------
    collect_at_step, collect_step_lagged, collect_time_lagged, collect
    """
    return collect(trajs, at=float(time), warnings=warnings, velocity=velocity,
                   func=func)

def collect_step_lagged(trajs: List[Trajectory], step: int, warnings: bool = True,
                        velocity: bool = False, concat: bool = True,
                        func: Callable[[Vector], Vector] = None) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory lagged by a
    given step.

    Parameters
    ----------
    trajs : List[Trajectory]
        List of trajectories.
    step : int
        Number of steps to lag.
    warnings : bool
        If True, warns if the trajectory is shorter than the step, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    concat : bool
        If True, the data is concatenated, by default True.
    func : Callable[[Vector], Vector]
        Function to apply to the collected vector of each trajectory.
        By default, the identity function.

    Returns
    -------
    np.ndarray
        Array of collected data.

    See Also
    --------
    collect_at_step, collect_at_step, collect_time_lagged, collect
    """
    return collect(trajs, lag=int(step), concat=concat, warnings=warnings,
                   velocity=velocity, func=func)

def collect_time_lagged(trajs: List[Trajectory], time: float, warnings: bool = True,
                        velocity: bool = False, concat: bool = True,
                        func: Callable[[Vector], Vector] = None) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory lagged by a
    given time.

    Parameters
    ----------
    trajs : List[Trajectory]
        List of trajectories.
    time : float
        Time to lag.
    warnings : bool
        If True, warns if the trajectory is shorter than the step, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    concat : bool
        If True, the data is concatenated, by default True.
    func : Callable[[Vector], Vector]
        Function to apply to the collected vector of each trajectory.
        By default, the identity function.

    Returns
    -------
    np.ndarray
        Array of collected data.

    See Also
    --------
    collect_at_step, collect_at_time, collect_step_lagged, collect
    """
    return collect(trajs, lag=float(time), concat=concat, warnings=warnings,
                   velocity=velocity, func=func)


def collect(trajs: List[Trajectory], lag: Union[int, float] = None,
            concat: bool = True, warnings: bool = True, velocity: bool = False,
            func: Callable[[Vector], Any] = None,
            at: Union[int, float] = None) -> np.ndarray:
    """
    Collect general function.

    It can collect the data of each trajectory lagged by a given step or time
    (step if ``lag`` is ``int``, time if ``lag`` is ``float``). It can also
    collect the data of each trajectory at a given step or time (step if ``at``
    is ``int``, time if ``at`` is ``float``). Both ``lag`` and ``at``
    parameters can not be used at the same time.

    Parameters
    ----------
    trajs : List[Trajectory]
        Group of trajectories.
    lag : int or float, optional
        If int, the number of samples to lag. If float, the time to lag.
    concat : bool, optional
        If true each trajectory stracted data will be concatenated in
        a single array, by default True.
    warnings : bool, optional
        If true, warnings will be printed if a trajectory is shorter
        than the lag, by default True.
    velocity : bool, optional
        If true, the velocity will be returned (calculated using the
        lag if given), by default False.
    func : Callable[[Vector], Any], optional
        Function to apply to each resulting vector, by default None.
    at : Union[int, float], optional
        If int, the index of the collected vector in the trajectory. If
        float, it is taken as time and the index is calculated using
        the trajectory's dt.

    Returns
    -------
    np.ndarray
        Collected data.

    Raises
    ------
    ValueError
        If ``lag`` and ``at`` are given at the same time.
    """

    is_step = lag is not None and isinstance(lag, int)
    is_time = lag is not None and isinstance(lag, float)
    is_at_step = at is not None and isinstance(at, int)
    is_at_time = at is not None and isinstance(at, float)

    if is_step + is_time + is_at_step + is_at_time == 0:
        is_step = True
        lag = 0
    if is_step + is_time + is_at_step + is_at_time > 1:
        raise ValueError("You can not set `lag` and `at` parameters at the "
                         "same time")
    is_lag = is_step or is_time
    is_at = is_at_step or is_at_time


    data = []

    for traj in trajs:
        if is_lag:
            step = int(lag / traj.dt) if is_time else int(lag)
        else:
            step = int(at / traj.dt) if is_at_time else int(at)

        current_vec = traj.r
        if step == 0:
            if velocity:
                current_vec = traj.v
            if func is not None:
                current_vec = func(current_vec)
            data.append(current_vec if is_lag else current_vec[step])
            continue

        if warnings and step >= len(current_vec):
            logging.warning(f"Trajectory {traj.traj_id} is shorten than "
                            f"{step} samples")
            continue

        if is_at:
            data.append(current_vec[step])
            continue

        lagged_vec = current_vec[step:] - current_vec[:-step]
        if velocity:
            lagged_vec /= traj.dt * step

        if func is not None:
            lagged_vec = func(lagged_vec)

        data.append(lagged_vec)

    if concat and is_lag:
        return np.concatenate(data)
    equal_len = np.all(len(d) == len(data[0]) for d in data)
    return np.array(data) if equal_len else np.array(data, dtype=object)


@_check_same_dt
@_check_exact_dim(2)
@_check_uniform_time_spaced
def turning_angles_ensemble(trajs: List[Trajectory], accumulate=False,
                            degrees=False, centered=False,
                            wrap=True) -> np.ndarray:
    """
    Return a concatenation of all the turning angles that forms
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

    theta = [t.turning_angles(accumulate, degrees, centered, wrap) for t in trajs]
    return np.concatenate(theta)


@_check_same_dim
def speed_ensemble(trajs: List[Trajectory], step: int = 1) -> np.ndarray:
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

    trajs_ = [subsample(traj, step) for traj in trajs]
    return np.concatenate([traj.v.norm for traj in trajs_])


@_check_same_t
def msd_ensemble(trajs: List[Trajectory]) -> np.ndarray:
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

    _msd = []
    for traj in trajs:
        # Position vectors
        r = traj.r

        # Square displacements
        r_2 = (r - r[0])**2            # Square coordinates
        r2 = np.sum(r_2, axis=1)       # Square distances
        _msd.append(r2)                 # Append square distances

    # Transpose to have time/trials as first/second axis
    _msd = np.transpose(_msd)
    return _msd


@_check_same_dt
@_check_uniform_time_spaced
def msd_time(trajs: List[Trajectory], lag: int) -> np.ndarray:
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

    _msd = []
    for traj in trajs:
        # Position vectors
        r = traj.r

        # Compute msd for a single trajectory
        current_msd = np.empty(lag)
        for lag_ in range(1, lag + 1):
            # Lag displacement vectors
            dr = r[lag_:] - r[:-lag_]
            # Lag displacement
            dr2 = np.sum(dr**2, axis=1)
            # Averaging over a single realization
            current_msd[lag_ - 1] = np.mean(dr2)

        # Append all square displacements
        _msd.append(current_msd)

    # Transpose to have time/trials as first/second axis
    _msd = np.transpose(_msd)
    return _msd


@_check_same_dim
def msd(trajs: List[Trajectory], time_avg: bool = True,
        lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the array of mean square displacements and
        the array of standard deviations.
    """

    if not time_avg:
        _msd = msd_ensemble(trajs)   # Ensemble average
    elif lag is None:
        raise ValueError("You must set 'lag' param if 'time_avg' is True")
    else:
        _msd = msd_time(trajs, lag)  # Time average

    msd_mean = np.mean(_msd, axis=1)  # Mean
    msd_std = np.std(_msd, axis=1)    # Standard deviation
    return msd_mean, msd_std


@_check_same_t
def vacf_ensemble(trajs: List[Trajectory]) -> np.ndarray:
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

    _vacf = []
    for traj in trajs:
        # Cartesian velocity components
        v = traj.v

        # Pair-wise dot product between velocities at t0 and t
        v0_dot_v = np.sum(v[0] * v, axis=1)

        # Append all veloctiy dot products
        _vacf.append(v0_dot_v)

    # Transpose to have time/trials as first/second axis
    _vacf = np.transpose(_vacf)
    return _vacf


@_check_same_dt
@_check_uniform_time_spaced
def vacf_time(trajs: List[Trajectory], lag: int) -> np.ndarray:
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

    _vacf = []
    for traj in trajs:
        # Cartesian velocity components
        v = traj.v

        # Compute vacf for a single trajectory
        current_vacf = np.empty(lag)
        for lag_ in range(1, lag + 1):
            # Multiply components given lag
            v1v2 = v[:-lag_] * v[lag_:]

            # Dot product for a given lag time
            v1_dot_v2 = np.sum(v1v2, axis=1)

            # Averaging over a single realization
            current_vacf[lag_ - 1] = np.mean(v1_dot_v2)

        # Append the vacf for a every single realization
        _vacf.append(current_vacf)

    # Aranspose to have time/trials as first/second axis
    _vacf = np.transpose(_vacf)
    return _vacf


@_check_same_dim
def vacf(trajs: List[Trajectory], time_avg: bool = True,
        lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the array of velocity autocorrelation function
        and the array of standard deviations.
    """

    if not time_avg:
        _vacf = vacf_ensemble(trajs)   # Ensemble average
    elif lag is None:
        raise ValueError("You must set 'lag' param if 'time_avg' is True")
    else:
        _vacf = vacf_time(trajs, lag)  # Time average

    vacf_mean = np.mean(_vacf, axis=1)  # Mean
    vacf_std = np.std(_vacf, axis=1)    # Standard deviation
    return vacf_mean, vacf_std


def _kurtosis(arr):
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
        kurt = m4 / m2**2
        return kurt

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
    kurt = np.mean(k**2)

    return kurt

@_check_same_t
def kurtosis_ensemble(trajs: List[Trajectory]) -> np.ndarray:
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
    kurt = [_kurtosis(r_) for r_ in r]

    return np.array(kurt)


@_check_same_dt
@_check_uniform_time_spaced
def kurtosis_time(trajs: List[Trajectory], lag: int) -> np.ndarray:
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

    kurt = []
    for traj in trajs:
        current_kurt = np.empty(lag)
        for lag_ in range(lag):
            try:
                dr = traj.r[lag_:] - traj.r[:-lag_]
            except ValueError:
                current_kurt[lag_] = 0
                continue
            current_kurt[lag_] = _kurtosis(dr.T)
        kurt.append(current_kurt)
    kurt = np.transpose(kurt)
    return kurt


@_check_same_dim
def kurtosis(trajs: List[Trajectory], time_avg: bool = True,
             lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
        Tuple containgin the kurtosis and the standar deviations.
    """

    if not time_avg:
        return kurtosis_ensemble(trajs), None
    elif lag is None:
        raise ValueError("You must set 'lag' param if 'time_avg' is True")

    kurt = kurtosis_time(trajs, lag)
    kurt_mean = np.mean(kurt, axis=1)
    kurt_std = np.std(kurt, axis=1)
    return kurt_mean, kurt_std


@_check_same_dim
def kurtosis_reference(trajs: List[Trajectory]) -> float:
    """Get the sampled kurtosis for the case of
    ``len(trajs)`` trajectories whose position
    vectors are normally distributed.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input trajectories.

    Returns
    -------
    float
        Value of kurtosis.
    """

    d = trajs[0].dim
    n = len(trajs)
    kurt = d * (d + 2)
    if n == 1:
        return kurt
    return kurt * (n - 1) / (n + 1)


@_check_same_dt
@_check_uniform_time_spaced
def psd(trajs: List[Trajectory], lag: int, omega: bool = True) -> np.ndarray:
    """
    Estimate the power spectral density of a list of Trajectory object
    as the Fourier transform of its velocity autocorrelation function.

    Parameters
    ----------
    trajs : List[Trajectory]
        Input list of trajectories.
    lag : int
        Number of steps that multiplied by ``dt`` defines the lag
        time.
    omega: bool
        If True, return the angular frequency instead of the frequency.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        Power spectral density mean, standard deviation, and
        frequency axis.
    """

    _vacf = vacf_time(trajs, lag)
    ft = np.fft.fft(_vacf, axis=0) * trajs[0].dt
    ft = np.fft.fftshift(ft)
    ft_abs = np.abs(ft)
    ft_mean = np.mean(ft_abs, axis=1)
    ft_std = np.std(ft_abs, axis=1)

    frec = 2*np.pi * np.fft.fftfreq(lag, trajs[0].dt)
    frec = np.fft.fftshift(frec)

    return ft_mean, ft_std, frec * 2 * np.pi if omega else frec
