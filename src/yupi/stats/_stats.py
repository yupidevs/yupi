"""
This contains all the statistical functions.
"""

# pylint: disable=too-many-arguments

import logging
from collections.abc import Callable

import numpy as np

from yupi.trajectory import Trajectory
from yupi.vector import Vector

logger = logging.getLogger(__name__)


def collect_at_step(
    trajs: list[Trajectory],
    step: int,
    warnings: bool = True,
    velocity: bool = False,
    func: Callable[[Vector], Vector] | None = None,
) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory at a given
    step.

    Parameters
    ----------
    trajs : list[Trajectory]
        List of trajectories.
    step : int
        Index of the collected vector of each trajectory.
    warnings : bool
        If True, warns if the trajectory is shorter than the step, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    func : Callable[[Vector], Vector] | None
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
    return collect(trajs, at=int(step), warnings=warnings, velocity=velocity, func=func)


def collect_at_time(
    trajs: list[Trajectory],
    time: float,
    warnings: bool = True,
    velocity: bool = False,
    func: Callable[[Vector], Vector] | None = None,
) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory at a given
    time.

    Parameters
    ----------
    trajs : list[Trajectory]
        List of trajectories.
    time : float
        Time of the collected vector of each trajectory.

        It is calculated using the trajectory's dt.
    warnings : bool
        If True, warns if the trajectory is shorter than the time, by default
        True.
    velocity : bool
        If True, the velocity of the trajectory is used, by default False.
    func : Callable[[Vector], Vector] | None
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
    return collect(trajs, at=float(time), warnings=warnings, velocity=velocity, func=func)


def collect_step_lagged(
    trajs: list[Trajectory],
    step: int,
    warnings: bool = True,
    velocity: bool = False,
    concat: bool = True,
    func: Callable[[Vector], Vector] | None = None,
) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory lagged by a
    given step.

    Parameters
    ----------
    trajs : list[Trajectory]
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
    func : Callable[[Vector], Vector] | None
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
    return collect(
        trajs, lag=int(step), concat=concat, warnings=warnings, velocity=velocity, func=func
    )


def collect_time_lagged(
    trajs: list[Trajectory],
    time: float,
    warnings: bool = True,
    velocity: bool = False,
    concat: bool = True,
    func: Callable[[Vector], Vector] | None = None,
) -> np.ndarray:
    """
    Collects the positional data (or velocity) of each trajectory lagged by a
    given time.

    Parameters
    ----------
    trajs : list[Trajectory]
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
    func : Callable[[Vector], Vector] | None
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
    return collect(
        trajs, lag=float(time), concat=concat, warnings=warnings, velocity=velocity, func=func
    )


def collect(
    trajs: list[Trajectory],
    lag: int | float | None = None,
    concat: bool = True,
    warnings: bool = True,
    velocity: bool = False,
    func: Callable[[Vector], Vector] | None = None,
    at: int | float | None = None,  # pylint: disable=invalid-name
) -> np.ndarray:
    """
    Collect general function.

    It can collect the data of each trajectory lagged by a given step or time
    (step if ``lag`` is ``int``, time if ``lag`` is ``float``). It can also
    collect the data of each trajectory at a given step or time (step if ``at``
    is ``int``, time if ``at`` is ``float``). Both ``lag`` and ``at``
    parameters can not be used at the same time.

    Parameters
    ----------
    trajs : list[Trajectory]
        Group of trajectories.
    lag : int | float | None
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
    func : Callable[[Vector], Vector] | None
        Function to apply to each resulting vector, by default None.
    at : int | float | None
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

    checks = [
        isinstance(lag, int),
        isinstance(lag, float),
        isinstance(at, int),
        isinstance(at, float),
    ]

    if sum(checks) == 0:
        lag = 0
        checks[0] = True
    if sum(checks) > 1:
        raise ValueError("You can not set `lag` and `at` parameters at the same time")
    is_lag = checks[0] or checks[1]

    data = []
    for traj in trajs:
        step = _resolve_step(traj, lag, "lag") if is_lag else _resolve_step(traj, at, "at")

        if step == 0:
            current_vec = _apply(func, traj.v if velocity else traj.r)
            data.append(current_vec if is_lag else current_vec[step])
            continue

        if warnings and step >= len(traj.r):
            logger.warning("Trajectory %s is shorten than %i samples", traj.traj_id, step)
            continue

        if not is_lag:  # Is at
            data.append(traj.r[step])
            continue

        lagged_vec = traj.r[step:] - traj.r[:-step]
        if velocity:
            lagged_vec /= traj.dt * step
        data.append(_apply(func, lagged_vec))

    if concat and is_lag:
        return np.concatenate(data)
    equal_len = np.all([len(d) == len(data[0]) for d in data])
    return np.array(data) if equal_len else np.array(data, dtype=object)


def _resolve_step(traj: Trajectory, value: int | float | None, name: str) -> int:
    """A float `lag` or `at` is in time units; an int is already a number of samples."""
    if value is None:
        raise ValueError(f"{name} must be set")
    return int(value / traj.dt) if isinstance(value, float) else int(value)


def _apply(func: Callable[[Vector], Vector] | None, vec: Vector) -> Vector:
    return vec if func is None else func(vec)
