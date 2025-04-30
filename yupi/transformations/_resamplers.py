"""
This constains resampling functions for trajectories.
"""

from typing import Collection

import numpy as np

from yupi._differentiation import _get_coeff
from yupi.trajectory import Trajectory


def _get_k_value_neighbors(
    val: float, data: np.ndarray, k: int, _from: int
) -> tuple[int, int]:
    lower_bound, upper_bound = _from, _from
    for _ in range(k):
        look_forwards = (
            data[upper_bound] - val <= val - data[lower_bound] or lower_bound == 0
        )
        if look_forwards and upper_bound < len(data) - 1:
            upper_bound = min(upper_bound + 1, len(data) - 1)
        else:
            lower_bound = max(lower_bound - 1, 0)
    return lower_bound, upper_bound


def _interpolate_axis(
    axis_data: np.ndarray, old_t: np.ndarray, new_t: np.ndarray, order: int
) -> np.ndarray:
    new_t_idxs = np.searchsorted(old_t, new_t)
    assert isinstance(new_t_idxs, np.ndarray)
    new_dim = np.empty(len(new_t))
    for i, new_t_idx in enumerate(new_t_idxs):
        val = new_t[i]
        min_neighbor, max_neighbor = _get_k_value_neighbors(
            val, old_t, order, int(new_t_idx)
        )
        alphas = old_t[min_neighbor : max_neighbor + 1]
        _coeff = _get_coeff(val, alphas, M=1)
        new_dim[i] = np.sum(
            _coeff[0, len(alphas) - 1, :] * axis_data[min_neighbor : max_neighbor + 1]
        )
    return new_dim


def resample(
    traj: Trajectory,
    new_dt: float | None = None,
    new_t: Collection[float] | None = None,
    new_traj_id: str | None = None,
    order: int = 1,
) -> Trajectory:
    """
    Resamples a trajectory to a new dt or a new array of time.

    One of ``new_dt`` or ``new_t`` must be specified.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    new_dt: float | None
        New dt. By default None.
    new_t: Collection[float] | None
        New sample rate or array of time. By default None.
    new_traj_id : str | None
        New trajectory ID. By default None.
    order : int, optional
        How many points to use for the interpolation of each value. By default 2.

    Returns
    -------
    Trajectory
        Output trajectory.

    Raises
    ------
    ValueError
        If neither ``new_dt`` nor ``new_t`` is specified.
    ValueError
        If both ``new_dt`` and ``new_t`` are specified.
    """

    if traj.extra:
        raise ValueError(
            "Resampling is not supported for trajectories with extra data."
        )

    if new_t is not None and new_dt is not None:
        raise ValueError("new_t and new_dt cannot be both specified")
    if new_t is None and new_dt is None:
        raise ValueError("new_t or new_dt must be specified")

    from_dt = new_dt is not None

    new_t = (
        np.arange(traj.t[0], traj.t[-1], new_dt)
        if new_dt is not None
        else np.array(new_t)
    )
    new_dims: list[np.ndarray] = []
    old_t = traj.t

    for dim in range(traj.dim):
        dim_data = traj.r.component(dim)
        new_dim = _interpolate_axis(dim_data, old_t, new_t, order)
        new_dims.append(new_dim)

    if from_dt:
        return Trajectory(
            axes=new_dims,
            dt=new_dt,
            units=traj.units,
            traj_id=new_traj_id,
            diff_est=traj.diff_est,
            **traj.metadata,
        )
    return Trajectory(
        axes=new_dims,
        t=new_t,
        units=traj.units,
        traj_id=new_traj_id,
        diff_est=traj.diff_est,
        **traj.metadata,
    )


def subsample(
    traj: Trajectory, step: int = 1, new_traj_id: str | None = None
) -> Trajectory:
    """
    Sample the trajectory ``traj`` by removing evenly spaced
    points according to ``step``.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    step : int, optional
        Number of sample points or period. By default 1.
    new_traj_id : str | None
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Output trajectory.
    """

    points = traj.r[::step]
    t = traj.t[::step] if traj.t is not None else None
    new_extra = {k: v[::step] for k, v in traj.extra.items()}

    return Trajectory(
        points=points,
        t=t,
        units=traj.units,
        extra=new_extra,
        dt=step * traj.dt,
        traj_id=new_traj_id,
        diff_est=traj.diff_est,
        **traj.metadata,
    )
