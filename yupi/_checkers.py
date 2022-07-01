"""
This contains a series of decorators that check the consistency of
the trajectories according to different criteria.
"""

from typing import List

import numpy as np

from yupi.trajectory import _THRESHOLD, Trajectory


def check_uniform_time_spaced(func):
    """Check that the trajectories are uniformly time-spaced."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if any((abs(t.dt_std - 0) > _THRESHOLD for t in trajs)):
            raise ValueError("All trajectories must be uniformly time spaced")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


def check_same_dt(func):
    """Check that the trajectories have the same dt."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dt = trajs[0].dt
            if any((abs(t.dt - dt) > _THRESHOLD for t in trajs)):
                raise ValueError("All trajectories must have the same 'dt'")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


def check_same_dim(func):
    """Check that the trajectories have the same dimension."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            dim = trajs[0].dim
            if any((t.dim != dim for t in trajs)):
                raise ValueError("All trajectories must have the same dimensions")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


def check_exact_dim(dim):
    """
    Check that the trajectories have the same given dimension.

    Parameters
    ----------
    dim : int
        The dimension to check.
    """

    def _check_exact_dim_decorator(func):
        def wrapper(trajs: List[Trajectory], *args, dim=dim, **kwargs):
            if any((t.dim != dim for t in trajs)):
                raise ValueError(f"All trajectories must be {dim}-dimensional")
            return func(trajs, *args, **kwargs)

        wrapper.__doc__ = func.__doc__
        return wrapper

    return _check_exact_dim_decorator


def check_same_r0(func):
    """Check that the trajectories have the same initial position."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            r_0 = trajs[0].r[0]
            if any((abs(t.r[0] - r_0) > _THRESHOLD for t in trajs)):
                raise ValueError("All trajectories must have the same initial position")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


def check_same_lenght(func):
    """Check that the trajectories have the same lenght."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            length = len(trajs[0])
            if any((abs(len(t) - length) > _THRESHOLD for t in trajs)):
                raise ValueError("All trajectories must have the same length")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


def check_same_t(func):
    """Check that the trajectories have the same time data."""

    def wrapper(trajs: List[Trajectory], *args, **kwargs):
        if trajs:
            zipped_trajs = zip(trajs[:-1], trajs[1:])
            if not all(
                (np.allclose(t0.t, t1.t, atol=_THRESHOLD) for t0, t1 in zipped_trajs)
            ):
                raise ValueError("All trajectories must have the same 't' values")
        return func(trajs, *args, **kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper
