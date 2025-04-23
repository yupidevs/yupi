"""
This contains a series of decorators that check the consistency of
the trajectories according to different criteria.
"""

from functools import wraps
from itertools import pairwise
from typing import Any, Callable, List, TypeVar

import numpy as np

from yupi.exceptions import TrajectoryError, TrajectoryGroupError
from yupi.trajectory import _THRESHOLD, Trajectory

T = TypeVar("T")


class NotUniformTimeSpacedError(TrajectoryError):
    """Raised when a trajectory should be uniformly time spaced but it's not."""

    def __init__(self, traj: Trajectory) -> None:
        super().__init__(
            traj, f"Trajectory {traj.traj_id} must be uniformly time spaced"
        )
        self.traj = traj


class DifferentDtError(TrajectoryGroupError):
    """Raised when the trajectories should have the same dt but they don't."""

    def __init__(self, trajs: list[Trajectory]) -> None:
        super().__init__(trajs, "All trajectories must have the same 'dt'")
        self.trajs = trajs


class DifferentDimensionError(TrajectoryGroupError):
    """Raised when the trajectories should have the same dimension but they don't."""

    def __init__(self, trajs: list[Trajectory], dim: int | None = None) -> None:
        message = "All trajectories must have the same dimension"
        if dim is not None:
            message += f": {dim}"
        super().__init__(trajs, message)
        self.trajs = trajs
        self.dim = dim


class DifferentTimeVectorError(TrajectoryGroupError):
    """Raised when the trajectories should have the same time vector but they don't."""

    def __init__(self, trajs: list[Trajectory]) -> None:
        super().__init__(trajs, "All trajectories must have the same time vector")
        self.trajs = trajs


class DifferentLengthError(TrajectoryGroupError):
    """Raised when the trajectories should have the same length but they don't."""

    def __init__(self, trajs: list[Trajectory]) -> None:
        super().__init__(trajs, "All trajectories must have the same length")
        self.trajs = trajs


def check_uniform_time_spaced(func: Callable[..., T]) -> Callable[..., T]:
    """Check that the trajectories are uniformly time-spaced."""

    @wraps(func)
    def wrapper(trajs: List[Trajectory], *args: Any, **kwargs: Any) -> T:
        first_non_uniform_time_spaced = next(
            (t for t in trajs if abs(t.dt_std) > _THRESHOLD), None
        )
        if first_non_uniform_time_spaced is not None:
            raise NotUniformTimeSpacedError(first_non_uniform_time_spaced)
        return func(trajs, *args, **kwargs)

    return wrapper


def check_same_dt(func: Callable[..., T]) -> Callable[..., T]:
    """Check that the trajectories have the same dt."""

    @wraps(func)
    def wrapper(trajs: List[Trajectory], *args: Any, **kwargs: Any) -> T:
        dt = trajs[0].dt
        first_unequal_dt = next((t for t in trajs if abs(t.dt - dt) > _THRESHOLD), None)
        if first_unequal_dt is not None:
            raise DifferentDtError([trajs[0], first_unequal_dt])
        return func(trajs, *args, **kwargs)

    return wrapper


def check_same_dim(func: Callable[..., T]) -> Callable[..., T]:
    """Check that the trajectories have the same dimension."""

    @wraps(func)
    def wrapper(trajs: List[Trajectory], *args: Any, **kwargs: Any) -> T:
        dim = trajs[0].dim
        first_unequal_dim = next((t for t in trajs if t.dim != dim), None)
        if first_unequal_dim is not None:
            raise DifferentDimensionError([trajs[0], first_unequal_dim])
        return func(trajs, *args, **kwargs)

    return wrapper


def check_exact_dim(dim: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Check that the trajectories have the same given dimension.

    Parameters
    ----------
    dim : int
        The dimension to check.
    """

    def _check_exact_dim_decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(
            trajs: List[Trajectory], *args: Any, dim: int = dim, **kwargs: Any
        ) -> T:
            first_unequal_dim = next((t for t in trajs if t.dim != dim), None)
            if first_unequal_dim is not None:
                raise DifferentDimensionError([trajs[0], first_unequal_dim], dim)
            return func(trajs, *args, **kwargs)

        return wrapper

    return _check_exact_dim_decorator


def check_same_length(func: Callable[..., T]) -> Callable[..., T]:
    """Check that the trajectories have the same lenght."""

    @wraps(func)
    def wrapper(trajs: List[Trajectory], *args: Any, **kwargs: Any) -> T:
        if trajs:
            length = len(trajs[0])
            first_unequal_length = next((t for t in trajs if len(t) != length), None)
            if first_unequal_length is not None:
                raise DifferentLengthError([trajs[0], first_unequal_length])
        return func(trajs, *args, **kwargs)

    return wrapper


def check_same_t(func: Callable[..., T]) -> Callable[..., T]:
    """Check that the trajectories have the same time data."""

    @wraps(func)
    @check_same_length
    def wrapper(trajs: List[Trajectory], *args: Any, **kwargs: Any) -> T:
        if trajs:
            time_vec = trajs[0].t
            first_unequal_t = next(
                (
                    traj
                    for traj in trajs
                    if not np.allclose(time_vec, traj.t, atol=_THRESHOLD)
                ),
                None,
            )
            if first_unequal_t is not None:
                raise DifferentTimeVectorError([trajs[0], first_unequal_t])
        return func(trajs, *args, **kwargs)

    return wrapper
