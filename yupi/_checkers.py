"""
This contains a series of decorators that check the consistency of
the trajectories according to different criteria.
"""

from typing import TypeVar

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


def check_uniform_time_spaced(trajs: list[Trajectory]) -> None:
    """Check that the trajectories are uniformly time-spaced."""

    first_non_uniform_time_spaced = next(
        (t for t in trajs if abs(t.dt_std) > _THRESHOLD), None
    )
    if first_non_uniform_time_spaced is not None:
        raise NotUniformTimeSpacedError(first_non_uniform_time_spaced)


def check_same_dt(trajs: list[Trajectory]) -> None:
    """Check that the trajectories have the same dt."""

    dt = trajs[0].dt
    first_unequal_dt = next((t for t in trajs if abs(t.dt - dt) > _THRESHOLD), None)
    if first_unequal_dt is not None:
        raise DifferentDtError([trajs[0], first_unequal_dt])


def check_same_dim(trajs: list[Trajectory]) -> None:
    """Check that the trajectories have the same dimension."""

    dim = trajs[0].dim
    first_unequal_dim = next((t for t in trajs if t.dim != dim), None)
    if first_unequal_dim is not None:
        raise DifferentDimensionError([trajs[0], first_unequal_dim])


def check_exact_dim(trajs: list[Trajectory], dim: int) -> None:
    """
    Check that the trajectories have the same given dimension.

    Parameters
    ----------
    trajs : list[Trajectory]
        Input list of trajectories.
    dim : int
        The dimension to check.
    """

    first_unequal_dim = next((t for t in trajs if t.dim != dim), None)
    if first_unequal_dim is not None:
        raise DifferentDimensionError([trajs[0], first_unequal_dim], dim)


def check_same_length(trajs: list[Trajectory]) -> None:
    """Check that the trajectories have the same lenght."""

    length = len(trajs[0])
    first_unequal_length = next((t for t in trajs if len(t) != length), None)
    if first_unequal_length is not None:
        raise DifferentLengthError([trajs[0], first_unequal_length])


def check_same_t(trajs: list[Trajectory]) -> None:
    """Check that the trajectories have the same time data."""

    time_vec = trajs[0].t
    first_unequal_t = next(
        (traj for traj in trajs if not np.allclose(time_vec, traj.t, atol=_THRESHOLD)),
        None,
    )
    if first_unequal_t is not None:
        raise DifferentTimeVectorError([trajs[0], first_unequal_t])
