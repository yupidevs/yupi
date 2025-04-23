import pytest

from yupi.trajectory import Trajectory

APPROX_REL_TOLERANCE = 1e-10


def trajectories() -> list[Trajectory]:
    t_1 = Trajectory(points=[[1, 4], [2, 5], [3, 6]])
    t_2 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], dt=0.1)
    t_3 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], t=[0.0, 0.5, 2.0])
    return [t_1, t_2, t_3]


def compare_trajectories(t1: Trajectory, t2: Trajectory) -> None:
    pytest.approx(t1.t, t2.t, APPROX_REL_TOLERANCE)
    pytest.approx(t1.r, t2.r, APPROX_REL_TOLERANCE)
