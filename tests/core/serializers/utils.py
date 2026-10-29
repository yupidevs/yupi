import pytest

from yupi.trajectory import Trajectory

APPROX_REL_TOLERANCE = 1e-10


def trajectories() -> list[Trajectory]:
    t_1 = Trajectory(points=[[1, 4], [2, 5], [3, 6]])
    t_2 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], dt=0.1)
    t_3 = Trajectory(points=[[1, 4], [2, 5], [3, 6]], t=[0.0, 0.5, 2.0])
    t_4 = Trajectory(x=[1, 2, 3], extra={"foo": [1, 2, 3]})
    return [t_1, t_2, t_3, t_4]


def compare_trajectories(t1: Trajectory, t2: Trajectory) -> None:
    _compare_time_and_positions(t1, t2)
    for key in t1.extra:
        assert key in t2.extra, f"Key {key} not found in second trajectory."
        assert t1.extra[key] == pytest.approx(t2.extra[key], rel=APPROX_REL_TOLERANCE)


def compare_trajectories_csv(t1: Trajectory, t2: Trajectory) -> None:
    """CSV stores extra data as text, so `t2` holds each extra value as a string."""
    _compare_time_and_positions(t1, t2)
    for key in t1.extra:
        assert key in t2.extra, f"Key {key} not found in second trajectory."
        assert t2.extra[key] == [str(value) for value in t1.extra[key]]


def _compare_time_and_positions(t1: Trajectory, t2: Trajectory) -> None:
    assert t1.t == pytest.approx(t2.t, rel=APPROX_REL_TOLERANCE)
    assert t1.r == pytest.approx(t2.r, rel=APPROX_REL_TOLERANCE)
