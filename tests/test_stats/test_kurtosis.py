import pytest
from matplotlib import pyplot as plt

from yupi.stats.kurtosis import KurtosisStat, KurtosisTimeAvgStat, kurtosis_reference
from yupi.trajectory import Trajectory


@pytest.fixture
def traj() -> Trajectory:
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    return Trajectory(points=points)


@pytest.fixture
def traj1() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2)


@pytest.fixture
def traj2() -> Trajectory:
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(x=x, dt=2)


def test_kurtosis_stat(traj: Trajectory, traj1: Trajectory, traj2: Trajectory) -> None:
    r0 = kurtosis_reference([traj])
    assert r0 == 8

    r1 = kurtosis_reference([traj1, traj2])
    assert r1 == 1.0

    stat = KurtosisStat([traj1, traj2])
    assert stat.kurtosis == pytest.approx([0, 1, 1, 1])

    stat_t = KurtosisTimeAvgStat([traj1, traj2], lag=2)
    assert stat_t.kurtosis_mean == pytest.approx([0, 1.5])
    assert stat_t.kurtosis_std == pytest.approx([0, 0])


def test_plot_kurtosis(monkeypatch: pytest.MonkeyPatch, traj: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    KurtosisStat([traj]).plot()
