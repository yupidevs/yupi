import pytest
from matplotlib import pyplot as plt

from yupi.stats import MsdStat, MsdTimeAvgStat
from yupi.trajectory import Trajectory


@pytest.fixture
def traj1() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2)


@pytest.fixture
def traj2() -> Trajectory:
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(x=x, dt=2)


def test_msd_stat(traj1: Trajectory, traj2: Trajectory) -> None:
    stat = MsdStat([traj1, traj2])

    assert stat.msd_mean == pytest.approx([0.0, 68.125, 24.505, 115.625])
    assert stat.msd_std == pytest.approx([0, 4.125, 0.495, 5.375])

    stat_t = MsdTimeAvgStat([traj1, traj2], lag=2)
    assert stat_t.msd_mean == pytest.approx([37.595, 15.5025])
    assert stat_t.msd_std == pytest.approx([1.26166667, 1.4975])


def test_msd_stat_plot(monkeypatch: pytest.MonkeyPatch, traj1: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    MsdTimeAvgStat([traj1], lag=1).plot()
