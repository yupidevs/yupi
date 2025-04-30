import pytest
from matplotlib import pyplot as plt

from yupi.stats.vacf import VacfStat, VacfTimeAvgStat
from yupi.trajectory import Trajectory


@pytest.fixture
def traj1() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2)


@pytest.fixture
def traj2() -> Trajectory:
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(x=x, dt=2)


def test_vacf_stat(traj1: Trajectory, traj2: Trajectory) -> None:
    stat = VacfStat([traj1, traj2])
    assert stat.vacf_mean == pytest.approx([17.03125, -6.825, 11.95, 11.95])
    assert stat.vacf_std == pytest.approx([1.03125, 0.825, 0.05, 0.05])

    stat_t = VacfTimeAvgStat([traj1, traj2], lag=2)
    assert stat_t.vacf_mean == pytest.approx([-3.54166667, 0.0])
    assert stat_t.vacf_std == pytest.approx([0.29166667, 0.0])


def test_vacf_stat_plot(monkeypatch: pytest.MonkeyPatch, traj1: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    VacfTimeAvgStat([traj1], lag=2).plot()
