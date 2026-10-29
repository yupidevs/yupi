import pytest
from matplotlib import pyplot as plt

from yupi.stats import SpeedStat
from yupi.trajectory import Trajectory
from yupi.units import Units


@pytest.fixture
def traj() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2, units=Units.parse("km/h"))


def test_speed_stat(traj: Trajectory) -> None:
    assert SpeedStat([traj, traj]).speeds == pytest.approx([4, 1.5, 3, 3, 4, 1.5, 3, 3])


def test_speed_stat_plot(monkeypatch: pytest.MonkeyPatch, traj: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    ax = SpeedStat([traj]).plot(density=True)

    assert ax.get_xlabel() == "Speed [km/h]"
