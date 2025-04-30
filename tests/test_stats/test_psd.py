import pytest
from matplotlib import pyplot as plt

from yupi.stats.psd import PsdStat
from yupi.trajectory import Trajectory


@pytest.fixture
def traj() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2)


def test_psd_stat(traj: Trajectory) -> None:
    stat = PsdStat([traj], lag=2, omega=True)
    assert stat.psd_mean == pytest.approx([6.5, 6.5])
    assert stat.psd_std == pytest.approx([0, 0])
    assert stat.frecuency == pytest.approx([-9.8696044, 0.0])


def test_psd_stat_plot(monkeypatch: pytest.MonkeyPatch, traj: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    PsdStat([traj], lag=2).plot_psd()
