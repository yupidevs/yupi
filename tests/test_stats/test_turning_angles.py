import numpy as np
import pytest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch

from yupi._checkers import DifferentDimensionError
from yupi.exceptions import TrajectoryError
from yupi.stats import TurningAngleStat, turning_angles
from yupi.trajectory import Trajectory


@pytest.fixture
def traj() -> Trajectory:
    #       |
    #       |    [2]---[3]
    #       |     |
    # - - -[0]---[1]- - -
    #       |
    #       |
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    return Trajectory(points=points)


def test_turning_angles(traj: Trajectory) -> None:
    tae = turning_angles(traj)
    assert tae == pytest.approx([np.pi / 2, 3 * np.pi / 2])

    tae = turning_angles(traj, degrees=True, wrap=False)
    assert tae == pytest.approx([90, -90])

    with pytest.raises(TrajectoryError):
        turning_angles(Trajectory(x=[1, 2, 3, 4]))


def test_turning_angles_stat(traj: Trajectory) -> None:
    assert TurningAngleStat(trajs=[traj, traj]).theta == pytest.approx(
        [np.pi / 2, 3 * np.pi / 2, np.pi / 2, 3 * np.pi / 2]
    )

    assert TurningAngleStat(
        trajs=[traj, traj],
        degrees=True,
        wrap=False,
    ).theta == pytest.approx([90, -90, 90, -90])

    assert TurningAngleStat(
        trajs=[traj, traj],
        accumulate=True,
        degrees=True,
        centered=True,
        wrap=True,
    ).theta == pytest.approx([0, 90, 0, 0, 90, 0])

    with pytest.raises(DifferentDimensionError):
        TurningAngleStat(
            trajs=[Trajectory(x=[1, 2, 3, 4])],
        )


def test_turning_angles_stat_plot(monkeypatch: MonkeyPatch, traj: Trajectory) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)

    ax = plt.gca()
    with pytest.raises(ValueError):
        TurningAngleStat(trajs=[traj]).plot(ax=ax)

    TurningAngleStat(trajs=[traj]).plot()

    ax = plt.axes(projection="polar")
    TurningAngleStat(trajs=[traj]).plot(ax=ax)
    plt.axes()  # Reset to default axes
