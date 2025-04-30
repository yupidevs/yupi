import pytest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch

from yupi import Trajectory
from yupi.exceptions import TrajectoryError
from yupi.graphics import plot_2d
from yupi.graphics._trajs_plots import plot_3d, plot_vs_time


@pytest.fixture
def trajs() -> list[Trajectory]:
    t1 = Trajectory(x=[0, 0, 4], y=[0, 3, 3])
    t2 = Trajectory(x=[4, 7, 7], y=[4, 4, 8], t=[0, 0.1, 0.2])
    t3 = Trajectory(x=[1, 2, 3], y=[2, 3, 6])
    return [t1, t2, t3]


@pytest.fixture
def trajs_1d() -> list[Trajectory]:
    t1 = Trajectory(x=[0, 0, 4])
    t2 = Trajectory(x=[4, 7, 7], t=[0, 0.1, 0.2])
    t3 = Trajectory(x=[1, 2, 3])
    return [t1, t2, t3]


@pytest.fixture
def trajs_3d() -> list[Trajectory]:
    t1 = Trajectory(x=[0, 0, 4], y=[0, 3, 3], z=[0, 1, 2])
    t2 = Trajectory(x=[4, 7, 7], y=[4, 4, 8], z=[1, 2, 3], t=[0, 0.1, 0.2])
    t3 = Trajectory(x=[1, 2, 3], y=[2, 3, 6], z=[1, 4, 7])
    return [t1, t2, t3]


def test_wrong_dimension(
    trajs_1d: list[Trajectory], trajs: list[Trajectory], trajs_3d: list[Trajectory]
) -> None:
    with pytest.raises(TrajectoryError):
        plot_2d(trajs_1d, show=False)

    with pytest.raises(TrajectoryError):
        plot_2d(trajs_3d, show=False)

    with pytest.raises(TrajectoryError):
        plot_3d(trajs_1d, show=False)

    with pytest.raises(TrajectoryError):
        plot_3d(trajs, show=False)


def test_plot_2d(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    """Test the plot_2d function."""

    monkeypatch.setattr(plt, "show", lambda: None)

    # single trajectory
    plot_2d(trajs[0])

    # multiple trajectories
    plot_2d(trajs)

    # with color list
    plot_2d(trajs, color=["red", "blue", "green"])
    plot_2d(trajs, color=["red", "blue"])

    # connected with different length
    plot_2d(
        [*trajs, Trajectory(x=[1, 2, 3, 4], y=[2, 3, 6, 6], t=[0, 0.1, 0.2, 0.3])],
        connected=True,
    )

    # with other kwargs
    plot_2d(
        trajs,
        title="Test",
        color="red",
        units="nm",
        ax=plt.gca(),
        connected=True,
        legend=True,
    )

    plt.axes()


def test_plot_3d(monkeypatch: MonkeyPatch, trajs_3d: list[Trajectory]) -> None:
    """Test the plot_3d function."""

    monkeypatch.setattr(plt, "show", lambda: None)

    # single trajectory
    plot_3d(trajs_3d[0])

    # multiple trajectories
    plot_3d(trajs_3d)

    # with color list
    plot_3d(trajs_3d, color=["red", "blue", "green"])
    plot_3d(trajs_3d, color=["red", "blue"])

    # connected with different length
    plot_3d(
        [
            *trajs_3d,
            Trajectory(
                x=[1, 2, 3, 4], y=[2, 3, 6, 6], z=[1, 2, 3, 4], t=[0, 0.1, 0.2, 0.3]
            ),
        ],
        connected=True,
    )

    # with other kwargs
    plot_3d(
        trajs_3d,
        title="Test",
        color="red",
        units="nm",
        ax=plt.gca(),
        connected=True,
        legend=True,
    )

    plt.axes()


def test_plot_vs_time(monkeypatch: MonkeyPatch, trajs: list[Trajectory]) -> None:
    """Test the plot_vs_time function."""

    monkeypatch.setattr(plt, "show", lambda: None)

    # single trajectory
    plot_vs_time(trajs[0], lambda t: t.r.x)

    # multiple trajectories
    plot_vs_time(trajs, lambda t: t.r.x)

    # with color list
    plot_vs_time(trajs, lambda t: t.r.x, color=["red", "blue", "green"])
    plot_vs_time(trajs, lambda t: t.r.x, color=["red", "blue"])

    # with other kwargs
    plot_vs_time(
        trajs,
        lambda t: t.r.x,
        title="Test",
        y_label="X position",
        color="red",
        legend=True,
    )

    plt.axes()
