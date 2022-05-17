import pytest
from yupi import Trajectory, DiffMethod, WindowType

APPROX_REL_TOLERANCE = 1e-10


def test_creation_by_xyz():
    Trajectory(x=[1, 2, 3])
    Trajectory(x=[1, 2, 4], y=[2, 3, 6])
    Trajectory(x=[1, 2, 4], y=[2, 3, 6], z=[1, 4, 7])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2], z=[2, 5])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 6], z=[5])

    with pytest.raises(ValueError):
        Trajectory(x=[2], y=[2, 6], z=[5, 8])


def test_creation_by_axes():
    Trajectory(axes=[[1, 2, 3]])
    Trajectory(axes=[[1, 2, 5], [2, 3, 3]])
    Trajectory(axes=[[1, 2, 1], [2, 3, 9], [1, 4, 3]])
    Trajectory(axes=[[1, 2, 5], [2, 3, 3], [1, 4, 8], [7, 8, 7]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2, 6], [5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[2], [2, 6], [5, 8]])


def test_creation_by_points():
    Trajectory(points=[[1, 2], [2, 3], [6, 7]])
    Trajectory(points=[[1, 2, 4], [2, 3, 2], [1, 4, 8], [2, 6, 8]])
    Trajectory(points=[[1, 2, 7, 3], [2, 3, 5, 3], [3, 7, 2, 1]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2]])


def test_creation_with_time():
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2])
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], dt=0.1)
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2], dt=0.1)
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0.4, 0.5, 0.6])
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0.4, 0.5, 0.6], dt=0.1, t_0=0.4)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 1])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2], dt=0.2)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0.4, 0.5, 0.6], dt=0.1)


def test_creation_general():
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 1, 2], traj_id="test")
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], dt=0.5, traj_id="test")
    Trajectory(points=[[1, 2], [2, 3], [3, 6]], dt=0.5, traj_id="test")
    Trajectory(axes=[[1, 2, 4], [2, 3, 6]], dt=0.5, t=[1, 1.5, 2], t_0=1, traj_id="test")


def test_velocity_estimation_methods():
    x = [1, 2, 4, 8, 16]

    Trajectory.global_diff_method(DiffMethod.LINEAR_DIFF, WindowType.FORWARD)
    traj = Trajectory(x=x)

    assert traj.v == pytest.approx([1, 2, 4, 8, 8], rel=APPROX_REL_TOLERANCE)

    Trajectory.global_diff_method(DiffMethod.LINEAR_DIFF)
    traj.set_diff_method(DiffMethod.LINEAR_DIFF, WindowType.BACKWARD)

    assert traj.v == pytest.approx([1, 1, 2, 4, 8], rel=APPROX_REL_TOLERANCE)

    traj = Trajectory(x=x)

    assert traj.v == pytest.approx([3 / 2, 3 / 2, 3, 6, 6], rel=APPROX_REL_TOLERANCE)

    vel_est = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 2,
    }

    traj = Trajectory(x=x, diff_est=vel_est)

    vel_est["accuracy"] = 3

    with pytest.raises(ValueError):
        traj.set_diff_method(**vel_est)

    vel_est["accuracy"] = 2

    traj = Trajectory(x=x, y=[i**2 for i in x], diff_est=vel_est)
