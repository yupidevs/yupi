import pytest
from yupi import Trajectory, VelocityMethod, WindowType

APPROX_REL_TOLERANCE = 1e-10


def test_creation_by_xyz():
    Trajectory(x=[1, 2])
    Trajectory(x=[1, 2], y=[2, 3])
    Trajectory(x=[1, 2], y=[2, 3], z=[1, 4])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2], z=[2, 5])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 6], z=[5])

    with pytest.raises(ValueError):
        Trajectory(x=[2], y=[2, 6], z=[5, 8])


def test_creation_by_axes():
    Trajectory(axes=[[1, 2]])
    Trajectory(axes=[[1, 2], [2, 3]])
    Trajectory(axes=[[1, 2], [2, 3], [1, 4]])
    Trajectory(axes=[[1, 2], [2, 3], [1, 4], [7, 8]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2, 6], [5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[2], [2, 6], [5, 8]])


def test_creation_by_points():
    Trajectory(points=[[1, 2], [2, 3]])
    Trajectory(points=[[1, 2, 4], [2, 3, 2], [1, 4, 8]])
    Trajectory(points=[[1, 2, 7, 3], [2, 3, 5, 3]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2]])


def test_creation_with_time():
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1])
    Trajectory(x=[1, 2], y=[2, 3], dt=0.1)
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1], dt=0.1)
    Trajectory(x=[1, 2], y=[2, 3], t=[0.4, 0.5])
    Trajectory(x=[1, 2], y=[2, 3], t=[0.4, 0.5], dt=0.1, t0=0.4)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0, 0.1], dt=0.2)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], t=[0.4, 0.5], dt=0.1)


def test_creation_with_ang():
    Trajectory(x=[1, 2], y=[2, 3], ang=[0, 0.1])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 3], ang=[0.1])


def test_creation_general():
    Trajectory(x=[1, 2], y=[2, 3], t=[0, 1], ang=[0, 0], traj_id="test")
    Trajectory(x=[1, 2], y=[2, 3], dt=0.5, ang=[0, 1.2], traj_id="test")
    Trajectory(points=[[1, 2], [2, 3]], dt=0.5, ang=[0, 1.2], traj_id="test")
    Trajectory(axes=[[1, 2], [2, 3]], dt=0.5, t=[1, 1.5], t0=1, traj_id="test")


def test_velocity_estimation_methods():
    x = [1, 2, 4, 8, 16]

    Trajectory.global_vel_method(VelocityMethod.LINEAR_DIFF, WindowType.FORWARD)
    traj = Trajectory(x=x)

    assert traj.v == pytest.approx([1, 2, 4, 8, 8], rel=APPROX_REL_TOLERANCE)

    Trajectory.global_vel_method(VelocityMethod.LINEAR_DIFF)
    traj.set_vel_method(VelocityMethod.LINEAR_DIFF, WindowType.BACKWARD)

    assert traj.v == pytest.approx([1, 1, 2, 4, 8], rel=APPROX_REL_TOLERANCE)

    traj = Trajectory(x=x)

    assert traj.v == pytest.approx([3 / 2, 3 / 2, 3, 6, 6], rel=APPROX_REL_TOLERANCE)

    vel_est = {
        "method": VelocityMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 2,
    }

    traj = Trajectory(x=x, vel_est=vel_est)

    vel_est["accuracy"] = 3

    with pytest.raises(ValueError):
        traj.set_vel_method(**vel_est)

    vel_est["accuracy"] = 2

    traj = Trajectory(x=x, y=[i**2 for i in x], vel_est=vel_est)
