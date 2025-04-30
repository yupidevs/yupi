import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import (
    exp_convolutional_filter,
    exp_moving_average_filter,
    resample,
    rotate_2d,
    rotate_3d,
    subsample,
)


@pytest.fixture
def x() -> list[int]:
    return list(range(10))


@pytest.fixture
def traj(x: list[int]) -> Trajectory:
    return Trajectory(x=x)


@pytest.fixture
def non_zero_origin() -> list[int]:
    return [7, 7, 7]  # Initial position


@pytest.fixture
def simple_traj() -> Trajectory:
    return Trajectory(x=[0, 1], y=[0, 1])


@pytest.fixture
def constant_v_non_zero_origin_traj(non_zero_origin: np.ndarray) -> Trajectory:
    num_steps = 500
    noise_std = 0.8  # Standard deviation of Gaussian noise
    velocity = np.array([1.0, 0.5, 0.2])
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = non_zero_origin
    t_vals = list(range(num_steps))
    # Generate trajectory
    for t in range(1, num_steps):
        trajectory[t] = (
            trajectory[t - 1] + velocity * t + np.random.normal(0, noise_std, size=3)
        )
    return Trajectory(points=trajectory, t=t_vals)


def test_subsample(x: list[int], traj: Trajectory) -> None:
    sub_sample = subsample(traj, 2)

    assert len(sub_sample) == len(x) // 2


def test_resample_new_dt(x: list[int], traj: Trajectory) -> None:
    new_dt = 0.5
    new_traj = resample(traj, new_dt=new_dt)

    assert new_traj.dt == new_dt
    assert new_traj.r.x == pytest.approx(np.arange(0, 9, 0.5))
    assert new_traj.r[0] == pytest.approx(traj.r[0])


def test_resample_new_t(x: list[int], traj: Trajectory) -> None:
    new_t = [0, 1, 4, 7, 9]
    new_traj = resample(traj, new_t=new_t)

    assert new_traj.dt_std != 0
    assert new_traj.r.x == pytest.approx([0, 1, 4, 7, 9])


def test_threshold() -> None:
    dt = 1 / 3
    traj1 = Trajectory(x=np.arange(742), dt=dt)
    traj2 = Trajectory(x=np.arange(743), dt=dt)
    subsample(traj1)
    subsample(traj2)


def test_exp_convolution_origin(
    constant_v_non_zero_origin_traj: Trajectory, non_zero_origin: list[int]
) -> None:
    smooted_trajectory = exp_convolutional_filter(
        constant_v_non_zero_origin_traj, 1 / 100
    )
    assert smooted_trajectory.r[0] == pytest.approx(non_zero_origin)


def test_ema_origin(
    constant_v_non_zero_origin_traj: Trajectory, non_zero_origin: list[int]
) -> None:
    smooted_trajectory = exp_moving_average_filter(
        constant_v_non_zero_origin_traj, alpha=1 / 100
    )
    assert smooted_trajectory.r[0] == pytest.approx(non_zero_origin)


def test_rotation(simple_traj: Trajectory) -> None:
    # 45 degrees
    ang = np.pi / 4

    # [0, 0] -> [0,       0]
    # [1, 1] -> [0, sqrt(2)]
    rotate_2d(simple_traj, ang)

    assert simple_traj.r[0] == pytest.approx([0, 0])
    assert simple_traj.r[1] == pytest.approx([0, np.sqrt(2)])


def test_rotation_3d() -> None:
    traj = Trajectory(x=[0, 1], y=[0, 0], z=[0, 0])

    rotate_3d(traj, -np.pi / 2, [0, 0, 3])

    assert traj.r[0] == pytest.approx([0, 0, 0])
    assert traj.r[1] == pytest.approx([0, 1, 0])

    rotate_3d(traj, np.pi, [1, 0, 0])

    assert traj.r[1] == pytest.approx([0, -1, 0])
