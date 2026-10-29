import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import exp_convolutional_filter, exp_moving_average_filter


@pytest.fixture
def non_zero_origin() -> list[int]:
    return [7, 7, 7]  # Initial position


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
        trajectory[t] = trajectory[t - 1] + velocity * t + np.random.normal(0, noise_std, size=3)
    return Trajectory(points=trajectory, t=t_vals)


def test_exp_convolution_origin(
    constant_v_non_zero_origin_traj: Trajectory, non_zero_origin: list[int]
) -> None:
    smooted_trajectory = exp_convolutional_filter(constant_v_non_zero_origin_traj, 1 / 100)
    assert smooted_trajectory.r[0] == pytest.approx(non_zero_origin)


def test_ema_origin(
    constant_v_non_zero_origin_traj: Trajectory, non_zero_origin: list[int]
) -> None:
    smooted_trajectory = exp_moving_average_filter(constant_v_non_zero_origin_traj, alpha=1 / 100)
    assert smooted_trajectory.r[0] == pytest.approx(non_zero_origin)
