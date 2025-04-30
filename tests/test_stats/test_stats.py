import numpy as np
import pytest

from yupi import NotUniformTimeSpacedError, Trajectory
from yupi._checkers import (
    DifferentDimensionError,
    DifferentDtError,
    DifferentLengthError,
    DifferentTimeVectorError,
)
from yupi.stats._stats import (
    collect,
)
from yupi.stats.msd import MsdStat
from yupi.stats.speed import SpeedStat
from yupi.stats.turning_angles import TurningAngleStat

APPROX_REL_TOLERANCE = 1e-10


@pytest.fixture
def traj() -> Trajectory:
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    return Trajectory(points=points)


@pytest.fixture
def traj1() -> Trajectory:
    x = [0, 8, 5, 11]
    return Trajectory(x=x, dt=2)


@pytest.fixture
def traj2() -> Trajectory:
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(x=x, dt=2)


def test_checkers() -> None:
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    simple_traj = Trajectory(points=points)
    non_equal_dt_traj = Trajectory(points=points, dt=2)
    non_equal_spacing_traj = Trajectory(points=points, t=[0, 0.1, 0.3, 0.35])
    non_equal_t0_traj = Trajectory(points=points, t_0=1)
    non_equal_dim_traj = Trajectory(points=[[*p, 0] for p in points])

    # Exact dimension checker
    with pytest.raises(DifferentDimensionError):
        TurningAngleStat([non_equal_dim_traj])

    # Uniform time spaced checker
    with pytest.raises(NotUniformTimeSpacedError):
        TurningAngleStat([non_equal_spacing_traj])

    # Same dt checker
    with pytest.raises(DifferentDtError):
        TurningAngleStat([simple_traj, non_equal_dt_traj])

    # Same dim checker
    with pytest.raises(DifferentDimensionError):
        SpeedStat([simple_traj, non_equal_dim_traj])

    # Same length checker
    with pytest.raises(DifferentLengthError):
        MsdStat([simple_traj, Trajectory(points=points[:-2])])

    # Same t checker
    with pytest.raises(DifferentTimeVectorError):
        MsdStat([simple_traj, non_equal_t0_traj])


def test_collect(traj: Trajectory, traj1: Trajectory) -> None:
    # Collect position
    traj_r = collect([traj])
    assert np.allclose(traj_r, traj.r)

    # Collect velocity
    traj_v = collect([traj], velocity=True)
    assert np.allclose(traj_v, traj.v)

    # Collect with lag as step
    traj_r = collect([traj], lag=2)
    assert np.allclose(traj_r, traj.r[2:] - traj.r[:-2])

    # Collect with lag as step and velocity
    traj_v = collect([traj], velocity=True, lag=2)
    true_val = (traj.r[2:] - traj.r[:-2]) / (traj.dt * 2)
    assert np.allclose(traj_v, true_val)

    # Collect multiple trajectories
    traj_r = collect([traj, traj])
    assert np.allclose(traj_r, np.concatenate([traj.r, traj.r]))

    # Collect multiple trajectories with lag as step
    traj_r = collect([traj, traj], lag=2)
    true_r = traj.r[2:] - traj.r[:-2]
    assert np.allclose(traj_r, np.concatenate([true_r, true_r]))

    # Collect with lag as time
    traj1_r = collect([traj1], lag=2.0)
    step = int(2 / traj1.dt)
    true_r = traj1.r[step:] - traj1.r[:-step]
    assert np.allclose(traj1_r, true_r)

    # Collect with lag as time and velocity
    traj1_v = collect([traj1], velocity=True, lag=2.0)
    step = int(2 / traj1.dt)
    true_val = (traj1.r[step:] - traj1.r[:-step]) / (traj1.dt)
    assert np.allclose(traj1_v, true_val)

    # Collect at parameter
    traj1_r = collect([traj1], at=1)
    true_r = traj1.r[traj1.t == 1]
    assert np.allclose(traj1_r, true_r)

    # Collect with at and func
    traj1_r = collect([traj1], at=0, func=lambda vec: vec + 1)
    true_r = traj1.r[traj1.t == 0] + 1
    pytest.approx(traj1_r, true_r)

    traj1_r = collect([traj1], lag=2.0, func=lambda vec: vec + 1)
    pytest.approx(traj1_r, traj1.r[1:] - traj1.r[:-1] + 1)

    # Collect with lag and at parameters at the same time
    with pytest.raises(ValueError):
        collect([traj1], lag=2.0, at=1.5)

    # Should log a warning
    collect([traj1], at=100)
