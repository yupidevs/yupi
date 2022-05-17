import numpy as np
import pytest

from yupi import Trajectory, DiffMethod, WindowType
from yupi.stats import *

APPROX_REL_TOLERANCE = 1e-10


@pytest.fixture
def traj():
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    return Trajectory(
        points=points,
        diff_est={
            "method": DiffMethod.LINEAR_DIFF,
            "window_type": WindowType.FORWARD,
        },
    )


@pytest.fixture
def traj1():
    x = [0, 8, 5, 11]
    return Trajectory(
        x=x,
        dt=2,
        diff_est={
            "method": DiffMethod.LINEAR_DIFF,
            "window_type": WindowType.FORWARD,
        },
    )


@pytest.fixture
def traj2():
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(
        x=x,
        dt=2,
        diff_est={
            "method": DiffMethod.LINEAR_DIFF,
            "window_type": WindowType.FORWARD,
        },
    )


def test_turning_angles(traj):
    tae = turning_angles_ensemble([traj])
    assert tae == pytest.approx([np.pi / 2, 3 * np.pi / 2], APPROX_REL_TOLERANCE)

    tae = turning_angles_ensemble([traj], degrees=True, wrap=False)
    assert tae == pytest.approx([90, -90], APPROX_REL_TOLERANCE)


def test_speed_ensemble(traj1):
    print(traj1.v)
    se = speed_ensemble([traj1, traj1])
    assert se == pytest.approx([4, 1.5, 3, 3, 4, 1.5, 3, 3], APPROX_REL_TOLERANCE)


def test_msd(traj1, traj2):
    msd_e = msd([traj1, traj2], time_avg=False)

    assert msd_e[0] == pytest.approx([0.0, 68.125, 24.505, 115.625])
    assert msd_e[1] == pytest.approx([0, 4.125, 0.495, 5.375])

    lag = 2
    msd_t = msd([traj1, traj2], time_avg=True, lag=lag)
    assert msd_t[0] == pytest.approx([37.595, 15.5025])
    assert msd_t[1] == pytest.approx([1.26166667, 1.4975])


def test_vacf(traj1, traj2):
    vacf_e = vacf([traj1, traj2], time_avg=False)
    assert vacf_e[0] == pytest.approx([17.03125, -6.825, 11.95, 11.95])
    assert vacf_e[1] == pytest.approx([1.03125, 0.825, 0.05, 0.05])

    vacf_t = vacf([traj1, traj2], time_avg=True, lag=2)
    assert vacf_t[0] == pytest.approx([-1.05833333, 3.59])
    assert vacf_t[1] == pytest.approx([0.55833333, 0.16])


def test_kurtosis(traj1, traj2):
    kurt_e = kurtosis([traj1, traj2], time_avg=False)
    assert kurt_e[0] == pytest.approx([0, 1, 1, 1])

    lag = 2
    kurt_t = kurtosis([traj1, traj2], time_avg=True, lag=lag)
    assert kurt_t[0] == pytest.approx([0, 1.5])
    assert kurt_t[1] == pytest.approx([0, 0])


def test_psd(traj1):
    lag = 2
    psd_o = psd([traj1], lag=lag, omega=True)
    assert psd_o[0] == pytest.approx([8.5, 6.5])
    assert psd_o[1] == pytest.approx([0, 0])
    assert psd_o[2] == pytest.approx([-9.8696044, 0.0])


def test_checkers():
    points = [[0, 0], [1, 0], [1, 1], [2, 1]]
    simple_traj = Trajectory(points=points)
    non_equal_dt_traj = Trajectory(points=points, dt=2)
    non_equal_spacing_traj = Trajectory(points=points, t=[0, 0.1, 0.3, 0.35])
    non_equal_t0_traj = Trajectory(points=points, t_0=1)
    non_equal_dim_traj = Trajectory(points=[p + [0] for p in points])

    # Exact dimension checker
    with pytest.raises(ValueError):
        turning_angles_ensemble([non_equal_dim_traj])

    # Uniform time spaced checker
    with pytest.raises(ValueError):
        turning_angles_ensemble([simple_traj, non_equal_spacing_traj])

    # Same dt checker
    with pytest.raises(ValueError):
        turning_angles_ensemble([simple_traj, non_equal_dt_traj])

    # Same dim checker
    with pytest.raises(ValueError):
        speed_ensemble([simple_traj, non_equal_dim_traj])

    # Same t checker
    with pytest.raises(ValueError):
        msd([simple_traj, non_equal_t0_traj], time_avg=False)


def test_collect(traj, traj1):
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

    # Collect with lag and at parameters at the same time
    with pytest.raises(ValueError):
        collect([traj1], lag=2.0, at=1.5)
