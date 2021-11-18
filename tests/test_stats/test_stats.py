import pytest
import numpy as np
from yupi import Trajectory
from yupi.stats import *

APPROX_REL_TOLERANCE = 1e-10

@pytest.fixture
def traj():
    points = [[0,0], [1,0], [1,1], [2,1]]
    return Trajectory(points=points)

@pytest.fixture
def traj1():
    x = [0,8,5,11]
    return Trajectory(x, dt=2)

@pytest.fixture
def traj2():
    x = [0, 8.5, 4.9, 10.5]
    return Trajectory(x, dt=2)

def test_turning_angles(traj):
    tae = turning_angles_ensemble([traj])
    assert tae == pytest.approx([np.pi/2, 3*np.pi/2], APPROX_REL_TOLERANCE)
   
    tae = turning_angles_ensemble([traj], degrees=True, wrap=False)
    assert tae == pytest.approx([90, -90], APPROX_REL_TOLERANCE)

def test_speed_ensemble(traj1):
    se = speed_ensemble([traj1, traj1])
    assert se == pytest.approx([4, 1.5, 3, 4, 1.5, 3], APPROX_REL_TOLERANCE)

def test_msd(traj1, traj2):
    msd_e = msd([traj1, traj2], time_avg=False)

    assert msd_e[0] == pytest.approx([0., 68.125, 24.505, 115.625])
    assert msd_e[1] == pytest.approx([0, 4.125, 0.495, 5.375])

    lag = 2
    msd_t = msd([traj1, traj2], time_avg=True, lag=lag)
    assert msd_t[0] == pytest.approx([37.595, 15.5025])
    assert msd_t[1] == pytest.approx([1.26166667, 1.4975])
    

def test_vacf():
    traj1 = Trajectory([0,8,5,11])
    traj2 = Trajectory([0, 8.5, 4.9, 10.5])

    vacf_e = vacf([traj1, traj2], time_avg=False)
    assert vacf_e[0] == pytest.approx([68.125, -27.3, 47.8])
    assert vacf_e[1] == pytest.approx([4.125, 3.3  , 0.2 ])

    vacf_t = vacf([traj1, traj2], time_avg=True, lag=2)
    assert vacf_t[0] == pytest.approx([-23.19, 47.8])
    assert vacf_t[1] == pytest.approx([2.19, 0.2])

def test_kurtosis():
    traj1 = Trajectory([0,8,5,11])
    traj2 = Trajectory([0, 8.5, 4.9, 10.5])

    kurt_e = kurtosis([traj1, traj2], time_avg=False)
    assert kurt_e[0] == pytest.approx([0, 1, 1, 1])

    lag = 2
    kurt_t = kurtosis([traj1, traj2], time_avg=True, lag=lag)
    assert kurt_t[0] == pytest.approx([0, 1.5])
    assert kurt_t[1] == pytest.approx([0, 0])

def test_psd():
    traj1 = Trajectory([0,8,5,11])
    lag = 2
    psd_o = psd([traj1], lag=lag, omega=True)
    assert psd_o[0] == pytest.approx([69, 27])
    assert psd_o[1] == pytest.approx([0, 0])
    assert psd_o[2] == pytest.approx( [-3.14159265,  0])

def test_checkers():
    points = [[0,0], [1,0], [1,1], [2,1]]
    simple_traj = Trajectory(points=points)
    non_equal_dt_traj = Trajectory(points=points, dt=2)
    non_equal_spacing_traj = Trajectory(points=points, t=[0, 0.1, 0.3, 0.35])
    non_equal_t0_traj = Trajectory(points=points, t0=1)
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
    # Collect r
    traj_r = collect([traj])
    assert np.allclose(traj_r, traj.r)

    # Collect v
    traj_v = collect([traj], velocity=True)
    assert np.allclose(traj_v, traj.v)

    # Collect with lag_step
    traj_r = collect([traj], lag_step=2)
    assert np.allclose(traj_r, traj.r[2:] - traj.r[:-2])

    # Collect with lag_step and velocity
    traj_v = collect([traj], velocity=True, lag_step=2)
    true_val = (traj.r[2:] - traj.r[:-2]) / (traj.dt * 2)
    assert np.allclose(traj_v, true_val)

    # Collect multiple trajectories
    traj_r = collect([traj, traj])
    assert np.allclose(traj_r, np.concatenate([traj.r, traj.r]))

    # Collect multiple trajectories with lag_step
    traj_r = collect([traj, traj], lag_step=2)
    true_r = traj.r[2:] - traj.r[:-2]
    assert np.allclose(traj_r, np.concatenate([true_r, true_r]))

    # Collect with lag_time
    traj1_r = collect([traj1], lag_time=2)
    step = int(2 / traj1.dt)
    true_r = traj1.r[step:] - traj1.r[:-step]
    assert np.allclose(traj1_r, true_r)

    # Collect with lag_time and velocity
    traj1_v = collect([traj1], velocity=True, lag_time=2)
    step = int(2 / traj1.dt)
    true_val = (traj1.r[step:] - traj1.r[:-step]) / (traj1.dt * 2)
    assert np.allclose(traj1_v, true_val)
