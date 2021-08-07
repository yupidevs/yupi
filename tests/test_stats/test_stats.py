import pytest
import numpy as np
from yupi import Trajectory
from yupi.stats import *


@pytest.fixture
def traj():
    points = [[0,0], [1,0], [1,1], [2,1]]
    return Trajectory(points=points)

def test_turning_angles(traj):
    tae = turning_angles_ensemble([traj])
    assert tae == pytest.approx([np.pi/2, 3*np.pi/2])
   
    tae = turning_angles_ensemble([traj], degrees=True, wrap=False)
    assert tae == pytest.approx([90, -90])

def test_speed_ensemble():
    pass

def test_msd():
    pass

def test_vacf():
    pass

def test_kurtosis():
    pass

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
