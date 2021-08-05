import pytest
import numpy as np
from yupi import Trajectory
from yupi.analyzing import *


@pytest.fixture
def traj():
    points = [[0,0], [1,0], [1,1], [2,1]]
    return Trajectory(points=points)

@pytest.fixture
def non_uniform_traj():
    points = [[0,0], [1,0], [1,1], [2,1]]
    t = [0, 0.1, 0.3, 0.35]
    return Trajectory(points=points, t=t)

def test_turning_angles(traj, non_uniform_traj):
    
    ta = estimate_turning_angles([traj])
    assert ta == pytest.approx([np.pi/2, 3*np.pi/2])
    
    ta = estimate_turning_angles([traj], degrees=True, wrap=False)
    assert ta == pytest.approx([90, -90])