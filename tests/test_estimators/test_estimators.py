import pytest
import numpy as np
from yupi import Trajectory
from yupi.estimators import turning_angles

def test_turning_angles():
    points = [[0,0], [1,0], [1,1], [2,1]]
    traj = Trajectory(points=points)

    ta = turning_angles(traj)
    assert ta == pytest.approx([np.pi/2, 3*np.pi/2])
   
    ta = turning_angles(traj, degrees=True, wrap=False)
    assert ta == pytest.approx([90, -90])
