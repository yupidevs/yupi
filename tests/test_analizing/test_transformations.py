import pytest
import numpy as np
from yupi import Trajectory
from yupi.analyzing.transformations import subsample_trajectory


@pytest.fixture
def x():
    return list(range(10))

@pytest.fixture
def traj(x):
    return Trajectory(x=x)


def test_subsample_trajectory(x, traj):
    sub_sample = subsample_trajectory(traj, 2)

    assert len(sub_sample) == len(x) // 2
    assert sub_sample.r.x == pytest.approx(x[::2])