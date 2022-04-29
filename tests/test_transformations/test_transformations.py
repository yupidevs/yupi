import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import subsample


@pytest.fixture
def x():
    return list(range(10))


@pytest.fixture
def traj(x):
    return Trajectory(x=x)


def test_subsample(x, traj):
    sub_sample = subsample(traj, 2)

    assert len(sub_sample) == len(x) // 2
    assert sub_sample.r.x == pytest.approx(x[::2])


def test_threshold():
    dt = 1 / 3
    traj1 = Trajectory(x=np.arange(742), dt=dt)
    traj2 = Trajectory(x=np.arange(743), dt=dt)
    subsample(traj1)
    subsample(traj2)
