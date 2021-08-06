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
