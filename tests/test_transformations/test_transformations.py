import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import resample, subsample


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


def test_resample_new_dt(x, traj):
    new_dt = 0.5
    new_traj = resample(traj, new_dt=new_dt)

    assert new_traj.dt == new_dt
    assert new_traj.r.x == pytest.approx(np.arange(0, 9, 0.5))


def test_resample_new_t(x, traj):
    new_t = [0, 1, 4, 7, 9]
    new_traj = resample(traj, new_t=new_t)

    assert new_traj.dt_std != 0
    assert new_traj.r.x == pytest.approx([0, 1, 4, 7, 9])


def test_threshold():
    dt = 1 / 3
    traj1 = Trajectory(x=np.arange(742), dt=dt)
    traj2 = Trajectory(x=np.arange(743), dt=dt)
    subsample(traj1)
    subsample(traj2)
