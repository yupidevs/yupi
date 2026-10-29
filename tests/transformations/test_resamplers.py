import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import resample, subsample


@pytest.fixture
def x() -> list[int]:
    return list(range(10))


@pytest.fixture
def traj(x: list[int]) -> Trajectory:
    return Trajectory(x=x)


def test_subsample(x: list[int], traj: Trajectory) -> None:
    sub_sample = subsample(traj, 2)

    assert len(sub_sample) == len(x) // 2


def test_resample_new_dt(x: list[int], traj: Trajectory) -> None:
    new_dt = 0.5
    new_traj = resample(traj, new_dt=new_dt)

    assert new_traj.dt == new_dt
    assert new_traj.r.x == pytest.approx(np.arange(0, 9, 0.5))
    assert new_traj.r[0] == pytest.approx(traj.r[0])


def test_resample_new_t(x: list[int], traj: Trajectory) -> None:
    new_t = [0, 1, 4, 7, 9]
    new_traj = resample(traj, new_t=new_t)

    assert new_traj.dt_std != 0
    assert new_traj.r.x == pytest.approx([0, 1, 4, 7, 9])


def test_threshold() -> None:
    dt = 1 / 3
    traj1 = Trajectory(x=np.arange(742), dt=dt)
    traj2 = Trajectory(x=np.arange(743), dt=dt)
    subsample(traj1)
    subsample(traj2)
