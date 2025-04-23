import numpy as np
import pytest

from yupi import Trajectory
from yupi.generators import DiffDiffGenerator, LangevinGenerator, RandomWalkGenerator


def test_random_walk() -> None:
    trajs = RandomWalkGenerator(T=5, dt=0.5, dim=2, N=3).generate()

    assert len(trajs) == 3
    assert all(isinstance(traj, Trajectory) for traj in trajs)
    assert all(len(traj) == 10 for traj in trajs)
    assert all(traj.dt == pytest.approx(0.5) for traj in trajs)
    assert all(traj.t[0] == pytest.approx(0) for traj in trajs)
    assert all(traj.t[-1] == pytest.approx(4.5) for traj in trajs)
    assert all(traj.dim == 2 for traj in trajs)


def test_random_walk_wrong_() -> None:
    with pytest.raises(ValueError):
        RandomWalkGenerator(T=5, dim=3, N=3, actions_prob=[[1], [1]])

    with pytest.raises(ValueError):
        RandomWalkGenerator(T=5, dim=2, N=3, actions_prob=[[1 / 3, 1 / 3], [1]])


def test_langevin() -> None:
    trajs = LangevinGenerator(T=5, dt=0.5, dim=2, N=3).generate()

    assert len(trajs) == 3
    assert all(isinstance(traj, Trajectory) for traj in trajs)
    assert all(len(traj) == 10 for traj in trajs)
    assert all(traj.dt == pytest.approx(0.5) for traj in trajs)
    assert all(traj.t[0] == pytest.approx(0) for traj in trajs)
    assert all(traj.t[-1] == pytest.approx(4.5) for traj in trajs)
    assert all(traj.dim == 2 for traj in trajs)


def test_langevin_initial_conditions() -> None:
    trajs = LangevinGenerator(T=5, dt=0.5, dim=2, N=3, r0=0.1).generate()
    assert all(traj.r[0] == pytest.approx([0.1, 0.1]) for traj in trajs)

    trajs = LangevinGenerator(T=5, dt=0.5, dim=2, N=3, r0=[0.1, 3.4]).generate()

    assert all(traj.r[0] == pytest.approx([0.1, 3.4]) for traj in trajs)

    trajs = LangevinGenerator(
        T=5, dt=0.5, dim=2, N=2, r0=[[0.1, 3.4], [4.0, 5.0]]
    ).generate()
    assert trajs[0].r[0] == pytest.approx([0.1, 4.0])
    assert trajs[1].r[0] == pytest.approx([3.4, 5.0])

    trajs = LangevinGenerator(T=5, dt=0.5, dim=2, N=3, v0=0.1).generate()
    assert all(traj.v[0] == pytest.approx([0.1, 0.1]) for traj in trajs)

    trajs = LangevinGenerator(T=5, dt=0.5, dim=2, N=3, v0=[0.1, 3.4]).generate()
    assert all(traj.v[0] == pytest.approx([0.1, 3.4]) for traj in trajs)

    trajs = LangevinGenerator(
        T=5, dt=0.5, dim=2, N=2, v0=[[0.1, 3.4], [4.0, 5.0]]
    ).generate()
    assert trajs[0].v[0] == pytest.approx([0.1, 4.0])
    assert trajs[1].v[0] == pytest.approx([3.4, 5.0])

    with pytest.raises(ValueError):
        LangevinGenerator(T=5, dim=3, N=3, r0=[[1], [1]]).generate()

    with pytest.raises(ValueError):
        LangevinGenerator(T=5, dim=3, N=3, v0=[[1], [1]]).generate()


def test_langevin_bounds() -> None:
    traj = LangevinGenerator(
        T=5, dt=0.5, dim=2, N=1, bounds=np.array([[-1, -1], [1, 1]])
    ).generate()[0]

    np.all(traj.r >= -1)
    np.all(traj.r <= 1)

    with pytest.raises(ValueError):
        LangevinGenerator(T=5, dim=3, N=3, bounds=np.array([[1, 1], [2, 2]])).generate()

    with pytest.raises(ValueError):
        LangevinGenerator(
            T=5, dim=3, N=3, bounds=np.array([[-2, -2], [-1, -1]])
        ).generate()


def test_diffdiff() -> None:
    trajs = DiffDiffGenerator(T=5, dt=0.5, dim=2, N=3).generate()

    assert len(trajs) == 3
    assert all(isinstance(traj, Trajectory) for traj in trajs)
    assert all(len(traj) == 10 for traj in trajs)
    assert all(traj.dt == pytest.approx(0.5) for traj in trajs)
    assert all(traj.t[0] == pytest.approx(0) for traj in trajs)
    assert all(traj.t[-1] == pytest.approx(4.5) for traj in trajs)
    assert all(traj.dim == 2 for traj in trajs)


def test_diffdiff_initial_conditions() -> None:
    trajs = DiffDiffGenerator(T=5, dt=0.5, dim=2, N=3, r0=0.1).generate()
    assert all(traj.r[0] == pytest.approx([0.1, 0.1]) for traj in trajs)

    trajs = DiffDiffGenerator(T=5, dt=0.5, dim=2, N=3, r0=[0.1, 3.4]).generate()
    assert all(traj.r[0] == pytest.approx([0.1, 3.4]) for traj in trajs)

    trajs = DiffDiffGenerator(
        T=5, dt=0.5, dim=2, N=2, r0=[[0.1, 3.4], [4.0, 5.0]]
    ).generate()
    assert trajs[0].r[0] == pytest.approx([0.1, 4.0])
    assert trajs[1].r[0] == pytest.approx([3.4, 5.0])

    with pytest.raises(ValueError):
        DiffDiffGenerator(T=5, dim=3, N=3, r0=[[1], [1]]).generate()


def test_diffdiff_bounds() -> None:
    traj = DiffDiffGenerator(
        T=5, dt=0.5, dim=2, N=1, bounds=np.array([[-1, -1], [1, 1]])
    ).generate()[0]

    np.all(traj.r >= -1)
    np.all(traj.r <= 1)

    with pytest.raises(ValueError):
        DiffDiffGenerator(T=5, dim=3, N=3, bounds=np.array([[1, 1], [2, 2]])).generate()

    with pytest.raises(ValueError):
        DiffDiffGenerator(
            T=5, dim=3, N=3, bounds=np.array([[-2, -2], [-1, -1]])
        ).generate()
