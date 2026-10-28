import numpy as np
import pytest

from yupi import Trajectory
from yupi.transformations import rotate_2d, rotate_3d


@pytest.fixture
def simple_traj() -> Trajectory:
    return Trajectory(x=[0, 1], y=[0, 1])


def test_rotation(simple_traj: Trajectory) -> None:
    # 45 degrees
    ang = np.pi / 4

    # [0, 0] -> [0,       0]
    # [1, 1] -> [0, sqrt(2)]
    rotate_2d(simple_traj, ang)

    assert simple_traj.r[0] == pytest.approx([0, 0])
    assert simple_traj.r[1] == pytest.approx([0, np.sqrt(2)])


def test_rotation_3d() -> None:
    traj = Trajectory(x=[0, 1], y=[0, 0], z=[0, 0])

    rotate_3d(traj, -np.pi / 2, [0, 0, 3])

    assert traj.r[0] == pytest.approx([0, 0, 0])
    assert traj.r[1] == pytest.approx([0, 1, 0])

    rotate_3d(traj, np.pi, [1, 0, 0])

    assert traj.r[1] == pytest.approx([0, -1, 0])
