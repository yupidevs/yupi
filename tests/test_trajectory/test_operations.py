import numpy as np
from pytest import approx, fixture
import pytest
from yupi import Trajectory

APPROX_REL_TOLERANCE = 1e-12

@fixture
def points():
    return np.array([[1, 2], [4, 3], [4, 1], [6, 8], [5, 7]], dtype=float)


@fixture
def angles():
    return np.array([0, 1, 2, 1, 1.5], dtype=float)


@fixture
def traj(points, angles):
    return Trajectory(points=points, ang=angles)


@fixture
def time():
    return [0, 0.1, 0.18, 0.26, 0.41]


@fixture
def timed_traj(points, angles, time):
    return Trajectory(points=points, ang=angles, t=time)


@fixture
def simple_traj():
    return Trajectory(x=[0,1], y=[0,1])


def test_length(points, traj):
    assert len(traj) == len(points)


def test_copy(traj):
    copy_traj = traj.copy()

    assert traj.r == approx(copy_traj.r, APPROX_REL_TOLERANCE)
    assert traj.dt == approx(copy_traj.dt, APPROX_REL_TOLERANCE)
    assert traj.t == approx(copy_traj.t, APPROX_REL_TOLERANCE)
    assert traj.v == approx(copy_traj.v, APPROX_REL_TOLERANCE)
    assert traj.ang == approx(copy_traj.ang, APPROX_REL_TOLERANCE)


def test_iteration(points, angles, traj):
    time = traj.t

    for i, tp in enumerate(traj):
        point = points[i]
        ang = angles[i]
        t = time[i]

        assert point == approx(tp.r, APPROX_REL_TOLERANCE)  # Position
        assert t == approx(tp.t, APPROX_REL_TOLERANCE)      # Time
        assert ang == approx(tp.ang, APPROX_REL_TOLERANCE)  # Angle

def test_rotation(simple_traj):
    # 45 degrees
    ang = np.pi / 4

    # [0, 0] -> [0,       0]
    # [1, 1] -> [0, sqrt(2)]
    simple_traj.rotate2d(ang)

    assert simple_traj.r[0] == approx([0,0], APPROX_REL_TOLERANCE)
    assert simple_traj.r[1] == approx([0,np.sqrt(2)], APPROX_REL_TOLERANCE)


def test_rotation_3d():
    traj = Trajectory(x=[0,1], y=[0,0], z=[0,0])

    traj.rotate3d(-np.pi / 2, [0, 0, 3])

    assert traj.r[0] == approx([0, 0, 0], APPROX_REL_TOLERANCE)
    assert traj.r[1] == approx([0, 1, 0], APPROX_REL_TOLERANCE)

    traj.rotate3d(np.pi, [1, 0, 0])

    assert traj.r[1] == approx([0, -1, 0], APPROX_REL_TOLERANCE)


def test_constant_addition(points, traj):
    new_traj = traj + 10
    new_points = points + 10

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_point_addition(points, traj):
    new_traj = traj + (1, 3)
    new_points = points + (1, 3)

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_traj_addition(points, traj):
    other_traj = traj.copy()
    new_traj = traj + other_traj
    new_points = points + points

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_addition(traj):
    with pytest.raises(TypeError):
        traj += 'wrong'


def test_constant_substraction(points, traj):
    new_traj = traj - 10
    new_points = points - 10

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_point_substraction(points, traj):
    new_traj = traj - (1, 3)
    new_points = points - (1, 3)

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_traj_substraction(points, traj):
    other_traj = traj.copy()
    new_traj = traj - other_traj
    new_points = points - points

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_substraction(traj):
    with pytest.raises(TypeError):
        traj -= 'wrong'


def test_constant_multiplication(points, traj):
    new_traj = traj * 3
    new_points = points * 3

    for true_point, point in zip(new_points, new_traj.r):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_multiplication(traj):
    with pytest.raises(TypeError):
        traj *= 'wrong'
    with pytest.raises(TypeError):
        traj *= [1, 2]


def test_slicing(traj, timed_traj):
    slice_1 = timed_traj[:]
    slice_2 = timed_traj[2:]
    slice_3 = timed_traj[:-2]
    slice_4 = timed_traj[1:4]
    slice_5 = traj[::2]
    slice_6 = traj[1:4:2]

    # Test lengths
    assert len(slice_1) == len(timed_traj)
    assert len(slice_2) == len(timed_traj) - 2
    assert len(slice_3) == len(timed_traj) - 2
    assert len(slice_4) == 3
    assert len(slice_5) == 3
    assert len(slice_6) == 2

    # Test points
    assert slice_1.r == approx(timed_traj.r[:], APPROX_REL_TOLERANCE)
    assert slice_2.r == approx(timed_traj.r[2:], APPROX_REL_TOLERANCE)
    assert slice_3.r == approx(timed_traj.r[:-2], APPROX_REL_TOLERANCE)
    assert slice_4.r == approx(timed_traj.r[1:4], APPROX_REL_TOLERANCE)
    assert slice_5.r == approx(traj.r[::2], APPROX_REL_TOLERANCE)
    assert slice_6.r == approx(traj.r[1:4:2], APPROX_REL_TOLERANCE)

    # Test time
    assert slice_1.t == approx(timed_traj.t[:], APPROX_REL_TOLERANCE)
    assert slice_2.t == approx(timed_traj.t[2:], APPROX_REL_TOLERANCE)
    assert slice_3.t == approx(timed_traj.t[:-2], APPROX_REL_TOLERANCE)
    assert slice_4.t == approx(timed_traj.t[1:4], APPROX_REL_TOLERANCE)
    assert slice_5.t == approx(traj.t[::2], APPROX_REL_TOLERANCE)
    assert slice_6.t == approx(traj.t[1:4:2], APPROX_REL_TOLERANCE)

    # Test angle
    assert slice_1.ang == approx(timed_traj.ang[:], APPROX_REL_TOLERANCE)
    assert slice_2.ang == approx(timed_traj.ang[2:], APPROX_REL_TOLERANCE)
    assert slice_3.ang == approx(timed_traj.ang[:-2], APPROX_REL_TOLERANCE)
    assert slice_4.ang == approx(timed_traj.ang[1:4], APPROX_REL_TOLERANCE)
    assert slice_5.ang == approx(traj.ang[::2], APPROX_REL_TOLERANCE)
    assert slice_6.ang == approx(traj.ang[1:4:2], APPROX_REL_TOLERANCE)

    # Test dt
    assert slice_5.dt == approx(traj.dt * 2, APPROX_REL_TOLERANCE)
    assert slice_6.dt == approx(traj.dt * 2, APPROX_REL_TOLERANCE)

    
