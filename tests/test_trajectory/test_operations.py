import numpy as np
from pytest import approx, fixture
import pytest
from yupi import Trajectory

APPROX_REL_TOLERANCE = 1e-12

@fixture
def points():
    return np.array([[1, 2], [4, 3], [4, 1]], dtype=float)


@fixture
def angles():
    return np.array([0, 1, 2], dtype=float)


@fixture
def traj(points, angles):
    return Trajectory(points=points, ang=angles)


def test_length(traj):
    assert len(traj) == 3


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
