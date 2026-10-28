import numpy as np
import pytest
from pytest import approx, fixture

from yupi import Trajectory
from yupi.trajectory import TrajectoryPoint
from yupi.units import Units

APPROX_REL_TOLERANCE = 1e-12


@fixture
def points() -> np.ndarray:
    return np.array([[1, 2], [4, 3], [4, 1], [6, 8], [5, 7]], dtype=float)


@fixture
def traj(points: np.ndarray) -> Trajectory:
    return Trajectory(points=points, extra={"foo": [1, 2, 3, 4, 5]}, bar=42)


@fixture
def time() -> np.ndarray:
    return np.array([0, 0.1, 0.18, 0.26, 0.41])


@fixture
def timed_traj(points: np.ndarray, time: np.ndarray) -> Trajectory:
    return Trajectory(points=points, t=time)


@fixture
def traj_with_extra(points: np.ndarray) -> Trajectory:
    return Trajectory(
        points=points,
        extra={"foo": np.array([10, 20, 30, 40, 50]), "bar": np.array([20, 40, 60, 80, 100])},
    )


def test_length(points: np.ndarray, traj: np.ndarray) -> None:
    assert len(traj) == len(points)


def test_copy(traj: Trajectory) -> None:
    copy_traj = traj.copy()

    assert traj.r == approx(copy_traj.r, APPROX_REL_TOLERANCE)
    assert traj.dt == approx(copy_traj.dt, APPROX_REL_TOLERANCE)
    assert traj.t == approx(copy_traj.t, APPROX_REL_TOLERANCE)
    assert traj.v == approx(copy_traj.v, APPROX_REL_TOLERANCE)
    assert traj.diff_est == copy_traj.diff_est
    assert traj.extra == copy_traj.extra
    assert traj.bar == copy_traj.bar


def test_iteration(points: np.ndarray, traj: Trajectory) -> None:
    time = traj.t

    for i, tp in enumerate(traj):
        point = points[i]
        t = time[i]

        assert point == approx(tp.r, APPROX_REL_TOLERANCE)  # Position
        assert t == approx(tp.t, APPROX_REL_TOLERANCE)  # Time


def test_constant_addition(points: np.ndarray, traj: Trajectory) -> None:
    new_traj = traj + 10
    new_points = points + 10

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_point_addition(points: np.ndarray, traj: Trajectory) -> None:
    new_traj = traj + (1, 3)
    new_points = points + (1, 3)

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_traj_addition(points: np.ndarray, traj: Trajectory) -> None:
    other_traj = traj.copy()
    new_traj = traj + other_traj
    new_points = points + points

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_addition(traj: Trajectory) -> None:
    with pytest.raises(TypeError):
        traj += "wrong"  # type: ignore[arg-type]


def test_constant_substraction(points: np.ndarray, traj: Trajectory) -> None:
    new_traj = traj - 10
    new_points = points - 10

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_point_substraction(points: np.ndarray, traj: Trajectory) -> None:
    new_traj = traj - (1, 3)
    new_points = points - (1, 3)

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_traj_substraction(points: np.ndarray, traj: Trajectory) -> None:
    other_traj = traj.copy()
    new_traj = traj - other_traj
    new_points = points - points

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_substraction(traj: Trajectory) -> None:
    with pytest.raises(TypeError):
        traj -= "wrong"  # type: ignore[arg-type]


def test_constant_multiplication(points: np.ndarray, traj: Trajectory) -> None:
    new_traj = traj * 3
    new_points = points * 3

    for true_point, point in zip(new_points, new_traj.r, strict=True):
        assert true_point == approx(point, APPROX_REL_TOLERANCE)


def test_wrong_multiplication(traj: Trajectory) -> None:
    with pytest.raises(TypeError):
        traj *= "wrong"  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        traj *= [1, 2]  # type: ignore[arg-type]


def test_slicing(traj: Trajectory, timed_traj: Trajectory, traj_with_extra: Trajectory) -> None:
    slice_1 = timed_traj[:]
    slice_2 = timed_traj[2:]
    slice_3 = timed_traj[:-2]
    slice_4 = timed_traj[1:4]
    slice_5 = traj[::2]
    slice_6 = traj[0:5:2]

    assert isinstance(slice_1, Trajectory)
    assert isinstance(slice_2, Trajectory)
    assert isinstance(slice_3, Trajectory)
    assert isinstance(slice_4, Trajectory)
    assert isinstance(slice_5, Trajectory)
    assert isinstance(slice_6, Trajectory)

    # Test lengths
    assert len(slice_1) == len(timed_traj)
    assert len(slice_2) == len(timed_traj) - 2
    assert len(slice_3) == len(timed_traj) - 2
    assert len(slice_4) == 3
    assert len(slice_5) == 3
    assert len(slice_6) == 3

    # Test points
    assert slice_1.r == approx(timed_traj.r[:], APPROX_REL_TOLERANCE)
    assert slice_2.r == approx(timed_traj.r[2:], APPROX_REL_TOLERANCE)
    assert slice_3.r == approx(timed_traj.r[:-2], APPROX_REL_TOLERANCE)
    assert slice_4.r == approx(timed_traj.r[1:4], APPROX_REL_TOLERANCE)
    assert slice_5.r == approx(traj.r[::2], APPROX_REL_TOLERANCE)
    assert slice_6.r == approx(traj.r[0:5:2], APPROX_REL_TOLERANCE)

    # Test time
    assert slice_1.t == approx(timed_traj.t[:], APPROX_REL_TOLERANCE)
    assert slice_2.t == approx(timed_traj.t[2:], APPROX_REL_TOLERANCE)
    assert slice_3.t == approx(timed_traj.t[:-2], APPROX_REL_TOLERANCE)
    assert slice_4.t == approx(timed_traj.t[1:4], APPROX_REL_TOLERANCE)
    assert slice_5.t == approx(traj.t[::2], APPROX_REL_TOLERANCE)
    assert slice_6.t == approx(traj.t[0:5:2], APPROX_REL_TOLERANCE)

    # Test dt
    assert slice_5.dt == approx(traj.dt * 2, APPROX_REL_TOLERANCE)
    assert slice_6.dt == approx(traj.dt * 2, APPROX_REL_TOLERANCE)

    # Test extra data
    assert traj_with_extra[:].extra["foo"] == approx(
        traj_with_extra.extra["foo"], APPROX_REL_TOLERANCE
    )

    assert traj_with_extra[2:].extra["foo"] == approx(
        traj_with_extra.extra["foo"][2:], APPROX_REL_TOLERANCE
    )

    assert traj_with_extra[:-2].extra["foo"] == approx(
        traj_with_extra.extra["foo"][:-2], APPROX_REL_TOLERANCE
    )

    assert traj_with_extra[1:4].extra["bar"] == approx(
        traj_with_extra.extra["bar"][1:4], APPROX_REL_TOLERANCE
    )

    assert traj_with_extra[::2].extra["bar"] == approx(
        traj_with_extra.extra["bar"][::2], APPROX_REL_TOLERANCE
    )


def test_indexing(timed_traj: Trajectory, traj_with_extra: Trajectory) -> None:
    index_1 = timed_traj[0]
    index_2 = timed_traj[2]
    index_3 = timed_traj[-2]
    index_4 = traj_with_extra[0]
    index_5 = traj_with_extra[2]
    index_6 = traj_with_extra[-2]

    assert isinstance(index_1, TrajectoryPoint)
    assert isinstance(index_2, TrajectoryPoint)
    assert isinstance(index_3, TrajectoryPoint)
    assert isinstance(index_4, TrajectoryPoint)
    assert isinstance(index_5, TrajectoryPoint)
    assert isinstance(index_6, TrajectoryPoint)

    # Test points
    assert index_1.r == approx(timed_traj.r[0], APPROX_REL_TOLERANCE)
    assert index_2.r == approx(timed_traj.r[2], APPROX_REL_TOLERANCE)
    assert index_3.r == approx(timed_traj.r[-2], APPROX_REL_TOLERANCE)

    assert index_1.t == approx(timed_traj.t[0], APPROX_REL_TOLERANCE)
    assert index_2.t == approx(timed_traj.t[2], APPROX_REL_TOLERANCE)
    assert index_3.t == approx(timed_traj.t[-2], APPROX_REL_TOLERANCE)

    assert index_1.v == approx(timed_traj.v[0], APPROX_REL_TOLERANCE)
    assert index_2.v == approx(timed_traj.v[2], APPROX_REL_TOLERANCE)
    assert index_3.v == approx(timed_traj.v[-2], APPROX_REL_TOLERANCE)

    assert index_4.foo == approx(traj_with_extra.foo[0], APPROX_REL_TOLERANCE)

    assert index_5.foo == approx(traj_with_extra.foo[2], APPROX_REL_TOLERANCE)

    assert index_6.extra["foo"] == approx(traj_with_extra.extra["foo"][-2], APPROX_REL_TOLERANCE)

    with pytest.raises(AttributeError):
        _ = index_4.non_existent

    with pytest.raises(TypeError):
        _ = timed_traj["foo"]  # type: ignore[index]


def test_units_conversion() -> None:
    # m/s by default
    traj = Trajectory(x=[1000, 2000, 3000, 4000, 5000], t=[3600, 7200, 10800, 14400, 18000])

    _, _ = traj.v, traj.a  # to cache calculate velocity and acceleration

    new_traj = traj.to("km/h")

    assert new_traj.r.x == approx([1, 2, 3, 4, 5])
    assert new_traj.t == approx([1, 2, 3, 4, 5])
    assert new_traj.v == approx([1, 1, 1, 1, 1])

    traj.to(Units("km", "h"), inplace=True)

    assert traj.r.x == approx([1, 2, 3, 4, 5])
    assert traj.t == approx([1, 2, 3, 4, 5])
    assert traj.v == approx([1, 1, 1, 1, 1])


def test_creation_by_xyz() -> None:
    Trajectory(x=[1, 2, 3])
    Trajectory(x=[1, 2, 4], y=[2, 3, 6])
    Trajectory(x=[1, 2, 4], y=[2, 3, 6], z=[1, 4, 7])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2], z=[2, 5])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], z=[2, 5])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2], y=[2, 6], z=[5])

    with pytest.raises(ValueError):
        Trajectory(x=[2], y=[2, 6], z=[5, 8])


def test_creation_by_axes() -> None:
    Trajectory(axes=[[1, 2, 3]])
    Trajectory(axes=[[1, 2, 5], [2, 3, 3]])
    Trajectory(axes=[[1, 2, 1], [2, 3, 9], [1, 4, 3]])
    Trajectory(axes=[[1, 2, 5], [2, 3, 3], [1, 4, 8], [7, 8, 7]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[1, 2], [2, 6], [5]])

    with pytest.raises(ValueError):
        Trajectory(axes=[[2], [2, 6], [5, 8]])


def test_creation_by_points() -> None:
    Trajectory(points=[[1, 2], [2, 3], [6, 7]])
    Trajectory(points=[[1, 2, 4], [2, 3, 2], [1, 4, 8], [2, 6, 8]])
    Trajectory(points=[[1, 2, 7, 3], [2, 3, 5, 3], [3, 7, 2, 1]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2], [2], [2, 5]])

    with pytest.raises(ValueError):
        Trajectory(points=[[1, 2]])


def test_creation_with_time() -> None:
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2])
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], dt=0.1)
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2], dt=0.1)
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0.4, 0.5, 0.6])
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0.4, 0.5, 0.6], dt=0.1, t_0=0.4)

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 1])

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 0.1, 0.2], dt=0.2)


def test_creation_general() -> None:
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 1, 2], traj_id="test")
    Trajectory(x=[1, 2, 3], y=[2, 3, 6], dt=0.5, traj_id="test")
    Trajectory(points=[[1, 2], [2, 3], [3, 6]], dt=0.5, traj_id="test")
    Trajectory(axes=[[1, 2, 4], [2, 3, 6]], dt=0.5, t=[1, 1.5, 2], t_0=1, traj_id="test")

    # No positional data
    with pytest.raises(ValueError):
        Trajectory(t=[0, 1], dt=0.5)

    # At least 2 points
    with pytest.raises(ValueError):
        Trajectory(x=[1], y=[2], t=[0])

    # t_0 not matching t[0]
    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 1, 2], t_0=0.5)

    # dt but std not 0
    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], y=[2, 3, 6], t=[0, 4, 6], dt=3)


def test_creation_with_extra_data() -> None:
    traj = Trajectory(x=[1, 2, 3], t=[0, 1, 2], extra={"foo": [False, True, True]})

    assert traj.foo == [False, True, True]

    with pytest.raises(ValueError):
        Trajectory(x=[1, 2, 3], t=[0, 1, 2], extra={"foo": [False, True]})


def test_creation_with_metadata() -> None:
    traj = Trajectory(x=[1, 2, 3], t=[0, 1, 2], id="test", label="foo")

    assert traj.id == "test"
    assert traj.label == "foo"

    with pytest.raises(AttributeError):
        _ = traj.not_a_valid_metadata


def test_bounds() -> None:
    traj = Trajectory(x=[1, 2, 3], y=[1, 1, 2])

    assert traj.bounds == [pytest.approx((1, 3)), pytest.approx((1, 2))]


def test_properties() -> None:
    traj = Trajectory(x=[1, 2, 3], y=[1, 1, 2], t=[1, 2, 3])

    assert traj.dim == 2
    assert traj.dt == 1.0
    assert traj.t_0 == 1.0

    assert traj.delta_r == pytest.approx(traj.r.delta)
    assert traj.delta_v == pytest.approx(traj.v.delta)


def test_lazy() -> None:
    traj = Trajectory(x=[1, 2, 3], y=[1, 1, 2], t=[1, 2, 3], lazy=True)

    v1 = traj.v
    a1 = traj.a

    assert traj._Trajectory__v is not None
    assert traj._Trajectory__a is not None

    v2 = traj.v
    a2 = traj.a

    assert traj._Trajectory__v is v1
    assert traj._Trajectory__v is v2
    assert traj._Trajectory__a is a1
    assert traj._Trajectory__a is a2

    traj = Trajectory(x=[1, 2, 3], y=[1, 1, 2], t=[1, 2, 3], lazy=False)

    v1 = traj.v
    a1 = traj.a

    v2 = traj.v
    a2 = traj.a

    assert traj._Trajectory__v is not v1
    assert traj._Trajectory__v is v2

    assert traj._Trajectory__a is not a1
    assert traj._Trajectory__a is a2
