import pytest

from yupi import Trajectory

APPROX_REL_TOLERANCE = 1e-10


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
    Trajectory(
        axes=[[1, 2, 4], [2, 3, 6]], dt=0.5, t=[1, 1.5, 2], t_0=1, traj_id="test"
    )

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

    assert traj.bounds == pytest.approx([(1, 3), (1, 2)])


def test_properties() -> None:
    traj = Trajectory(x=[1, 2, 3], y=[1, 1, 2], t=[1, 2, 3])

    assert traj.dim == 2
    assert traj.dt == 1.0
    assert traj.t_0 == 0.0

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
