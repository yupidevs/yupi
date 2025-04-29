import pytest

from yupi import DiffMethod, Trajectory, WindowType


def test_linear_forward_diff() -> None:
    diff = {
        "method": DiffMethod.LINEAR_DIFF,
        "window_type": WindowType.FORWARD,
        "accuracy": 1,
    }
    traj = Trajectory(x=[0, 2, 6], diff_est=diff)

    assert traj.v.x == pytest.approx([2.0, 4.0, 4.0])
    assert traj.a.x == pytest.approx([2.0, 0.0, 0.0])


def test_linear_central_diff() -> None:
    diff = {
        "method": DiffMethod.LINEAR_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 1,
    }
    traj = Trajectory(x=[0, 2, 6, 10], diff_est=diff)

    # v.x[0] = v.x[1] = 3
    # v.x[1] = (x[2] - x[0]) / (t[2] - t[0])
    #        = (6 - 0) / (2 - 0)
    #        = 3
    # v.x[2] = (x[3] - x[1]) / (t[3] - t[1])
    #        = (10 - 2) / (4 - 2)
    #        = 4
    # v.x[3] = v.x[2] = 4
    assert traj.v.x == pytest.approx([3.0, 3.0, 4.0, 4.0])

    # a.x[0] = a.x[1] = 0.5
    # a.x[1] = (v.x[2] - v.x[0]) / (t[2] - t[0])
    #        = (4 - 3) / (2 - 0)
    #        = 0.5
    # a.x[2] = (v.x[3] - v.x[1]) / (t[3] - t[1])
    #        = (4 - 3) / (4 - 2)
    #        = 0.5
    # a.x[3] = a.x[2] = 0.5
    assert traj.a.x == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_linear_backward_diff() -> None:
    diff = {
        "method": DiffMethod.LINEAR_DIFF,
        "window_type": WindowType.BACKWARD,
        "accuracy": 1,
    }

    traj = Trajectory(x=[0, 2, 6], diff_est=diff)

    assert traj.v.x == pytest.approx([2.0, 2.0, 4.0])
    assert traj.a.x == pytest.approx([0.0, 0.0, 2.0])


def test_fornberg_forward_diff() -> None:
    diff = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.FORWARD,
        "accuracy": 2,
    }

    traj = Trajectory(x=[0, 2, 6, 10], diff_est=diff)

    assert traj.v.x == pytest.approx([1.0, 4.0, 4.0, 4.0])
    assert traj.a.x == pytest.approx([2.0, 0.0, 0.0, 0.0])


def test_fornberg_central_diff() -> None:
    diff = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 2,
    }

    traj = Trajectory(x=[0, 2, 6, 10], diff_est=diff)

    assert traj.v.x == pytest.approx([1.0, 3.0, 4.0, 4.0])
    assert traj.a.x == pytest.approx([2.0, 2.0, 0.0, 0.0])


def test_fornberg_backward_diff() -> None:
    diff = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.BACKWARD,
        "accuracy": 2,
    }

    traj = Trajectory(x=[0, 2, 6, 10], diff_est=diff)

    assert traj.v.x == pytest.approx([1.0, 3.0, 5.0, 7.0])
    assert traj.a.x == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_wrong_diff_parameters() -> None:
    wrong_method = {
        "method": 5,
        "window_type": WindowType.BACKWARD,
        "accuracy": 2,
    }

    wrong_win = {
        "method": DiffMethod.LINEAR_DIFF,
        "window_type": 4,
    }

    wrong_acc = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.BACKWARD,
        "accuracy": 10,
    }

    wrong_odd_acc = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 3,
    }

    wrong_win_fornberg = {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": 10,
        "accuracy": 2,
    }

    with pytest.raises(ValueError, match="Invalid method to estimate the velocity."):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_method)
        _ = traj.v

    with pytest.raises(
        ValueError, match="Invalid method to estimate the acceleration."
    ):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_method)
        _ = traj.a

    with pytest.raises(ValueError, match="Invalid window type."):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_win)
        _ = traj.v

    with pytest.raises(ValueError, match="Invalid window type."):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_win)
        _ = traj.a

    with pytest.raises(ValueError, match="Invalid window type."):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_win_fornberg)
        _ = traj.v

    with pytest.raises(ValueError, match="Invalid window type."):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_win_fornberg)
        _ = traj.a

    with pytest.raises(ValueError):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_acc)
        _ = traj.v

    with pytest.raises(ValueError):
        traj = Trajectory(x=[0, 2, 6, 10], diff_est=wrong_acc)
        _ = traj.a

    with pytest.raises(ValueError, match="The accuracy must be an EVEN integer"):
        traj = Trajectory(x=[0, 2, 6, 10, 12], diff_est=wrong_odd_acc)
        _ = traj.v

    with pytest.raises(ValueError, match="The accuracy must be an EVEN integer"):
        traj = Trajectory(x=[0, 2, 6, 10, 12], diff_est=wrong_odd_acc)
        _ = traj.a


def test_set_diff_method() -> None:
    traj = Trajectory(x=[0, 2, 6, 10])

    traj.set_diff_method(
        method=DiffMethod.FORNBERG_DIFF,
        window_type=WindowType.CENTRAL,
        accuracy=2,
    )

    assert traj.diff_est == {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 2,
    }

    default = Trajectory.general_diff_est
    Trajectory.global_diff_method(
        method=DiffMethod.FORNBERG_DIFF,
        window_type=WindowType.CENTRAL,
        accuracy=2,
    )

    traj = Trajectory(x=[0, 2, 6, 10])

    assert traj.diff_est == {
        "method": DiffMethod.FORNBERG_DIFF,
        "window_type": WindowType.CENTRAL,
        "accuracy": 2,
    }

    Trajectory.global_diff_method(**default)
