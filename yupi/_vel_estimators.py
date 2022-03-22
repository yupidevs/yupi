import enum
import logging

import numpy as np

from yupi.vector import Vector


class VelocityMethod(enum.Enum):
    """
    Enum to define the method to calculate the velocity.
    """

    LINEAR_DIFF = enum.auto()
    FORNBERG_DIFF = enum.auto()


class WindowType(enum.Enum):
    """
    Enum to define the type of window to use.
    """

    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    CENTRAL = enum.auto()


def coeff(x_0: float, a: np.ndarray, coeff_arr: np.ndarray = None):
    N = len(a)
    M = 2
    _coeff = np.zeros((M, N, N)) if coeff_arr is None else coeff_arr

    _coeff[0, 0, 0] = 1
    c1 = 1
    for n in range(1, N):
        c2 = 1
        for v in range(n):
            c3 = a[n] - a[v]
            c2 = c2 * c3
            if n < M:
                _coeff[n, n - 1, v] = 0

            _a = a[n] - x_0
            _coeff[0, n, v] = (_a * _coeff[0, n - 1, v]) / c3
            _coeff[1, n, v] = (_a * _coeff[1, n - 1, v] - _coeff[0, n - 1, v]) / c3

        _c = c1 / c2
        _a = a[n - 1] - x_0
        _coeff[0, n, n] = _c * _a * _coeff[0, n - 1, n - 1]
        _coeff[1, n, n] = _c * _coeff[1, n - 1, n - 1] - _a * _coeff[1, n - 1, n - 1]

        c1 = c2

    return _coeff


def validate_traj(traj, method, window_type, accuracy):
    l = len(traj)
    if method == VelocityMethod.LINEAR_DIFF:
        return l >= 3 if window_type == WindowType.CENTRAL else l >= 2
    if method == VelocityMethod.FORNBERG_DIFF:
        return l >= accuracy + 1
    raise ValueError("Invalid method to estimate the velocity.")


def _linear_diff(traj, window_type):
    v = np.zeros_like(traj.r)
    if window_type == WindowType.FORWARD:
        diff = ((traj.r[1:] - traj.r[:-1]).T / (traj.t[1:] - traj.t[:-1])).T
        v[:-1] = diff
        v[-1] = diff[-1]
    elif window_type == WindowType.BACKWARD:
        diff = ((traj.r[1:] - traj.r[:-1]).T / (traj.t[1:] - traj.t[:-1])).T
        v[1:] = diff
        v[0] = diff[0]
    elif window_type == WindowType.CENTRAL:
        diff = ((traj.r[2:] - traj.r[:-2]).T / (traj.t[2:] - traj.t[:-2])).T
        v[1:-1] = diff
        v[0] = diff[0]
        v[-1] = diff[-1]
    else:
        raise ValueError("Invalid window type to estimate the velocity.")
    return Vector.create(v)


def _fornberg_diff_forward(traj, window_type, n):
    v = np.zeros_like(traj.r)
    _coeff = None
    a_len = n + 1
    for i in range(len(traj.r)):
        alpha = traj.t[i : i + a_len] if i < len(traj.r) - a_len else traj.t[-a_len:]
        _y = traj.r[i : i + a_len] if i < len(traj.r) - a_len else traj.r[-a_len:]
        _coeff = coeff(traj.t[i], alpha, _coeff)
        v[i] = np.sum(_coeff[1, n, :] * _y.T, axis=1)
    return Vector.create(v)


def _fornberg_diff_backward(traj, window_type, n):
    v = np.zeros_like(traj.r)
    _coeff = None
    a_len = n + 1
    for i in range(len(traj.r)):
        alpha = traj.t[i - a_len : i] if i >= a_len else traj.t[:a_len]
        _y = traj.r[i - a_len : i] if i >= a_len else traj.r[:a_len]
        _coeff = coeff(traj.t[i], alpha, _coeff)
        v[i] = np.sum(_coeff[1, n, :] * _y.T, axis=1)
    return Vector.create(v)


def _fornberg_diff_central(traj, window_type, n):
    v = np.zeros_like(traj.r)
    _coeff = None
    a_len = n + 1
    midd = a_len // 2
    for i in range(len(traj.r)):
        if midd <= i < len(traj.r) - midd:
            alpha = traj.t[i - midd : i + midd + 1]
            _y = traj.r[i - midd : i + midd + 1]
        elif i < midd:
            alpha = traj.t[: i + a_len]
            _y = traj.r[: i + a_len]
        else:
            alpha = traj.t[-a_len:]
            _y = traj.r[-a_len:]
        _coeff = coeff(traj.t[i], alpha, _coeff)
        v[i] = np.sum(_coeff[1, n, :] * _y.T, axis=1)
    return Vector.create(v)


def estimate_velocity(
    traj,
    method: VelocityMethod,
    window_type: WindowType = WindowType.CENTRAL,
    accuracy: int = 1,
):
    """
    Estimate the velocity of a trajectory.

    Parameters
    ----------
    traj : Trajectory
        Trajectory to estimate the velocity.
    method : VelocityMethod
        Method to use to estimate the velocity.
    window_type : WindowType
        Type of window to use.
    accuracy : int
        Accuracy of the estimation (only used if method is FORNBERG_DIFF).

    Returns
    -------
    Vector
        Estimated velocity.

    Raises
    ------
    ValueError
        If the trajectory is too short to estimate the velocity.
    """
    if not validate_traj(traj, method, window_type, accuracy):
        logging.warning("Trajectory is too short to estimate the velocity.")
        return None

    if method == VelocityMethod.LINEAR_DIFF:
        return _linear_diff(traj, window_type)
    elif method == VelocityMethod.FORNBERG_DIFF:
        if window_type == WindowType.FORWARD:
            return _fornberg_diff_forward(traj, window_type, accuracy)
        elif window_type == WindowType.BACKWARD:
            return _fornberg_diff_backward(traj, window_type, accuracy)
        elif window_type == WindowType.CENTRAL:
            if accuracy % 2 != 0:
                raise ValueError(
                    "The accuracy must be an EVEN integer for"
                    " central window type in FORNBERG_DIFF method."
                )
            return _fornberg_diff_central(traj, window_type, accuracy)
        else:
            raise ValueError("Invalid window type to estimate the velocity.")
    else:
        raise ValueError("Invalid method to estimate the velocity.")
