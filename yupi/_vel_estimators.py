import enum

import numpy as np

from yupi.vector import Vector


class VelMethod(enum.Enum):
    """
    Enum to define the method to calculate the velocity.
    """

    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    CENTERED = enum.auto()


class VelPadding(enum.Enum):
    """
    Enum to define the padding method.
    """

    EXTEND = enum.auto()
    VALUE = enum.auto()


def _validate_parameters(kwargs):
    h = kwargs.get("h", 1)
    padd = kwargs.get("padding", VelPadding.EXTEND)
    if h <= 0:
        raise ValueError("h must be positive")
    if not isinstance(h, int):
        raise ValueError("h must be an integer")
    if padd == VelPadding.VALUE and "padding_val" not in kwargs:
        raise ValueError("padding_value must be specified")
    if padd not in VelPadding:
        raise ValueError("padding must be one of Padding")
    kwargs["h"] = h
    kwargs["padding"] = padd


def validate_traj(traj, vel_est):
    l = len(traj)
    h = vel_est.get("h", 1)
    method = vel_est["method"]
    min_len = l - h if method != VelMethod.CENTERED else l - 2 * h
    return min_len >= 1


def forward(traj, **kwargs) -> Vector:
    if len(traj) < 2:
        raise ValueError(
            "Trajectory must have at least 2 points to "
            "calculate velocity using forward method"
        )

    _validate_parameters(kwargs)
    h, padd = kwargs["h"], kwargs["padding"]

    v = Vector.create(np.zeros(tuple(traj.r.shape)))
    dx = traj.r[h:] - traj.r[:-h]
    dt = (traj.t[h:] - traj.t[:-h]).reshape((traj.r.shape[0] - h, 1))
    v[:-h] = dx / dt

    if padd == VelPadding.EXTEND:
        v[-h:] = v[-h - 1]
    elif padd == VelPadding.VALUE:
        v[-h:] = kwargs["padding_val"]
    else:
        raise ValueError("Padding method not supported")
    return v


def backward(traj, **kwargs) -> Vector:
    if len(traj) < 2:
        raise ValueError(
            "Trajectory must have at least 2 points to "
            "calculate velocity using backward method"
        )

    _validate_parameters(kwargs)
    h, padd = kwargs["h"], kwargs["padding"]

    v = Vector.create(np.zeros(tuple(traj.r.shape)))
    dx = traj.r[h:] - traj.r[:-h]
    dt = (traj.t[h:] - traj.t[:-h]).reshape((traj.r.shape[0] - h, 1))
    v[h:] = dx / dt

    if padd == VelPadding.EXTEND:
        v[:h] = v[h]
    elif padd == VelPadding.VALUE:
        v[:h] = kwargs["padding_val"]
    else:
        raise ValueError("Padding method not supported")
    return v


def centered(traj, **kwargs) -> Vector:
    if len(traj) < 3:
        raise ValueError(
            "Trajectory must have at least 3 points to "
            "calculate velocity using centered method"
        )
    _validate_parameters(kwargs)
    h, padd = kwargs["h"], kwargs["padding"]
    h2 = h * 2

    v = Vector.create(np.zeros(tuple(traj.r.shape)))
    dx = traj.r[h2:] - traj.r[:-h2]
    dt = (traj.t[h2:] - traj.t[:-h2]).reshape((traj.r.shape[0] - h2, 1))
    v[h:-h] = dx / dt

    if padd == VelPadding.EXTEND:
        v[:h] = v[h]
        v[-h:] = v[-h - 1]
    elif padd == VelPadding.VALUE:
        val = kwargs["padding_val"]
        if not isinstance(val, tuple):
            raise ValueError(
                "padding_val must be a tuple of length 2 for centered method"
            )
        v[:h] = val[0]
        v[-h:] = val[1]
    else:
        raise ValueError("Padding method not supported")
    return v


FUNCTIONS = {
    VelMethod.FORWARD: forward,
    VelMethod.BACKWARD: backward,
    VelMethod.CENTERED: centered,
}
